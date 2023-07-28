//
// Copyright © 2023 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "papi_profiler.h"

#include <stdio.h>
#include <stdlib.h>
#include <tensorflow/lite/core/api/profiler.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

// Triton backend headers
#include "model_instance_utils.h"
#include "papi.h"
#include "triton/backend/backend_common.h"

constexpr uint32_t kInvalidEventHandle = static_cast<uint32_t>(~0) - 1;

void
handle_error(int retval, int line, const std::string& file)
{
  LOG_MESSAGE(
      TRITONSERVER_LOG_ERROR,
      ("PAPI error at line " + file + ":" + std::to_string(line) + " " +
       std::to_string(retval) + ", " + PAPI_strerror(retval))
          .c_str());

  // TODO: graceful exit here
  exit(1);
}

class PapiProfiler : public tflite::Profiler {
 public:
  PapiProfiler(
      const std::vector<std::string>& papi_events,
      const std::vector<std::string>& papi_uncore_events)
      : supported_event_types_(
            static_cast<uint64_t>(EventType::DELEGATE_OPERATOR_INVOKE_EVENT) +
            static_cast<uint64_t>(EventType::OPERATOR_INVOKE_EVENT)),
        papi_events_(papi_events), papi_uncore_events_(papi_uncore_events)
  {
    // We only care about the 4th thread in the process on, as these are used
    // for inference
    std::vector<pid_t> current_threads = CurrentThreadIds();
    inf_thread_ids_ =
        std::vector<pid_t>(current_threads.begin() + 3, current_threads.end());

    papi_regions_.reserve(1000);
    timings_.reserve(1000);

    int retval;

    // Handle core specific events per inference thread
    if (!papi_events_.empty()) {
      for (uint64_t i = 0; i < inf_thread_ids_.size(); ++i) {
        event_sets_.push_back(PAPI_NULL);
        retval = PAPI_create_eventset(&event_sets_.back());
        if (retval != PAPI_OK) {
          handle_error(retval, __LINE__, __FILE__);
        }
        for (auto& event_name : papi_events_) {
          retval = PAPI_add_named_event(event_sets_.back(), event_name.c_str());
          if (retval != PAPI_OK)
            handle_error(retval, __LINE__, __FILE__);
        }

        // Attach event to thread
        LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            ("Attaching to " + std::to_string(inf_thread_ids_[i])).c_str());
        retval = PAPI_attach(event_sets_.back(), inf_thread_ids_[i]);
        if (retval != PAPI_OK)
          handle_error(retval, __LINE__, __FILE__);

        // Start eventset
        retval = PAPI_start(event_sets_.back());
        if (retval != PAPI_OK)
          handle_error(retval, __LINE__, __FILE__);
      }
      event_values_.resize(papi_events_.size());

      // Separately we will also track operation timings in nanos
      papi_events_.push_back("TIME_NS");
    }

    // Handle uncore events separately
    if (!papi_uncore_events_.empty()) {
      retval = PAPI_create_eventset(&uncore_event_set_);
      if (retval != PAPI_OK) {
        handle_error(retval, __LINE__, __FILE__);
      }
      for (auto& event_name : papi_uncore_events_) {
        retval = PAPI_add_named_event(uncore_event_set_, event_name.c_str());
        if (retval != PAPI_OK)
          handle_error(retval, __LINE__, __FILE__);
      }
      uncore_event_values_.resize(papi_uncore_events_.size());
      // Start uncore eventset
      retval = PAPI_start(uncore_event_set_);
      if (retval != PAPI_OK)
        handle_error(retval, __LINE__, __FILE__);
    }
  }

  ~PapiProfiler()
  {
    // Save results to file
    std::ofstream myfile;
    auto now = std::chrono::system_clock::now();
    auto utc =
        std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch())
            .count();

    myfile.open(("counters_" + std::to_string(utc) + ".csv").c_str());
    // Header
    myfile << "op_id,thread_id,papi_event,value\n";
    // Iterate over map keyed on tflite operation id, with values being a vector
    // of counter values for each tracked perf event
    pid_t inf_thread_id;
    for (auto& event : results_) {
      // Write all of the per-core events first, broken down by thread
      for (uint64_t i = 0; i < event.second.size(); ++i) {
        inf_thread_id =
            inf_thread_ids_[i / papi_events_.size() % event_sets_.size()];
        myfile << event.first << "," << inf_thread_id << ","
               << papi_events_[i % papi_events_.size()] << ","
               << event.second[i] << "\n";
      }
    }
    for (auto& event : results_uncore_) {
      // Now write the uncore events with a dummy thread id of -1
      for (uint64_t i = 0; i < results_uncore_[event.first].size(); ++i) {
        myfile << event.first << "," << -1 << ","
               << papi_uncore_events_[i % papi_uncore_events_.size()] << ","
               << results_uncore_[event.first][i] << "\n";
      }
    }
    myfile.close();

    for (auto& event_set : event_sets_) {
      PAPI_cleanup_eventset(event_set);
      PAPI_destroy_eventset(&event_set);
    }
  }

  // This function wants to return a handle to the profile event, which seems to
  // be a unique value. Because we are interested in the op names, we just has
  // the op tag to generate the event handle value.
  // In the case of and Op event, metadata1 holds the node index, and metadata 2
  // holds the subgraph index
  uint32_t BeginEvent(
      const char* tag, EventType event_type, int64_t event_metadata1,
      int64_t event_metadata2) override
  {
    if (!ShouldAddEvent(event_type)) {
      return kInvalidEventHandle;
    }

    // Get a unique name for the papi computation region
    std::string trace_event_tag = tag;
    trace_event_tag += ("_" + std::to_string(event_metadata1));

    int retval;

    if (!papi_events_.empty()) {  // Reset event set attached to each thread
      for (uint64_t i = 0; i < event_sets_.size(); ++i) {
        // Reset counters
        retval = PAPI_reset(event_sets_[i]);
        if (retval != PAPI_OK)
          handle_error(retval, __LINE__, __FILE__);
      }
    }

    // Handle uncore events
    if (!papi_uncore_events_.empty()) {
      // Reset counters
      retval = PAPI_reset(uncore_event_set_);
      if (retval != PAPI_OK)
        handle_error(retval, __LINE__, __FILE__);
    }

    event_index_++;
    papi_regions_[event_index_] = std::move(trace_event_tag);
    timings_[event_index_] = PAPI_get_real_nsec();
    return event_index_;
  }

  void EndEvent(uint32_t event_handle) override
  {
    if (event_handle == kInvalidEventHandle) {
      return;
    }

    long long op_latency = PAPI_get_real_nsec() - timings_[event_handle];

    // For performance reserve space for 10000 elements for each perf event in
    // results
    if (results_[papi_regions_[event_handle]].empty()) {
      results_[papi_regions_[event_handle]].reserve(
          papi_events_.size() * 10000);
    }
    if (results_uncore_[papi_regions_[event_handle]].empty()) {
      results_uncore_[papi_regions_[event_handle]].reserve(
          papi_uncore_events_.size() * 10000);
    }

    int retval;

    if (!papi_events_.empty()) {  // For each thread we are profiling
      for (uint64_t i = 0; i < event_sets_.size(); ++i) {
        retval = PAPI_read(event_sets_[i], event_values_.data());
        if (retval != PAPI_OK)
          handle_error(retval, __LINE__, __FILE__);
        // Write event counter values to end of results vector for current op
        results_[papi_regions_[event_handle]].insert(
            results_[papi_regions_[event_handle]].end(), event_values_.begin(),
            event_values_.end());
      }

      // Push back the op timing
      results_[papi_regions_[event_handle]].push_back(op_latency);
    }
    // Handle uncore events
    if (!papi_uncore_events_.empty()) {
      retval = PAPI_read(uncore_event_set_, uncore_event_values_.data());
      if (retval != PAPI_OK)
        handle_error(retval, __LINE__, __FILE__);
      // For each of the events we collected a counter value for
      results_uncore_[papi_regions_[event_handle]].insert(
          results_uncore_[papi_regions_[event_handle]].end(),
          uncore_event_values_.begin(), uncore_event_values_.end());
    }
  }

 protected:
  inline bool ShouldAddEvent(EventType event_type)
  {
    return (static_cast<uint64_t>(event_type) & supported_event_types_) != 0;
  }

 private:
  uint32_t event_index_ = 0;
  std::unordered_map<uint32_t, std::string> papi_regions_;
  std::unordered_map<uint32_t, long long> timings_;
  const uint64_t supported_event_types_;

  // Vector holding the papi event names we are tracking for each core/thread
  std::vector<std::string> papi_events_;

  // Vector holding the papi event names we are tracking which are socket
  // specific
  std::vector<std::string> papi_uncore_events_;

  // Vector holding papi event set data structures (one per tracked inf thread)
  std::vector<int> event_sets_;

  // Vector holding papi event set data structures for our uncore events because
  // this is per socket, we only need one event set
  int uncore_event_set_ = PAPI_NULL;

  // We only care about the 4th thread in the process on, as these are used for
  // inference
  std::vector<pid_t> inf_thread_ids_;

  // Vector to hold papi counter values when we read them
  std::vector<long long> event_values_;

  // Vector to hold papi uncore values when we read them
  std::vector<long long> uncore_event_values_;

  // Vector holding all per core counter values to be processed at end
  std::unordered_map<std::string, std::vector<long long>> results_;

  // Vector holding all per core counter values to be processed at end
  std::unordered_map<std::string, std::vector<long long>> results_uncore_;
};

std::unique_ptr<tflite::Profiler>
MaybeCreatePapiProfiler()
{
  // Per core events
  char* papi_events = getenv("PAPI_EVENTS");
  std::vector<std::string> papi_events_vec;
  if (papi_events != NULL) {
    // Parse out all papi events indivdually
    std::stringstream ss(papi_events);
    while (ss.good()) {
      std::string substr;
      std::getline(ss, substr, ',');
      if (!PAPIEventValid(substr)) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_WARN,
            ("Event: " + substr + " invalid, op level profiling disabled!")
                .c_str());
        return nullptr;
      }
      papi_events_vec.push_back(substr);
    }
  }

  // Uncore events
  char* papi_uncore_events = getenv("PAPI_UNCORE_EVENTS");
  std::vector<std::string> papi_uncore_events_vec;
  if (papi_uncore_events != NULL) {
    // Parse out all papi events indivdually
    std::stringstream ss(papi_uncore_events);
    while (ss.good()) {
      std::string substr;
      std::getline(ss, substr, ',');
      if (!PAPIEventValid(substr)) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_WARN,
            ("Event: " + substr + " invalid, op level profiling disabled!")
                .c_str());
        return nullptr;
      }
      papi_uncore_events_vec.push_back(substr);
    }
  }

  if ((papi_events == NULL) && (papi_uncore_events == NULL)) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN,
        "PAPI_EVENTS nor PAPI_UNCORE_EVENTS specified, op level profiling "
        "disabled!");
    return nullptr;
  }

  return std::unique_ptr<tflite::Profiler>(
      new PapiProfiler(papi_events_vec, papi_uncore_events_vec));
}
