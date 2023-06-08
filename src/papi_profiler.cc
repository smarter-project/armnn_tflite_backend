//
// Copyright © 2023 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "papi_profiler.h"

#include <stdio.h>
#include <stdlib.h>
#include <tensorflow/lite/core/api/profiler.h>

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
  exit(1);
}

class PapiProfiler : public tflite::Profiler {
 public:
  PapiProfiler(const std::vector<std::string>& papi_events)
      : supported_event_types_(
            static_cast<uint64_t>(EventType::DELEGATE_OPERATOR_INVOKE_EVENT) +
            static_cast<uint64_t>(EventType::OPERATOR_INVOKE_EVENT)),
        papi_events_(papi_events)
  {
    // We only care about the 4th thread in the process on, as these are used
    // for inference
    std::vector<pid_t> current_threads = CurrentThreadIds();
    inf_thread_ids_ =
        std::vector<pid_t>(current_threads.begin() + 3, current_threads.end());

    int retval;
    // The first 3 threads for the model instance don't do anything for
    // inference, so we aren't interested in them
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
    }
    event_values_.resize(papi_events_.size());
  }

  ~PapiProfiler()
  {
    // Save results to file
    std::ofstream myfile;
    myfile.open("counters.csv");
    // Header
    myfile << "op_id,thread_id,papi_event,value\n";
    // Iterate over map keyed on tflite operation id
    for (auto& event : results_) {
      for (uint64_t i = 0; i < event.second.size(); ++i) {
        myfile << event.first << ","
               << inf_thread_ids_[i / papi_events_.size() % event_sets_.size()]
               << "," << papi_events_[i % papi_events_.size()] << ","
               << event.second[i] << "\n";
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
    // For the event set attached to each thread, start or restart the event set
    for (uint64_t i = 0; i < event_sets_.size(); ++i) {
      int state;
      PAPI_state(event_sets_[i], &state);
      if (!(state & PAPI_RUNNING)) {
        // Begin tracking counters
        retval = PAPI_start(event_sets_[i]);
        if (retval != PAPI_OK)
          handle_error(retval, __LINE__, __FILE__);

      } else {
        // Reset counters
        retval = PAPI_reset(event_sets_[i]);
        if (retval != PAPI_OK)
          handle_error(retval, __LINE__, __FILE__);
      }
    }

    uint32_t event_handle = event_index_++;
    papi_regions_[event_handle] = trace_event_tag;
    return event_handle;
  }

  void EndEvent(uint32_t event_handle) override
  {
    if (event_handle == kInvalidEventHandle) {
      return;
    }

    int retval;
    // For each thread we are profiling
    for (uint64_t i = 0; i < event_sets_.size(); ++i) {
      retval = PAPI_read(event_sets_[i], event_values_.data());
      if (retval != PAPI_OK)
        handle_error(retval, __LINE__, __FILE__);
      // For each of the events we collected a counter value for
      for (auto val : event_values_) {
        results_[papi_regions_[event_handle]].push_back(val);
      }
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
  const uint64_t supported_event_types_;
  std::vector<std::string> papi_events_;
  std::vector<int> event_sets_;

  // We only care about the 4th thread in the process on, as these are used for
  // inference
  std::vector<pid_t> inf_thread_ids_;

  std::vector<long long> event_values_;
  std::unordered_map<std::string, std::vector<long long>> results_;
};

std::unique_ptr<tflite::Profiler>
MaybeCreatePapiProfiler()
{
  char* papi_events = getenv("PAPI_EVENTS");
  std::vector<std::string> papi_events_vec;
  if (papi_events == NULL) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN,
        "PAPI_EVENTS not specified, op level profiling disabled!");
    return nullptr;
  } else {
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
  return std::unique_ptr<tflite::Profiler>(new PapiProfiler(papi_events_vec));
}
