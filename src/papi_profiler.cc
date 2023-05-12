//
// Copyright © 2023 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "papi_profiler.h"

#include <papi.h>
#include <stdio.h>
#include <stdlib.h>
#include <tensorflow/lite/core/api/profiler.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "triton/backend/backend_model.h"

constexpr uint32_t kInvalidEventHandle = static_cast<uint32_t>(~0) - 1;

void
handle_error(int retval)
{
  throw triton::backend::BackendModelException(TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INTERNAL, ("PAPI error " + std::to_string(retval) +
                                    std::string(PAPI_strerror(retval)))
                                       .c_str()));
}

class PapiProfiler : public tflite::Profiler {
 public:
  PapiProfiler()
      : supported_event_types_(
            static_cast<uint64_t>(EventType::DELEGATE_OPERATOR_INVOKE_EVENT) +
            static_cast<uint64_t>(EventType::OPERATOR_INVOKE_EVENT))
  {
  }

  ~PapiProfiler() { PAPI_hl_stop(); }

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

    // Begin tracking counters
    int retval = PAPI_hl_region_begin(trace_event_tag.c_str());
    if (retval != PAPI_OK)
      handle_error(retval);

    uint32_t event_handle = event_index_++;
    papi_regions_[event_handle] = trace_event_tag;
    return event_handle;
  }

  void EndEvent(uint32_t event_handle) override
  {
    if (event_handle == kInvalidEventHandle) {
      return;
    }
    int retval = PAPI_hl_region_end(papi_regions_[event_handle].c_str());
    if (retval != PAPI_OK)
      handle_error(retval);
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
};

std::unique_ptr<tflite::Profiler>
MaybeCreatePapiProfiler()
{
  if (getenv("PAPI_EVENTS") == NULL) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        "PAPI_EVENTS not specified, op level profiling disabled");
    return nullptr;
  } else {
    return std::unique_ptr<tflite::Profiler>(new PapiProfiler());
  }
}
