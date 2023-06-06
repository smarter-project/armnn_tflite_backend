//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <filesystem>
#include <vector>

#ifdef PAPI_PROFILING_ENABLE
#include "papi.h"

bool
PAPIEventValid(std::string& event_name)
{
  int event_set = PAPI_NULL;
  bool valid = false;
  if (PAPI_create_eventset(&event_set) == PAPI_OK) {
    valid = PAPI_add_named_event(event_set, event_name.c_str()) == PAPI_OK;
    if (valid) {
      if (PAPI_cleanup_eventset(event_set) != PAPI_OK) {
      }
    }
    if (PAPI_destroy_eventset(&event_set) != PAPI_OK) {
    }
  }
  return valid;
}
#endif  // PAPI_PROFILING_ENABLE

std::vector<pid_t>
CurrentThreadIds()
{
  std::vector<pid_t> r;
  for (auto& p : std::filesystem::directory_iterator("/proc/self/task")) {
    if (p.is_directory()) {
      r.push_back(std::stoi(p.path().filename().string()));
    }
  }
  return r;
}