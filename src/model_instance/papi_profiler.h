//
// Copyright © 2023 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <memory>

#include "tensorflow/lite/core/api/profiler.h"


// Creates a profiler which reports the papi traced events.
// Triton error will be returned if the there's an issue with the papi lib.
std::unique_ptr<tflite::Profiler> MaybeCreatePapiProfiler();