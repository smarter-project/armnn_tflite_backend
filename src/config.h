//
// Copyright © 2023 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <algorithm>
#include <string>

// This class is used to map an optimizer option to an index in an array so
// options can be sent across a tensorpipe payload
enum OptimizerOption {
  TFLITE_NUM_THREADS,
  XNNPACK_ENABLE,
  XNNPACK_CPU_NUM_THREADS,
  NUMA_ALLOC_POLICY,
  NUMA_LOCAL_NODE_ID,
  NUMA_REMOTE_NODE_ID,

#ifdef ARMNN_DELEGATE_ENABLE
  ARMNN_CPU_ENABLE,
  ARMNN_GPU_ENABLE,
  ARMNN_CPU_NUM_THREADS,
  ARMNN_CPU_REDUCE_FP32_TO_FP16,
  ARMNN_CPU_REDUCE_FP32_TO_BF16,
  ARMNN_CPU_FAST_MATH_ENABLED,
  ARMNN_GPU_FAST_MATH_ENABLED,
  ARMNN_GPU_REDUCE_FP32_TO_FP16,
  ARMNN_GPU_REDUCE_FP32_TO_BF16,
#endif  // ARMNN_DELEGATE_ENABLE

  COUNT  // Just used to track the number of options
};

enum class AllocationPolicy {
  LOCAL,
  WEIGHT_REMOTE_RESULT_LOCAL,
  WEIGHT_LOCAL_RESULT_REMOTE,
  REMOTE,
  NONE
};

inline AllocationPolicy
AllocationPolicyFromString(std::string str)
{
  // Convert copy of string to uppercase
  std::transform(str.begin(), str.end(), str.begin(), ::toupper);

  if (str == "LOCAL") {
    return AllocationPolicy::LOCAL;
  } else if (str == "WEIGHT_REMOTE_RESULT_LOCAL") {
    return AllocationPolicy::WEIGHT_REMOTE_RESULT_LOCAL;
  } else if (str == "WEIGHT_LOCAL_RESULT_REMOTE") {
    return AllocationPolicy::WEIGHT_LOCAL_RESULT_REMOTE;
  } else if (str == "REMOTE") {
    return AllocationPolicy::REMOTE;
  } else if (str == "NONE") {
    return AllocationPolicy::NONE;
  } else {
    return AllocationPolicy::NONE;
  }
}

inline std::string
AllocationPolicyToString(const AllocationPolicy& alloc_policy)
{
  switch (alloc_policy) {
    case AllocationPolicy::LOCAL: {
      return "LOCAL";
    }
    case AllocationPolicy::WEIGHT_REMOTE_RESULT_LOCAL: {
      return "WEIGHT_REMOTE_RESULT_LOCAL";
    }
    case AllocationPolicy::WEIGHT_LOCAL_RESULT_REMOTE: {
      return "WEIGHT_LOCAL_RESULT_REMOTE";
    }
    case AllocationPolicy::REMOTE: {
      return "REMOTE";
    }
    case AllocationPolicy::NONE: {
      return "NONE";
    }
  }
  return "NONE";
}