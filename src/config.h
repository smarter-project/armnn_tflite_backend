//
// Copyright © 2023 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

// This class is used to map an optimizer option to an index in an array so
// options can be sent across a tensorpipe payload
enum OptimizerOption {
  TFLITE_NUM_THREADS,
  XNNPACK_ENABLE,
  XNNPACK_CPU_NUM_THREADS,

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