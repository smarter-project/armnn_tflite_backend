//
// Copyright © 2021 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "tensorflow/lite/model.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace backend { namespace tensorflowlite {

#define RESPOND_ALL_AND_RETURN_IF_ERROR(RESPONSES, RESPONSES_COUNT, X) \
  do {                                                                 \
    TRITONSERVER_Error* raarie_err__ = (X);                            \
    if (raarie_err__ != nullptr) {                                     \
      SendErrorForResponses(RESPONSES, RESPONSES_COUNT, raarie_err__); \
      return;                                                          \
    }                                                                  \
  } while (false)

TRITONSERVER_DataType ConvertTFLiteTypeToDataType(const TfLiteType ttype);
std::pair<bool, TfLiteType> ConvertDataTypeToTFLiteType(
    const TRITONSERVER_DataType dtype);
std::pair<bool, TfLiteType> ModelConfigDataTypeToTFLiteType(
    const std::string& data_type_str);

}}}  // namespace triton::backend::tensorflowlite
