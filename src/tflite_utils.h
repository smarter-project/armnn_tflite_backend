//
// Copyright © 2021 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <sstream>

#include "tensorflow/lite/model.h"
#include "triton/backend/backend_model.h"
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

std::vector<int> StringToIntVector(std::string const& s);

template <typename T, typename A>
std::string
VectorToString(std::vector<T, A> const& v)
{
  std::stringstream ss;
  for (size_t i = 0; i < v.size(); i++) {
    if (i != 0) {
      ss << ",";
    }
    ss << v[i];
  }
  return ss.str();
}

}}}  // namespace triton::backend::tensorflowlite
