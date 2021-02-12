// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "armnn_utils.h"

namespace triton { namespace backend { namespace armnn {

TRITONSERVER_DataType
ConvertArmNNTypeToDataType(const armnn::DataType& ttype)
{
  switch (ttype) {
    case armnn::DataType::Boolean:
      return TRITONSERVER_TYPE_BOOL;
    case armnn::DataType::Float16:
      return TRITONSERVER_TYPE_FP16;
    case armnn::DataType::Float32:
      return TRITONSERVER_TYPE_FP32;
    case armnn::DataType::Signed64;
      return TRITONSERVER_TYPE_INT64;
    default:
      break;
  }

  return TRITONSERVER_TYPE_INVALID;
}

std::pair<bool, armnn::DataType>
ConvertDataTypeToArmNNType(const TRITONSERVER_DataType dtype)
{
  armnn::DataType type = armnn::DataType::Signed32;
  switch (dtype) {
    case TRITONSERVER_TYPE_BOOL:
      type = armnn::DataType::Boolean;
      break;
    case TRITONSERVER_TYPE_INT32:
      type = armnn::DataType::Signed32;
      break;
    case TRITONSERVER_TYPE_INT64:
      type = armnn::DataType::Signed64;
      break;
    case TRITONSERVER_TYPE_FP16:
      type = armnn::DataType::Float16;
      break;
    case TRITONSERVER_TYPE_FP32:
      type = armnn::DataType::Float32;
      break;
    case TRITONSERVER_TYPE_UINT8:
    case TRITONSERVER_TYPE_INT8:
    case TRITONSERVER_TYPE_INT16:
    case TRITONSERVER_TYPE_UINT16:
    case TRITONSERVER_TYPE_UINT32:
    case TRITONSERVER_TYPE_UINT64:
    case TRITONSERVER_TYPE_FP64:
    case TRITONSERVER_TYPE_STRING:
    default:
      return std::make_pair(false, type);
  }

  return std::make_pair(true, type);
}

std::pair<bool, armnn::DataType>
ModelConfigDataTypeToArmNNType(const std::string& data_type_str)
{
  armnn::DataType type = armnn::DataType::Signed32;

  // Must start with "TYPE_".
  if (data_type_str.rfind("TYPE_", 0) != 0) {
    return std::make_pair(false, type);
  }

  const std::string dtype = data_type_str.substr(strlen("TYPE_"));

  if (dtype == "BOOL") {
    type = armnn::DataType::Boolean;
  } else if (dtype == "INT32") {
    type = armnn::DataType::Signed32;
  } else if (dtype == "INT64") {
    type = armnn::DataType::Signed64;
  } else if (dtype == "FP16") {
    type = armnn::DataType::Float16;
  } else if (dtype == "FP32") {
    type = armnn::DataType::Float32;
  } else {
    return std::make_pair(false, type);
  }

  return std::make_pair(true, type);
}

}}}  // namespace triton::backend::armnn
