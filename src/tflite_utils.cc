#include "tflite_utils.h"

namespace triton { namespace backend { namespace tensorflowlite {

TRITONSERVER_DataType
ConvertTFLiteTypeToDataType(const TfLiteType ttype)
{
  switch (ttype) {
    case kTfLiteBool:
      return TRITONSERVER_TYPE_BOOL;
    case kTfLiteUInt8:
      return TRITONSERVER_TYPE_UINT8;
    case kTfLiteInt8:
      return TRITONSERVER_TYPE_INT8;
    case kTfLiteInt16:
      return TRITONSERVER_TYPE_INT16;
    case kTfLiteInt32:
      return TRITONSERVER_TYPE_INT32;
    case kTfLiteInt64:
      return TRITONSERVER_TYPE_INT64;
    case kTfLiteFloat16:
      return TRITONSERVER_TYPE_FP16;
    case kTfLiteFloat32:
      return TRITONSERVER_TYPE_FP32;
    case kTfLiteFloat64:
      return TRITONSERVER_TYPE_FP64;
    default:
      break;
  }

  return TRITONSERVER_TYPE_INVALID;
}

std::pair<bool, TfLiteType>
ConvertDataTypeToTFLiteType(const TRITONSERVER_DataType dtype)
{
  TfLiteType type = kTfLiteInt32;
  switch (dtype) {
    case TRITONSERVER_TYPE_BOOL:
      type = kTfLiteBool;
      break;
    case TRITONSERVER_TYPE_UINT8:
      type = kTfLiteUInt8;
      break;
    case TRITONSERVER_TYPE_INT8:
      type = kTfLiteInt8;
      break;
    case TRITONSERVER_TYPE_INT16:
      type = kTfLiteInt16;
      break;
    case TRITONSERVER_TYPE_INT32:
      type = kTfLiteInt32;
      break;
    case TRITONSERVER_TYPE_INT64:
      type = kTfLiteInt64;
      break;
    case TRITONSERVER_TYPE_FP16:
      type = kTfLiteFloat16;
      break;
    case TRITONSERVER_TYPE_FP32:
      type = kTfLiteFloat32;
      break;
    case TRITONSERVER_TYPE_FP64:
      type = kTfLiteFloat64;
      break;
    case TRITONSERVER_TYPE_BYTES:
    case TRITONSERVER_TYPE_UINT16:
    case TRITONSERVER_TYPE_UINT32:
    case TRITONSERVER_TYPE_UINT64:
    default:
      return std::make_pair(false, type);
  }

  return std::make_pair(true, type);
}

std::pair<bool, TfLiteType>
ModelConfigDataTypeToTFLiteType(const std::string& data_type_str)
{
  TfLiteType type = kTfLiteInt32;

  // Must start with "TYPE_".
  if (data_type_str.rfind("TYPE_", 0) != 0) {
    return std::make_pair(false, type);
  }

  const std::string dtype = data_type_str.substr(strlen("TYPE_"));

  if (dtype == "BOOL") {
    type = kTfLiteBool;
  } else if (dtype == "UINT8") {
    type = kTfLiteUInt8;
  } else if (dtype == "INT8") {
    type = kTfLiteInt8;
  } else if (dtype == "INT16") {
    type = kTfLiteInt16;
  } else if (dtype == "INT32") {
    type = kTfLiteInt32;
  } else if (dtype == "INT64") {
    type = kTfLiteInt64;
  } else if (dtype == "FP16") {
    type = kTfLiteFloat16;
  } else if (dtype == "FP32") {
    type = kTfLiteFloat32;
  } else if (dtype == "FP64") {
    type = kTfLiteFloat64;
  } else {
    return std::make_pair(false, type);
  }

  return std::make_pair(true, type);
}

}}}  // namespace triton::backend::tensorflowlite
