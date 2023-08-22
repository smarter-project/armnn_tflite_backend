//
// Copyright © 2023 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "tflite_utils.h"

#include <sstream>

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

std::vector<int>
StringToIntVector(std::string const& s)
{
  std::stringstream iss(s);

  int val;
  std::vector<int> result;
  while (iss >> val) {
    result.push_back(val);
  }
  return result;
}

void
PopulateCpusMap(std::unordered_map<int, std::vector<int>>& cpus)
{
  hwloc_topology_t topology;
  hwloc_topology_init(&topology);
  hwloc_topology_load(topology);

  int num_logical_cpus = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);
  int smt_threads_per_core =
      num_logical_cpus / hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);
  for (int cpu_id = 0; cpu_id < num_logical_cpus; ++cpu_id) {
    hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, cpu_id);
    if (obj) {
      hwloc_bitmap_t nodeset = obj->nodeset;
      if (cpu_id % smt_threads_per_core) {
        cpus[hwloc_bitmap_first(nodeset)].push_back(cpu_id);
      } else {
        cpus[hwloc_bitmap_first(nodeset)].insert(
            cpus[hwloc_bitmap_first(nodeset)].begin(), cpu_id);
      }
    }
  }
  hwloc_topology_destroy(topology);
}


}}}  // namespace triton::backend::tensorflowlite
