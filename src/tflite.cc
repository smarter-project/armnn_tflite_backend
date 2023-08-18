//
// Copyright © 2023 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <link.h>
#include <stdint.h>

#include <algorithm>
#include <exception>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Local headers
#include "config.h"
#include "tflite_utils.h"

// Tensorpipe headers
#include "tensorpipe/tensorpipe.h"

// Triton headers
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonbackend.h"

// TFLite headers
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/type_to_tflitetype.h"

// Reproc headers
#include "reproc++/reproc.hpp"

#ifdef LIBNUMA_ENABLE
// Lib Numa headers
#include <numa.h>
#include <numaif.h>
#endif  // LIBNUMA_ENABLE

//
// TFLite Backend that implements the TRITONBACKEND API.
//

namespace triton { namespace backend { namespace tensorflowlite {

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  ~ModelState();

  TRITONSERVER_Error* LoadModel();

  TRITONSERVER_Error* InitConfig();

  // Validate that model configuration is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

  void InitTensorPipe();

  // Default TFLite runtime options
  int32_t tflite_num_threads_ =
      static_cast<int32_t>(std::thread::hardware_concurrency());

#ifdef ARMNN_DELEGATE_ENABLE
  // ArmNN Delegate options
  bool use_armnn_delegate_cpu_ = false;
  bool use_armnn_delegate_gpu_ = false;

  int32_t armnn_cpu_num_threads_ =
      static_cast<int32_t>(std::thread::hardware_concurrency());
  std::string armnn_cpu_reduce_fp32_to_fp16_ = "off";
  std::string armnn_cpu_reduce_fp32_to_bf16_ = "off";
  std::string armnn_cpu_fast_math_enabled_ = "off";

  std::string armnn_gpu_fast_math_enabled_ = "off";
  std::string armnn_gpu_reduce_fp32_to_fp16_ = "off";
  std::string armnn_gpu_reduce_fp32_to_bf16_ = "off";
#endif  // ARMNN_DELEGATE_ENABLE

  // XNNPACK Delegate options
  bool use_xnnpack_delegate_ = false;
  int32_t num_threads_xnnpack_ =
      static_cast<int32_t>(std::thread::hardware_concurrency());

  // Map from configuration name for an input to the index of
  // that input in the model.
  std::unordered_map<std::string, int> input_index_map_;
  std::unordered_map<std::string, TRITONSERVER_DataType> input_dtype_map_;

  // Map from configuration name for an output to the index of
  // that output in the model.
  std::unordered_map<std::string, int> output_index_map_;
  std::unordered_map<std::string, TRITONSERVER_DataType> output_dtype_map_;
  std::unordered_map<std::string, std::vector<int64_t>> output_shape_map_;

  // The pointer to the tflite network
  std::unique_ptr<tflite::FlatBufferModel> model_;

  // Global context of tensorpipe
  std::shared_ptr<tensorpipe::Context> context_;

  // Path string for the model_instance binary
  const char* model_instance_location_;

#ifdef PAPI_PROFILING_ENABLE
  // String holding comma-separated list of events for child inference process
  std::string papi_events_ = "";

  // String holding comma-separated list of uncore events for child inference
  // process
  std::string papi_uncore_events_ = "";
#endif  // PAPI_PROFILING_ENABLE

  // Numa policy for instance
  AllocationPolicy numa_alloc_policy_ = AllocationPolicy::NONE;

  // Local numa node id
  int local_numa_node_id_ = 0;

  // remote numa node id
  int remote_numa_node_id_ = 1;

  // Map managing list of avail cpus in system, keyed on socket
  std::unordered_map<int, std::vector<int>> avail_cpus_;

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  TRITONSERVER_Error* AutoCompleteConfig();
};


TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model)
{  // Here we can add information to the model state that can be shared across
  // model instances. See onnx backend for example. MALI GPU optimization level
  // may be candidate.
  InitTensorPipe();
  THROW_IF_BACKEND_MODEL_ERROR(InitConfig());
  THROW_IF_BACKEND_MODEL_ERROR(LoadModel());

  PopulateCpusMap(avail_cpus_);

  // Get the directory of the backend to find the path to the model instance
  // binary
  TRITONBACKEND_Backend* backend;
  TRITONBACKEND_ArtifactType artifact_type;
  TRITONBACKEND_ModelBackend(triton_model, &backend);
  TRITONBACKEND_BackendArtifacts(
      backend, &artifact_type, &model_instance_location_);
}

ModelState::~ModelState() {}

TRITONSERVER_Error*
ModelState::InitConfig()
{
  // Handle tflite default interpeter options set in parameters
  {
    triton::common::TritonJson::Value params;
    if (ModelConfig().Find("parameters", &params)) {
      // Handle tflite_num_threads parameter
      std::string value_str;
      auto err = GetParameterValue(params, "tflite_num_threads", &value_str);

      // tflite_num_threads is not required so clear error if not found
      if (err != nullptr) {
        if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
          return err;
        } else {
          TRITONSERVER_ErrorDelete(err);
        }
      } else {
        RETURN_IF_ERROR(ParseIntValue(value_str, &tflite_num_threads_));
        if (tflite_num_threads_ < 0) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string(
                   "parameter 'tflite_num_threads' must be non-negative "
                   "number for tflite model '") +
               Name() + "'")
                  .c_str());
        }
      }

      // Handle numa parameters
      err = GetParameterValue(params, "numa_alloc_policy", &value_str);

      // numa_alloc_policy is not required so clear error if not found
      if (err != nullptr) {
        if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
          return err;
        } else {
          TRITONSERVER_ErrorDelete(err);
        }
      } else {
        numa_alloc_policy_ = AllocationPolicyFromString(value_str);
      }

#ifdef LIBNUMA_ENABLE
      err = GetParameterValue(params, "local_numa_node_id", &value_str);

      // local_numa_node_id is not required so clear error if not found
      if (err != nullptr) {
        if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
          return err;
        } else {
          TRITONSERVER_ErrorDelete(err);
        }
      } else {
        RETURN_IF_ERROR(ParseIntValue(value_str, &local_numa_node_id_));
        if (local_numa_node_id_ < 0 || local_numa_node_id_ > numa_max_node()) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string(
                   "parameter 'local_numa_node_id_' must be non-negative "
                   "or less than max numa node id for tflite model '") +
               Name() + "'")
                  .c_str());
        }
      }

      // Handle remote_numa_node_id parameter
      err = GetParameterValue(params, "remote_numa_node_id", &value_str);

      // remote_numa_node_id is not required so clear error if not found
      if (err != nullptr) {
        if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
          return err;
        } else {
          TRITONSERVER_ErrorDelete(err);
        }
      } else {
        RETURN_IF_ERROR(ParseIntValue(value_str, &remote_numa_node_id_));
        if (remote_numa_node_id_ < 0 ||
            remote_numa_node_id_ > numa_max_node()) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string(
                   "parameter 'remote_numa_node_id_' must be non-negative "
                   "or less than max numa node id for tflite model '") +
               Name() + "'")
                  .c_str());
        }
      }

#else
      RETURN_ERROR_IF_TRUE(
          numa_alloc_policy_ != AllocationPolicy::NONE,
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string("Backend built without NUMA support, only valid "
                      "allocation policy is 'NONE'"));
#endif  // LIBNUMA_ENABLE
    }
  }

  // Handle tflite optimizations from model config
  {
    triton::common::TritonJson::Value optimization;
    if (ModelConfig().Find("optimization", &optimization)) {
      triton::common::TritonJson::Value eas;
      if (optimization.Find("execution_accelerators", &eas)) {
        triton::common::TritonJson::Value cpu_eas;
        if (eas.Find("cpu_execution_accelerator", &cpu_eas)) {
          for (size_t ea_idx = 0; ea_idx < cpu_eas.ArraySize(); ea_idx++) {
            triton::common::TritonJson::Value ea;
            RETURN_IF_ERROR(cpu_eas.IndexAsObject(ea_idx, &ea));
            std::string name;
            RETURN_IF_ERROR(ea.MemberAsString("name", &name));

#ifdef ARMNN_DELEGATE_ENABLE
            if (name == "armnn") {
              use_armnn_delegate_cpu_ = true;
              LOG_MESSAGE(
                  TRITONSERVER_LOG_VERBOSE,
                  (std::string(
                       "ArmNN Delegate Execution Accelerator is set for '") +
                   Name() + "' on CPU")
                      .c_str());

              // Validate and set parameters
              triton::common::TritonJson::Value params;
              if (ea.Find("parameters", &params)) {
                std::vector<std::string> param_keys;
                RETURN_IF_ERROR(params.Members(&param_keys));
                for (const auto& param_key : param_keys) {
                  std::string value_string;
                  if (param_key == "reduce_fp32_to_fp16") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    if (value_string == "on" || value_string == "off") {
                      armnn_cpu_reduce_fp32_to_fp16_ = value_string;
                    } else {
                      RETURN_ERROR_IF_FALSE(
                          false, TRITONSERVER_ERROR_INVALID_ARG,
                          std::string(
                              "Please pass on/off for reduce_fp32_to_fp16. '") +
                              value_string + "' is requested");
                    }
                  } else if (param_key == "reduce_fp32_to_bf16") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    if (value_string == "on" || value_string == "off") {
                      armnn_cpu_reduce_fp32_to_bf16_ = value_string;
                    } else {
                      RETURN_ERROR_IF_FALSE(
                          false, TRITONSERVER_ERROR_INVALID_ARG,
                          std::string(
                              "Please pass on/off for reduce_fp32_to_bf16. '") +
                              value_string + "' is requested");
                    }
                  } else if (param_key == "fast_math_enabled") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    if (value_string == "on" || value_string == "off") {
                      armnn_cpu_fast_math_enabled_ = value_string;
                    } else {
                      RETURN_ERROR_IF_FALSE(
                          false, TRITONSERVER_ERROR_INVALID_ARG,
                          std::string(
                              "Please pass on/off for fast_math_enabled. '") +
                              value_string + "' is requested");
                    }
                  } else if (param_key == "num_threads") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    RETURN_IF_ERROR(
                        ParseIntValue(value_string, &armnn_cpu_num_threads_));
                    if (armnn_cpu_num_threads_ < -1) {
                      return TRITONSERVER_ErrorNew(
                          TRITONSERVER_ERROR_INVALID_ARG,
                          std::string(
                              "armnn thread count '" + value_string +
                              "' is not in range [-1-64]")
                              .c_str());
                    }
                  } else {
                    return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INVALID_ARG,
                        std::string(
                            "unknown parameter '" + param_key +
                            "' is provided for ArmNN CPU Acceleration")
                            .c_str());
                  }
                }
              }
            } else if (name == "xnnpack") {
#else
            if (name == "xnnpack") {
#endif  // ARMNN_DELEGATE_ENABLE
              use_xnnpack_delegate_ = true;
              LOG_MESSAGE(
                  TRITONSERVER_LOG_VERBOSE,
                  (std::string(
                       "XNNPACK Delegate Execution Accelerator is set for '") +
                   Name() + "' on CPU")
                      .c_str());
              // Validate and set parameters
              triton::common::TritonJson::Value params;
              if (ea.Find("parameters", &params)) {
                std::vector<std::string> param_keys;
                RETURN_IF_ERROR(params.Members(&param_keys));
                for (const auto& param_key : param_keys) {
                  std::string value_string;
                  if (param_key == "num_threads") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    RETURN_IF_ERROR(
                        ParseIntValue(value_string, &num_threads_xnnpack_));
                  } else {
                    return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INVALID_ARG,
                        std::string(
                            "unknown parameter '" + param_key +
                            "' is provided for XNNPACK Acceleration")
                            .c_str());
                  }
                }
              }
            } else {
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string("unknown Execution Accelerator '") + name +
                   "' is requested")
                      .c_str());
            }
          }
        }

#ifdef ARMNN_DELEGATE_ENABLE
        // GPU Execution Accelerator is disabled on CPU devices.
        triton::common::TritonJson::Value gpu_eas;
        if (eas.Find("gpu_execution_accelerator", &gpu_eas)) {
          for (size_t ea_idx = 0; ea_idx < gpu_eas.ArraySize(); ea_idx++) {
            triton::common::TritonJson::Value ea;
            RETURN_IF_ERROR(gpu_eas.IndexAsObject(ea_idx, &ea));
            std::string name;
            RETURN_IF_ERROR(ea.MemberAsString("name", &name));
            if (name == "armnn") {
              use_armnn_delegate_gpu_ = true;
              LOG_MESSAGE(
                  TRITONSERVER_LOG_VERBOSE,
                  (std::string(
                       "ArmNN Delegate Execution Accelerator is set for '") +
                   Name() + "' on GPU")
                      .c_str());
              // Validate and set parameters
              triton::common::TritonJson::Value params;
              if (ea.Find("parameters", &params)) {
                std::vector<std::string> param_keys;
                RETURN_IF_ERROR(params.Members(&param_keys));
                for (const auto& param_key : param_keys) {
                  std::string value_string;
                  if (param_key == "reduce_fp32_to_fp16") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    if (value_string == "on" || value_string == "off") {
                      armnn_gpu_reduce_fp32_to_fp16_ == value_string;
                    } else {
                      RETURN_ERROR_IF_FALSE(
                          false, TRITONSERVER_ERROR_INVALID_ARG,
                          std::string(
                              "Please pass on/off for reduce_fp32_to_fp16. '") +
                              value_string + "' is requested");
                    }
                  } else if (param_key == "reduce_fp32_to_bf16") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    if (value_string == "on" || value_string == "off") {
                      armnn_gpu_reduce_fp32_to_bf16_ == value_string;
                    } else {
                      RETURN_ERROR_IF_FALSE(
                          false, TRITONSERVER_ERROR_INVALID_ARG,
                          std::string(
                              "Please pass on/off for reduce_fp32_to_bf16. '") +
                              value_string + "' is requested");
                    }
                  } else if (param_key == "fast_math_enabled") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    if (value_string == "on" || value_string == "off") {
                      armnn_gpu_fast_math_enabled_ == value_string;
                    } else {
                      RETURN_ERROR_IF_FALSE(
                          false, TRITONSERVER_ERROR_INVALID_ARG,
                          std::string(
                              "Please pass on/off for fast_math_enabled. '") +
                              value_string + "' is requested");
                    }
                  } else {
                    return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INVALID_ARG,
                        std::string(
                            "unknown parameter '" + param_key +
                            "' is provided for ArmNN GPU Acceleration")
                            .c_str());
                  }
                }
              }
            } else {
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string("unknown Execution Accelerator '") + name +
                   "' is requested")
                      .c_str());
            }
          }
        }
#endif  // ARMNN_DELEGATE_ENABLE
      }
    }
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelState::LoadModel()
{
  std::string artifact_filename;
  RETURN_IF_ERROR(ModelConfig().MemberAsString(
      "default_model_filename", &artifact_filename));

  // Find the TFLite model file that describes the model. If the model
  // configuration doesn't have an explicit model file specified then
  // use the default name ("model.tflite").
  std::string cc_model_filename = artifact_filename;
  if (cc_model_filename.empty()) {
    cc_model_filename = "model.tflite";
  }

  std::string model_path = JoinPath(
      {RepositoryPath(), std::to_string(Version()), cc_model_filename});

  {
    bool exists;
    RETURN_IF_ERROR(FileExists(model_path, &exists));
    RETURN_ERROR_IF_FALSE(
        exists, TRITONSERVER_ERROR_UNAVAILABLE,
        std::string("unable to find '") + model_path +
            "' for model instance '" + Name() + "'");
  }

  // Load the Tflite FlatBufferModel into memory
  model_ = tflite::FlatBufferModel::BuildFromFile((model_path).c_str());

  if (!model_) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        ("failed to load model " + Name()).c_str());
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  // We have the json DOM for the model configuration...
  triton::common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

#ifdef PAPI_PROFILING_ENABLE
  // Take this opportunity to handle papi events
  triton::common::TritonJson::Value params;
  if (ModelConfig().Find("parameters", &params)) {
    auto err = GetParameterValue(params, "papi_events", &papi_events_);
    // papi_events is not required so clear error if not found
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    }

    err = GetParameterValue(params, "papi_uncore_events", &papi_uncore_events_);
    // papi_events is not required so clear error if not found
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    }
  }
#endif  // PAPI_PROFILING_ENABLE

  // To check input and output names we will load and release the model during
  // the validation process without allocating memory for inference
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  tflite::InterpreterBuilder builder(*model_, resolver);
  builder(&interpreter);
  if (!interpreter) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, ("failed to build tflite interpreter "
                                      "during validation process for model " +
                                      Name())
                                         .c_str());
  }

  // inputs/outputs hold the list of tensor indexes in the graph for each
  // respectively
  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();
  size_t num_inputs = inputs.size();
  size_t num_outputs = outputs.size();

  // Populate input name map
  for (size_t i = 0; i < num_inputs; i++) {
    TfLiteTensor* input_tensor = interpreter->tensor(inputs[i]);
    if (input_tensor->allocation_type == kTfLiteArenaRw) {
      // Only worry about inputs that require user input
      input_index_map_[input_tensor->name] = inputs[i];
      input_dtype_map_[input_tensor->name] =
          ConvertTFLiteTypeToDataType(input_tensor->type);
    }
  }

  // Populate output name, dtype, shape map
  for (size_t i = 0; i < num_outputs; i++) {
    TfLiteTensor* output_tensor = interpreter->tensor(outputs[i]);
    TfLiteIntArray* tflite_output_tensor_dims = output_tensor->dims;
    std::vector<int64_t> output_shape_vector = std::vector<int64_t>(
        tflite_output_tensor_dims->data,
        (tflite_output_tensor_dims->data + tflite_output_tensor_dims->size));
    output_shape_map_[output_tensor->name] = output_shape_vector;
    output_index_map_[output_tensor->name] = outputs[i];
    output_dtype_map_[output_tensor->name] =
        ConvertTFLiteTypeToDataType(output_tensor->type);
  }

  triton::common::TritonJson::Value ios;

  // Validate model inputs
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("input", &ios));
  RETURN_ERROR_IF_FALSE(
      input_index_map_.size() == ios.ArraySize(), TRITONSERVER_ERROR_INTERNAL,
      std::string(
          "Number of required inputs for model does not match provided"));

  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));

    // Fetch name of input
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));

    // Return an error if the input name within the model config DNE in model
    RETURN_ERROR_IF_TRUE(
        input_index_map_.count(io_name) == 0, TRITONSERVER_ERROR_NOT_FOUND,
        std::string(
            "Model input: " + std::string(io_name) +
            " is not a valid input name for '" + Name() + "'"));


    // Validate data type
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
    const auto pr = ModelConfigDataTypeToTFLiteType(io_dtype);
    RETURN_ERROR_IF_TRUE(
        !pr.first, TRITONSERVER_ERROR_INTERNAL,
        ("unsupported datatype " + io_dtype + " for input '" + io_name +
         "' for model '" + Name() + "'"));

    // Validate datatype matches expected from model
    TRITONSERVER_DataType config_dtype =
        TRITONSERVER_StringToDataType(io_dtype.substr(strlen("TYPE_")).c_str());
    RETURN_ERROR_IF_TRUE(
        config_dtype != input_dtype_map_[io_name], TRITONSERVER_ERROR_INTERNAL,
        ("data type " + io_dtype + " for input '" + io_name +
         "' does not match expected of '" +
         TRITONSERVER_DataTypeString(input_dtype_map_[io_name]) + "'" +
         "' for model '" + Name() + "'"));

    // Validate input shape matches expected from model
    const TfLiteIntArray* tflite_dims =
        interpreter->tensor(inputs[i])->dims_signature;
    std::vector<int64_t> model_input_shape(
        tflite_dims->data, tflite_dims->data + tflite_dims->size);

    // Sometimes tflite models don't have shape info for input/output encoded
    if (!model_input_shape.empty()) {
      std::vector<int64_t> config_input_shape;
      triton::common::TritonJson::Value shape;
      if (io.Find("shape", &shape)) {
        RETURN_IF_ERROR(ParseShape(shape, "shape", &config_input_shape));
      } else {
        RETURN_IF_ERROR(ParseShape(io, "dims", &config_input_shape));
      }
      if (max_batch_size_ > 0) {
        // if batching is supported, tflite encodes -1 as the signature dim like
        // tf does. So just insert a -1 as the batch dim for the config input
        // shape to see if it lines up
        config_input_shape.insert(config_input_shape.begin(), -1);
      }
      if (config_input_shape != model_input_shape) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            ("shape " + VectorToString(config_input_shape) + " for input '" +
             io_name + "' does not match expected of '" +
             VectorToString(model_input_shape) + "'" + "' for model '" +
             Name() + "'")
                .c_str());
      }
    }
  }

  // Validate model outputs
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("output", &ios));

  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));

    // Fetch name of output
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));

    // Return an error if the output name within the model config DNE in model
    RETURN_ERROR_IF_TRUE(
        output_index_map_.count(io_name) == 0, TRITONSERVER_ERROR_NOT_FOUND,
        std::string(
            "Model output: " + std::string(io_name) +
            " is not a valid output name for '" + Name() + "'"));

    // Validate data type
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
    const auto pr = ModelConfigDataTypeToTFLiteType(io_dtype);
    RETURN_ERROR_IF_TRUE(
        !pr.first, TRITONSERVER_ERROR_INTERNAL,
        ("unsupported datatype " + io_dtype + " for output '" + io_name +
         "' for model '" + Name() + "'"));
    // Validate datatype matches expected from model
    TRITONSERVER_DataType config_dtype =
        TRITONSERVER_StringToDataType(io_dtype.substr(strlen("TYPE_")).c_str());
    RETURN_ERROR_IF_TRUE(
        config_dtype != output_dtype_map_[io_name], TRITONSERVER_ERROR_INTERNAL,
        ("data type " + io_dtype + " for output '" + io_name +
         "' does not match expected of '" +
         TRITONSERVER_DataTypeString(output_dtype_map_[io_name]) + "'" +
         "' for model '" + Name() + "'"));

    // Validate output shape matches expected from model
    const TfLiteIntArray* tflite_dims =
        interpreter->tensor(outputs[i])->dims_signature;
    std::vector<int64_t> model_output_shape(
        tflite_dims->data, tflite_dims->data + tflite_dims->size);

    // Sometimes tflite models don't have shape info for input/output encoded
    if (!model_output_shape.empty()) {
      std::vector<int64_t> config_output_shape;
      triton::common::TritonJson::Value shape;
      if (io.Find("shape", &shape)) {
        RETURN_IF_ERROR(ParseShape(shape, "shape", &config_output_shape));
      } else {
        RETURN_IF_ERROR(ParseShape(io, "dims", &config_output_shape));
      }
      if (max_batch_size_ > 0) {
        config_output_shape.insert(config_output_shape.begin(), -1);
      }
      RETURN_ERROR_IF_TRUE(
          config_output_shape != model_output_shape,
          TRITONSERVER_ERROR_INTERNAL,
          ("shape " + VectorToString(config_output_shape) + " for output '" +
           io_name + "' does not match expected of '" +
           VectorToString(model_output_shape) + "'" + "' for model '" + Name() +
           "'"));
    }
  }

  return nullptr;  // success
}  // namespace tensorflowlite

TRITONSERVER_Error*
ModelState::AutoCompleteConfig()
{
  // AutoComplete can be supported however we will defer this feature for later
  LOG_MESSAGE(
      TRITONSERVER_LOG_WARN,
      (std::string("skipping model configuration auto-complete for '") +
       Name() + "': not supported for tflite backend")
          .c_str());

  return nullptr;  // success
}

void
ModelState::InitTensorPipe()
{
  context_ = std::make_shared<tensorpipe::Context>();
  auto transportContext = tensorpipe::transport::shm::create();
  // Consider here also registering tcp transport if shm not avail
  context_->registerTransport(0 /* priority */, "shm", transportContext);
  // Register cma shm channel
  auto cmaChannel = tensorpipe::channel::cma::create();
  context_->registerChannel(0 /* low priority */, "cma", cmaChannel);
}

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
// This class acts as a manager for a subprocess which handles the actual tflite
// inference.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      const std::string& model_instance_name, ModelInstanceState** state);
  virtual ~ModelInstanceState();

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  // Execute...
  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      const std::string& model_instance_name);
  TRITONSERVER_Error* ConnectModelInstance();
  TRITONSERVER_Error* SendModel();
  TRITONSERVER_Error* LaunchModelInstance();
  bool ModelInstanceRunning();
  TRITONSERVER_Error* SetInputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      BackendInputCollector* collector,
      std::vector<BackendMemory*>* input_memories, tensorpipe::Message* tp_msg);
  TRITONSERVER_Error* Execute(
      std::vector<TRITONBACKEND_Response*>* responses,
      const uint32_t response_count, tensorpipe::Message* tp_msg,
      std::unordered_map<std::string, std::vector<char>>& inference_output);
  TRITONSERVER_Error* ReadOutputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      const std::unordered_map<std::string, std::vector<char>>&
          inference_output);

  // Pointer to the model state shared between instances
  ModelState* model_state_;

  // Name of the model instance used as a unique indenfier for this
  // instance
  const std::string model_instance_name_;

  // Tensorpipe listener to establish connection with child process
  std::shared_ptr<tensorpipe::Listener> listener_{nullptr};

  // Tensorpipe to send input tensors over
  std::shared_ptr<tensorpipe::Pipe> pipe_;

  // Process object for our backend model instance
  reproc::process model_instance_process_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    const std::string& model_instance_name, ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(
        model_state, triton_model_instance, model_instance_name);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    const std::string& model_instance_name)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state), model_instance_name_(model_instance_name)
{
  THROW_IF_BACKEND_INSTANCE_ERROR(LaunchModelInstance());
}

ModelInstanceState::~ModelInstanceState()
{
  pipe_->close();
  listener_->close();
  reproc::stop_actions stop = {
      {reproc::stop::terminate, reproc::milliseconds(10000)},
      {reproc::stop::kill, reproc::milliseconds(2000)},
      {}};
  reproc::options options;
  options.stop = stop;
  std::error_code ec;
  int status = 0;
  std::tie(status, ec) = model_instance_process_.stop(options.stop);
  if (ec) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Failed to stop child process");
  }
}

TRITONSERVER_Error*
ModelInstanceState::LaunchModelInstance()
{
  // Start listening for child process to connect to shm channel
  listener_ = model_state_->context_->listen({"shm://" + model_instance_name_});
  auto done = std::make_shared<std::promise<bool>>();
  listener_->accept([&, this](
                        const tensorpipe::Error& error,
                        std::shared_ptr<tensorpipe::Pipe> pipe) {
    // When the child process connects, we act here in this lambda function
    if (error) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("Unexpected error when accepting incoming pipe: ") +
           error.what())
              .c_str());

      done->set_value(false);
      return;
    }
    pipe_ = std::move(pipe);
    done->set_value(true);
  });

  std::vector<std::string> model_instance_args = {
      std::string(model_state_->model_instance_location_) + "/model_instance",
      std::string("shm://") + model_instance_name_};

#ifdef LIBNUMA_ENABLE
  // Model instance will always be pinned to numa node set as local, it's the
  // membinding we change
  switch (model_state_->numa_alloc_policy_) {
    case AllocationPolicy::LOCAL:
    case AllocationPolicy::WEIGHT_REMOTE_RESULT_LOCAL:
      // In the case of local result tensors (heap), membind to local numa node
      model_instance_args.insert(
          model_instance_args.begin(),
          {"numactl", "--membind",
           std::to_string(model_state_->local_numa_node_id_), "--cpunodebind",
           std::to_string(model_state_->local_numa_node_id_)});
      break;
    case AllocationPolicy::WEIGHT_LOCAL_RESULT_REMOTE:
    case AllocationPolicy::REMOTE:
      // In the case of remote result tensors (heap), membind to remote numa
      // node
      model_instance_args.insert(
          model_instance_args.begin(),
          {"numactl", "--membind",
           std::to_string(model_state_->remote_numa_node_id_), "--cpunodebind",
           std::to_string(model_state_->local_numa_node_id_)});
      break;
    default: {
      break;
    }
  }
#endif  // LIBNUMA_ENABLE

  // We have the model_instance process inherit the parent's standard streams
  // so the it reads directly from the stdin and writes directly to the
  // stdout/stderr triton.
  reproc::options options;
  options.redirect.out.type = reproc::redirect::type::parent;
  options.redirect.err.type = reproc::redirect::type::parent;
  options.env.behavior = reproc::env::extend;

  // For the child process to use Triton logging infra, we have to give it the
  // location of the actual tritonserver.so lib, as the backend is just linked
  // against a stub
  std::string* tritonserver_lib_path;
  dl_iterate_phdr(
      [](struct dl_phdr_info* info, size_t size, void* data) -> int {
        if (std::string(info->dlpi_name).find("tritonserver.so") !=
            std::string::npos) {
          *(reinterpret_cast<std::string**>(data)) =
              new std::string(info->dlpi_name);
          return 1;
        }
        return 0;
      },
      &tritonserver_lib_path);

  auto base_path = [](const std::string& str) -> std::string {
    size_t found;
    found = str.find_last_of("/\\");
    return str.substr(0, found);
  };

  std::unordered_map<std::string, std::string> model_instance_env{
      {"LD_LIBRARY_PATH", base_path(*tritonserver_lib_path)}};

#ifdef PAPI_PROFILING_ENABLE
  if (!model_state_->papi_events_.empty()) {
    model_instance_env.insert({"PAPI_EVENTS", model_state_->papi_events_});
  }
  if (!model_state_->papi_uncore_events_.empty()) {
    model_instance_env.insert(
        {"PAPI_UNCORE_EVENTS", model_state_->papi_uncore_events_});
  }
#endif  // PAPI_PROFILING_ENABLE

  options.env.extra = model_instance_env;

  std::error_code ec =
      model_instance_process_.start(model_instance_args, options);

  RETURN_ERROR_IF_TRUE(
      ec == std::errc::no_such_file_or_directory, TRITONSERVER_ERROR_INTERNAL,
      std::string(
          "model_instance binary not found. Make sure it's available from the "
          "PATH."));
  RETURN_ERROR_IF_TRUE(
      ec, TRITONSERVER_ERROR_INTERNAL,
      (std::string("Failed to launch model instance process: ") +
       ec.message()));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Launched model instance: ") + model_instance_name_)
          .c_str());

  // If the process did not come up in time something has gone wrong
  RETURN_ERROR_IF_TRUE(
      done->get_future().wait_for(std::chrono::seconds(5)) ==
          std::future_status::timeout,
      TRITONSERVER_ERROR_INTERNAL,
      std::string(
          "Model instance failed: process did not connect back to parent"));

  SendModel();

  return nullptr;
}

bool
ModelInstanceState::ModelInstanceRunning()
{
  int events = 0;
  std::error_code ec;
  std::tie(events, ec) = model_instance_process_.poll(
      reproc::event::exit, reproc::milliseconds(1000));
  return !ec && ((events & reproc::event::exit) != 0);
}

TRITONSERVER_Error*
ModelInstanceState::SendModel()
{
  tensorpipe::Message tp_msg;
  tp_msg.metadata = "model_load";

  // Size the payloads vector
  tp_msg.payloads.resize(OptimizerOption::COUNT + 1);

  // Place deserialized flatbuffer model in msg payload field
  const tflite::Allocation* model_allocation =
      model_state_->model_->allocation();
  tensorpipe::Message::Payload model_payload{
      .data = const_cast<void*>(model_allocation->base()),
      .length = model_allocation->bytes(),
      .metadata = std::string(model_instance_name_),
  };
  tp_msg.payloads[OptimizerOption::COUNT] = model_payload;

  // Define a helper function for generating payloads for our options
  auto gen_metadata = [](std::string s) {
    tensorpipe::Message::Payload result{.metadata = s};
    return result;
  };

  // Add in model configuration data to message
  tp_msg.payloads[OptimizerOption::TFLITE_NUM_THREADS] =
      gen_metadata(std::to_string(model_state_->tflite_num_threads_));

  // Add in numa config data to message
  tp_msg.payloads[OptimizerOption::NUMA_ALLOC_POLICY] =
      gen_metadata(AllocationPolicyToString(model_state_->numa_alloc_policy_));

  tp_msg.payloads[OptimizerOption::NUMA_LOCAL_NODE_ID] =
      gen_metadata(std::to_string(model_state_->local_numa_node_id_));

  tp_msg.payloads[OptimizerOption::NUMA_REMOTE_NODE_ID] =
      gen_metadata(std::to_string(model_state_->remote_numa_node_id_));

  // Add in use xnnpack
  std::string use_xnnpack = std::string("n");
  if (model_state_->use_xnnpack_delegate_ &&
      Kind() == TRITONSERVER_INSTANCEGROUPKIND_CPU) {
    use_xnnpack = std::string("y");
  }
  tp_msg.payloads[OptimizerOption::XNNPACK_ENABLE] = gen_metadata(use_xnnpack);

  // Add in xnnpack threads
  tp_msg.payloads[OptimizerOption::XNNPACK_CPU_NUM_THREADS] =
      gen_metadata(std::to_string(model_state_->num_threads_xnnpack_));

#ifdef ARMNN_DELEGATE_ENABLE
  // Add in use armnn cpu
  std::string use_armnn_cpu = std::string("n");
  if (model_state_->use_armnn_delegate_cpu_ &&
      Kind() == TRITONSERVER_INSTANCEGROUPKIND_CPU) {
    use_armnn_cpu = std::string("y");
  }
  tp_msg.payloads[OptimizerOption::ARMNN_CPU_ENABLE] =
      gen_metadata(use_armnn_cpu);

  // Add in use armnn gpu
  std::string use_armnn_gpu = std::string("n");
  if (model_state_->use_armnn_delegate_gpu_ &&
      Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    use_armnn_gpu = std::string("y");
  }
  tp_msg.payloads[OptimizerOption::ARMNN_GPU_ENABLE] =
      gen_metadata(use_armnn_gpu);

  // Add in armnn threads
  tp_msg.payloads[OptimizerOption::ARMNN_CPU_NUM_THREADS] =
      gen_metadata(std::to_string(model_state_->armnn_cpu_num_threads_));

  // Add in armnn cpu and gpu options
  tp_msg.payloads[OptimizerOption::ARMNN_CPU_FAST_MATH_ENABLED] =
      gen_metadata(model_state_->armnn_cpu_fast_math_enabled_);

  tp_msg.payloads[OptimizerOption::ARMNN_CPU_REDUCE_FP32_TO_FP16] =
      gen_metadata(model_state_->armnn_cpu_reduce_fp32_to_fp16_);

  tp_msg.payloads[OptimizerOption::ARMNN_CPU_REDUCE_FP32_TO_BF16] =
      gen_metadata(model_state_->armnn_cpu_reduce_fp32_to_bf16_);

  tp_msg.payloads[OptimizerOption::ARMNN_GPU_FAST_MATH_ENABLED] =
      gen_metadata(model_state_->armnn_gpu_fast_math_enabled_);

  tp_msg.payloads[OptimizerOption::ARMNN_GPU_REDUCE_FP32_TO_BF16] =
      gen_metadata(model_state_->armnn_gpu_reduce_fp32_to_bf16_);

  tp_msg.payloads[OptimizerOption::ARMNN_GPU_REDUCE_FP32_TO_FP16] =
      gen_metadata(model_state_->armnn_gpu_reduce_fp32_to_fp16_);
#endif  // ARMNN_DELEGATE_ENABLE

  // Write the message
  auto done = std::make_shared<std::promise<bool>>();
  pipe_->write(tp_msg, [this, done](const tensorpipe::Error& error) {
    // We now listen for a message to come back indicating the model load was
    // successful
    if (error) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          ("Failed to model load message. Details:" + error.what()).c_str());
      done->set_value(false);
      return;
    }
    pipe_->readDescriptor([this, done](
                              const tensorpipe::Error& error,
                              tensorpipe::Descriptor descriptor) {
      if (error) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("Unexpected error when reading from accepted pipe: ") +
             error.what())
                .c_str());
        done->set_value(false);
        return;
      }
      tensorpipe::Allocation allocation;
      pipe_->read(
          allocation, [descriptor, done](const tensorpipe::Error& error) {
            done->set_value(descriptor.metadata == "success");
          });
    });
  });
  RETURN_ERROR_IF_TRUE(
      done->get_future().wait_for(std::chrono::seconds(30)) ==
          std::future_status::timeout,
      TRITONSERVER_ERROR_INTERNAL,
      std::string("Model instance failed: process did not send model load "
                  "acknowledgement"));
  return nullptr;
}

void
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelExecute: Running ") + Name() + " with " +
       std::to_string(request_count) + " requests")
          .c_str());

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  const int32_t max_batch_size = model_state_->MaxBatchSize();

  // For each request collect the total batch size for this inference
  // execution. The batch-size, number of inputs, and size of each
  // input has already been checked so don't need to do that here.
  size_t total_batch_size = 0;
  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "null request given to TFLite backend for '" + Name() + "'")
                  .c_str()));
      return;
    }
  }

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests.
  if ((total_batch_size != 1) &&
      (total_batch_size > static_cast<size_t>(max_batch_size))) {
    RequestsRespondWithError(
        requests, request_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string(
                "batch size " + std::to_string(total_batch_size) + " for '" +
                Name() + "', max allowed is " + std::to_string(max_batch_size))
                .c_str()));
    return;
  }

  // At this point we are committed to running inference with all
  // 'requests'. Create a response for each request. During input
  // processing if there is an error with any request that error will
  // be sent immediately with the corresponding response (and the
  // response unique_ptr will then be nullptr). The request object
  // itself will not be released until after all inferencing is done
  // (below) as we may need to access the request object when
  // determine how to process outputs (for example, even if we don't
  // need the outputs for a request that has an error, we do need to
  // know the size of those outputs associated with the request so we
  // can skip them in the output tensors).
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);
  bool all_response_failed = false;

  for (size_t i = 0; i < request_count; ++i) {
    TRITONBACKEND_Response* response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err == nullptr) {
      responses.emplace_back(response);
    } else {
      responses.emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    }
  }

  for (size_t i = 0; i < request_count; i++) {
    if (max_batch_size > 0) {
      // Retrieve the batch size from one of the inputs, if the model
      // supports batching, the first dimension size is batch size
      TRITONBACKEND_Input* input;
      TRITONSERVER_Error* err =
          TRITONBACKEND_RequestInputByIndex(requests[i], 0 /* index */, &input);
      if (err == nullptr) {
        const int64_t* shape;
        err = TRITONBACKEND_InputProperties(
            input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
        total_batch_size += shape[0];
      }
      if (err != nullptr) {
        RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
            responses, request_count, all_response_failed, err);
      }
    } else {
      total_batch_size += 1;
    }
  }

  // If there are no valid payloads then no need to run the inference.
  if (total_batch_size == 0) {
    return;
  }

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests.
  if (!all_response_failed) {
    if ((total_batch_size != 1) &&
        (total_batch_size > (size_t)max_batch_size)) {
      RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
          responses, request_count, all_response_failed,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "batch size " + std::to_string(total_batch_size) + " for '" +
                  Name() + "', max allowed is " +
                  std::to_string(max_batch_size))
                  .c_str()));
    }
  }

  // Here we allocate the space for the tensorpipe message that's used to
  // communicate with our backend ModelInstance process
  tensorpipe::Message tp_msg;

  // Here we allocate the space for the tensorpipe allocation that the result of
  // the inference is written to upon success
  std::unordered_map<std::string, std::vector<char>> inference_output;

  std::vector<BackendMemory*> input_memories;
  std::unique_ptr<BackendInputCollector> collector;

  if (!all_response_failed) {
    collector.reset(new BackendInputCollector(
        requests, request_count, &responses,
        model_state_->TritonMemoryManager(), false, nullptr));
    // Note here we are copying the triton input buffers to the tflite allocated
    // buffers
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        SetInputTensors(
            total_batch_size, requests, request_count, &responses,
            collector.get(), &input_memories, &tp_msg));
  }

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  // Run...
  if (!all_response_failed) {
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        Execute(&responses, request_count, &tp_msg, inference_output));
  }

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  // Free BackendMemory used for inputs
  for (BackendMemory* mem : input_memories) {
    delete mem;
  }
  input_memories.clear();

  if (!all_response_failed) {
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        ReadOutputTensors(
            total_batch_size, requests, request_count, &responses,
            inference_output));
  }

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  // Send all the responses that haven't already been sent because of
  // an earlier error. Note that the responses are not set to nullptr
  // here as we need that indication below to determine if the request
  // we successful or not.
  for (auto& response : responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed to send TFLite backend response");
    }
  }

  // Report statistics for each request.
  for (uint64_t r = 0; r < request_count; ++r) {
    auto& request = requests[r];
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            TritonModelInstance(), request,
            (responses[r] != nullptr) /* success */, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  if (!all_response_failed) {
    // Report the entire batch statistics.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportBatchStatistics(
            TritonModelInstance(), total_batch_size, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting batch request statistics");
  }
}

TRITONSERVER_Error*
ModelInstanceState::SetInputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    BackendInputCollector* collector,
    std::vector<BackendMemory*>* input_memories, tensorpipe::Message* tp_msg)
{
  const int32_t max_batch_size = model_state_->MaxBatchSize();

  // Construct tensorpipe message
  tp_msg->metadata = "model_input";
  tp_msg->tensors.resize(model_state_->input_index_map_.size());

  // All requests must have equally-sized input tensors so use any
  // request as the representative for the input tensors.
  uint32_t input_count;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(requests[0], &input_count));
  for (uint64_t input_idx = 0; input_idx < input_count; input_idx++) {
    TRITONBACKEND_Input* input;
    RETURN_IF_ERROR(
        TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input));

    const char* input_name;
    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    uint64_t byte_size;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &input_name, &input_datatype, &input_shape, &input_dims_count,
        &byte_size, nullptr));

    // Return an error if the input name within the request DNE in model
    if (model_state_->input_index_map_.count(input_name) == 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_NOT_FOUND,
          std::string(
              "Model input: " + std::string(input_name) +
              " is not a valid input name for '" + Name() + "'")
              .c_str());
    }

    // The shape for the entire input patch, [total_batch_size, ...]
    std::vector<int64_t> batchn_shape(
        input_shape, input_shape + input_dims_count);
    if (max_batch_size != 0) {
      batchn_shape[0] = total_batch_size;
    }

    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string(
             "total batch size for input " + std::string(input_name) +
             " is: " + std::to_string(total_batch_size) + "\n"))
            .c_str());

    // We use the metadata string field to pass the input tensor index.
    tp_msg->tensors[input_idx].metadata =
        std::to_string(model_state_->input_index_map_[input_name]);

    // Even if running on MALI GPU, we use CPU memory
    std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> alloc_perference;
    alloc_perference = {{TRITONSERVER_MEMORY_CPU, 0}};

    size_t batchn_byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;

    // Here we use ProcessTensor to manage the input buffer for the tensor. In
    // the overload of this function, the backend input collector manages the
    // memory, as opposed to copying it into the destination buffer we could
    // pass, `buffer`. At the end of this call, cpu_buffer will point to the
    // contiguous memory for the potentially batched input tensors
    tensorpipe::CpuBuffer cpu_buffer;
    RETURN_IF_ERROR(collector->ProcessTensor(
        input_name, nullptr, 0, alloc_perference,
        const_cast<const char**>(reinterpret_cast<char**>(&cpu_buffer.ptr)),
        &batchn_byte_size, &memory_type, &memory_type_id));

    // Set the space for the tensors for tensorpipe message
    tp_msg->tensors[input_idx].length = static_cast<size_t>(batchn_byte_size);
    tp_msg->tensors[input_idx].buffer = cpu_buffer;
  }

  // Finalize Backend Input Collector...
  collector->Finalize();

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::Execute(
    std::vector<TRITONBACKEND_Response*>* responses,
    const uint32_t response_count, tensorpipe::Message* tp_msg,
    std::unordered_map<std::string, std::vector<char>>& inference_output)
{
  // Write tensor across pipe and wait for completion asynchronously
  auto done = std::make_shared<std::promise<bool>>();
  pipe_->write(
      *tp_msg,
      [this, &inference_output, &done](const tensorpipe::Error& error) {
        if (error) {
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string(
                   "Failed to send model_input request to server. Details: ") +
               error.what())
                  .c_str());
          done->set_value(false);
          return;
        }
        // Read a response from the client with description of incoming
        // result tensors so we can get ready to write the data
        pipe_->readDescriptor([this, &inference_output, &done](
                                  const tensorpipe::Error& error,
                                  tensorpipe::Descriptor descriptor) {
          if (error) {
            LOG_MESSAGE(
                TRITONSERVER_LOG_ERROR,
                (std::string(
                     "Unexpected error when reading descriptor from accepted "
                     "pipe. Details: ") +
                 error.what())
                    .c_str());
            done->set_value(false);
            return;
          }

          tensorpipe::Allocation allocation;

          // If there was a problem running the inference we get that back in
          // the message metadata
          if (descriptor.metadata == "f") {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Failed to run inference");
            pipe_->read(allocation, [&done](const tensorpipe::Error& error) {});
            done->set_value(false);
            return;
          }

          // Create a cpu buffer instance and assign its buffer
          // pointer to that of the tflite allocated buffer for our
          // output tensor
          allocation.tensors.resize(descriptor.tensors.size());
          for (uint64_t i = 0; i < descriptor.tensors.size(); ++i) {
            inference_output[descriptor.tensors[i].metadata].resize(
                descriptor.tensors[i].length);

            allocation.tensors[i].buffer = tensorpipe::CpuBuffer{
                .ptr = static_cast<void*>(
                    inference_output[descriptor.tensors[i].metadata].data())};
          }

          // Read the data from the client response into the tensor
          // buffer assigned above
          pipe_->read(allocation, [&done](const tensorpipe::Error& error) {
            if (error) {
              LOG_MESSAGE(
                  TRITONSERVER_LOG_ERROR,
                  (std::string(
                       "Unexpected error when reading data from accepted "
                       "pipe. Details: ") +
                   error.what())
                      .c_str());
              done->set_value(false);
              return;
            }
            done->set_value(true);
          });
        });
      });

  RETURN_ERROR_IF_FALSE(
      done->get_future().get(), TRITONSERVER_ERROR_INTERNAL,
      std::string("TFLite execute failure"));
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::ReadOutputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    const std::unordered_map<std::string, std::vector<char>>& inference_output)
{
  BackendOutputResponder responder(
      requests, request_count, responses, model_state_->MaxBatchSize(),
      model_state_->TritonMemoryManager(), false, nullptr);

  // Respond to each output individually
  try {
    for (const auto& map_entry : model_state_->output_index_map_) {
      const std::string& output_name = map_entry.first;
      model_state_->output_shape_map_[output_name][0] = total_batch_size;

      responder.ProcessTensor(
          output_name, model_state_->output_dtype_map_[output_name],
          model_state_->output_shape_map_[output_name],
          inference_output.at(output_name).data(), TRITONSERVER_MEMORY_CPU, 0);
    }
  }
  catch (std::out_of_range& err) {
    responder.Finalize();
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_NOT_FOUND, "Failed to process output tensor");
  }

  // Finalize and wait for any pending buffer copies.
  responder.Finalize();

  return nullptr;
}

/////////////

extern "C" {

TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // Check the backend API version that Triton supports vs. what this
  // backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("Triton TRITONBACKEND API version: ") +
         std::to_string(api_version_major) + "." +
         std::to_string(api_version_minor) + " does not support '" + name +
         "' TRITONBACKEND API version: " +
         std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
         std::to_string(TRITONBACKEND_API_VERSION_MINOR))
            .c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  // Create a ModelState object and associate it with the
  // TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  // One of the primary things to do in ModelInitialize is to examine
  // the model configuration to ensure that it is something that this
  // backend can support. If not, returning an error from this
  // function will prevent the model from loading.
  RETURN_IF_ERROR(model_state->ValidateModelConfig());

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name +
       " (device " + std::to_string(device_id) + ")")
          .c_str());

  // Get the model state associated with this instance's model.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // Create a ModelInstanceState object and associate it with the
  // TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, name, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Suggested practice for this is to use only
  // function-local and model-instance-specific state (obtained from
  // 'instance'), which is what we do here.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  // This backend specifies BLOCKING execution policy. That means that
  // we should not return from this function until execution is
  // complete. Triton will automatically release 'instance' on return
  // from this function so that it is again available to be used for
  // another call to TRITONBACKEND_ModelInstanceExecute.

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model ") + model_state->Name() + ", instance " +
       instance_state->Name() + ", executing " + std::to_string(request_count) +
       " requests")
          .c_str());

  // At this point we accept ownership of 'requests', which means that
  // even if something goes wrong we must still return success from
  // this function. If something does go wrong in processing a
  // particular request then we send an error response just for the
  // specific request.
  instance_state->ProcessRequests(requests, request_count);

  return nullptr;  // success
}

}  // extern "C"
}}}  // namespace triton::backend::tensorflowlite
