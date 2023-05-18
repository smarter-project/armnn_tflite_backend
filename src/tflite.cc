//
// Copyright © 2023 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <stdint.h>

#include <algorithm>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tflite_utils.h"
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

#ifdef ARMNN_DELEGATE_ENABLE
// ArmNN headers
#include "armnn/ArmNN.hpp"
#include "armnn_delegate.hpp"
#endif  // ARMNN_DELEGATE_ENABLE

#ifdef PAPI_PROFILING_ENABLE
#include <papi.h>

#include "papi_profiler.h"
#endif  // PAPI_PROFILING_ENABLE

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
      TRITONBACKEND_Model* triton_model, ModelState** state,
      int32_t* armnn_threads);
  ~ModelState();

  // Load a serialized tflite model using 'artifact_name' as the name for the
  // tflite model file. Return in 'model_path' the full path to the
  // tflite model file. Return in 'model' the TFLite network,
  // representing the model.
  TRITONSERVER_Error* LoadModel(
      const std::string& artifact_name, std::string* model_path,
      common::TritonJson::Value& model_config,
      std::unique_ptr<tflite::FlatBufferModel>* model);

  // Validate that model configuration is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

  // Default TFLite runtime options
  int32_t tflite_num_threads_ =
      static_cast<int32_t>(std::thread::hardware_concurrency());

#ifdef ARMNN_DELEGATE_ENABLE
  // ArmNN Delegate options
  bool use_armnn_delegate_cpu_ = false;
  bool use_armnn_delegate_gpu_ = false;
  armnn::OptimizerOptions armnn_optimizer_options_cpu_;
  armnn::OptimizerOptions armnn_optimizer_options_gpu_;
  int32_t* armnn_threads_;
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

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  TRITONSERVER_Error* AutoCompleteConfig();
};


TRITONSERVER_Error*
ModelState::Create(
    TRITONBACKEND_Model* triton_model, ModelState** state,
    int32_t* armnn_threads)
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

#ifdef ARMNN_DELEGATE_ENABLE
  (*state)->armnn_threads_ = armnn_threads;
#endif

  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model)
{  // Here we can add information to the model state that can be shared across
   // model instances. See onnx backend for example. MALI GPU optimization level
   // may be candidate.
}

ModelState::~ModelState() {}

TRITONSERVER_Error*
ModelState::LoadModel(
    const std::string& artifact_name, std::string* model_path,
    common::TritonJson::Value& model_config,
    std::unique_ptr<tflite::FlatBufferModel>* model)
{
  // Find the TFLite model file that describes the model. If the model
  // configuration doesn't have an explicit model file specified then
  // use the default name ("model.tflite").
  std::string cc_model_filename = artifact_name;
  if (cc_model_filename.empty()) {
    cc_model_filename = "model.tflite";
  }

  *model_path = JoinPath(
      {RepositoryPath(), std::to_string(Version()), cc_model_filename});

  {
    bool exists;
    RETURN_IF_ERROR(FileExists(*model_path, &exists));
    RETURN_ERROR_IF_FALSE(
        exists, TRITONSERVER_ERROR_UNAVAILABLE,
        std::string("unable to find '") + *model_path +
            "' for model instance '" + Name() + "'");
  }

  // Load the Tflite FlatBufferModel into memory
  *model = tflite::FlatBufferModel::BuildFromFile((*model_path).c_str());

  if (!*model) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        ("failed to load model " + Name()).c_str());
  }

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
                    if (value_string == "on") {
                      armnn_optimizer_options_cpu_.m_ReduceFp32ToFp16 = true;
                    } else if (value_string == "off") {
                      armnn_optimizer_options_cpu_.m_ReduceFp32ToFp16 = false;
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
                    if (value_string == "on") {
                      armnn_optimizer_options_cpu_.m_ReduceFp32ToBf16 = true;
                    } else if (value_string == "off") {
                      armnn_optimizer_options_cpu_.m_ReduceFp32ToBf16 = false;
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
                    if (value_string == "on") {
                      armnn::BackendOptions option(
                          "CpuAcc", {{"FastMathEnabled", true}});
                      armnn_optimizer_options_cpu_.m_ModelOptions.push_back(
                          option);
                    } else if (value_string == "off") {
                      armnn::BackendOptions option(
                          "CpuAcc", {{"FastMathEnabled", false}});
                      armnn_optimizer_options_cpu_.m_ModelOptions.push_back(
                          option);
                    } else {
                      RETURN_ERROR_IF_FALSE(
                          false, TRITONSERVER_ERROR_INVALID_ARG,
                          std::string(
                              "Please pass on/off for fast_math_enabled. '") +
                              value_string + "' is requested");
                    }
                  } else if (param_key == "num_threads") {
                    int32_t num_threads;
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    RETURN_IF_ERROR(ParseIntValue(value_string, &num_threads));
                    if (num_threads < 0) {
                      return TRITONSERVER_ErrorNew(
                          TRITONSERVER_ERROR_INVALID_ARG,
                          std::string(
                              "armnn thread count '" + value_string +
                              "' is not in range [1-64]")
                              .c_str());
                    }

                    // Here we do an ugly hack to prevent armnn/acl thread
                    // issues For now we make sure the next armnn accelerated
                    // model loaded does not request more threads than the
                    // previous, as this creates a segfault
                    if (num_threads > *armnn_threads_) {
                      num_threads = *armnn_threads_;
                      LOG_MESSAGE(
                          TRITONSERVER_LOG_INFO,
                          (std::string("Model threads requested larger than "
                                       "that of first model loaded: ") +
                           value_string + " > " +
                           std::to_string(*armnn_threads_) +
                           ". Using smaller thread value instead.")
                              .c_str());
                    } else {
                      *armnn_threads_ = num_threads;
                    }
                    armnn::BackendOptions option(
                        "CpuAcc", {{"NumberOfThreads",
                                    static_cast<unsigned int>(num_threads)}});
                    armnn_optimizer_options_cpu_.m_ModelOptions.push_back(
                        option);
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
              armnn::OptimizerOptions armnn_optimizer_options_gpu_;
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
                    if (value_string == "on") {
                      armnn_optimizer_options_gpu_.m_ReduceFp32ToFp16 = true;
                    } else if (value_string == "off") {
                      armnn_optimizer_options_gpu_.m_ReduceFp32ToFp16 = false;
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
                    if (value_string == "on") {
                      armnn_optimizer_options_gpu_.m_ReduceFp32ToBf16 = true;
                    } else if (value_string == "off") {
                      armnn_optimizer_options_gpu_.m_ReduceFp32ToBf16 = false;
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
                    if (value_string == "on") {
                      armnn::BackendOptions option(
                          "GpuAcc", {{"FastMathEnabled", true}});
                      armnn_optimizer_options_gpu_.m_ModelOptions.push_back(
                          option);
                    } else if (value_string == "off") {
                      armnn::BackendOptions option(
                          "GpuAcc", {{"FastMathEnabled", false}});
                      armnn_optimizer_options_gpu_.m_ModelOptions.push_back(
                          option);
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

  return nullptr;  // success
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

  // To check input and output names we will load and release the model during
  // the validation process without allocating memory for inference
  std::string model_path;
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  std::string artifact_filename;
  RETURN_IF_ERROR(ModelConfig().MemberAsString(
      "default_model_filename", &artifact_filename));
  RETURN_IF_ERROR(
      LoadModel(artifact_filename, &model_path, ModelConfig(), &model));

  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
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
    input_index_map_[interpreter->GetInputName(i)] = inputs[i];
    input_dtype_map_[interpreter->GetInputName(i)] =
        ConvertTFLiteTypeToDataType(interpreter->tensor(inputs[i])->type);
  }

  // Populate output name and dtype map
  for (size_t i = 0; i < num_outputs; i++) {
    output_index_map_[interpreter->GetOutputName(i)] = outputs[i];
    output_dtype_map_[interpreter->GetOutputName(i)] =
        ConvertTFLiteTypeToDataType(interpreter->tensor(outputs[i])->type);
  }

  triton::common::TritonJson::Value ios;

  // Validate model inputs
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("input", &ios));

  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));

    // Fetch name of input
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));

    // Return an error if the input name within the model config DNE in model
    if (input_index_map_.count(io_name) == 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_NOT_FOUND,
          std::string(
              "Model input: " + std::string(io_name) +
              " is not a valid input name for '" + Name() + "'")
              .c_str());
    }

    // Validate data type
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
    const auto pr = ModelConfigDataTypeToTFLiteType(io_dtype);
    if (!pr.first) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("unsupported datatype " + io_dtype + " for input '" + io_name +
           "' for model '" + Name() + "'")
              .c_str());
    }

    // Validate datatype matches expected from model
    TRITONSERVER_DataType config_dtype =
        TRITONSERVER_StringToDataType(io_dtype.substr(strlen("TYPE_")).c_str());
    if (config_dtype != input_dtype_map_[io_name]) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("data type " + io_dtype + " for input '" + io_name +
           "' does not match expected of '" +
           TRITONSERVER_DataTypeString(input_dtype_map_[io_name]) + "'" +
           "' for model '" + Name() + "'")
              .c_str());
    }

    // Validate input shape matches expected from model
    TfLiteIntArray* tflite_dims = interpreter->tensor(inputs[i])->dims;
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
        // if batching is supported, you tflite doesn't encode -1 as
        // the dim like tf does, it's just a 1. So just insert a 1 as the
        // batch dim for the config input shape to see if it lines up
        config_input_shape.insert(config_input_shape.begin(), 1);
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
    if (output_index_map_.count(io_name) == 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_NOT_FOUND,
          std::string(
              "Model output: " + std::string(io_name) +
              " is not a valid output name for '" + Name() + "'")
              .c_str());
    }

    // Validate data type
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
    const auto pr = ModelConfigDataTypeToTFLiteType(io_dtype);
    if (!pr.first) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("unsupported datatype " + io_dtype + " for output '" + io_name +
           "' for model '" + Name() + "'")
              .c_str());
    }
    // Validate datatype matches expected from model
    TRITONSERVER_DataType config_dtype =
        TRITONSERVER_StringToDataType(io_dtype.substr(strlen("TYPE_")).c_str());
    if (config_dtype != output_dtype_map_[io_name]) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("data type " + io_dtype + " for output '" + io_name +
           "' does not match expected of '" +
           TRITONSERVER_DataTypeString(output_dtype_map_[io_name]) + "'" +
           "' for model '" + Name() + "'")
              .c_str());
    }

    // Validate output shape matches expected from model
    TfLiteIntArray* tflite_dims = interpreter->tensor(outputs[i])->dims;
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
        config_output_shape.insert(config_output_shape.begin(), 1);
      }
      if (config_output_shape != model_output_shape) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            ("shape " + VectorToString(config_output_shape) + " for output '" +
             io_name + "' does not match expected of '" +
             VectorToString(model_output_shape) + "'" + "' for model '" +
             Name() + "'")
                .c_str());
      }
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


//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState();

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  // Execute...
  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);
  TRITONSERVER_Error* BuildInterpreter();
  void LogDelegation(const std::string& delegate_name);
  void Execute(
      std::vector<TRITONBACKEND_Response*>* responses,
      const uint32_t response_count);
  void SetInputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      BackendInputCollector* collector,
      std::vector<BackendMemory*>* input_memories);
  void ReadOutputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);

  ModelState* model_state_;

  // The full path to the TFLite model file.
  std::string model_path_;

  // The pointer to the tflite network
  std::unique_ptr<tflite::FlatBufferModel> model_;

  // The pointer to the tflite interpreter instance
  std::unique_ptr<tflite::Interpreter> interpreter_;

  // State variable to register whether inference has been called at least once
  bool first_inference_ = true;

#ifdef PAPI_PROFILING_ENABLE
  std::unique_ptr<tflite::Profiler> papi_profiler_ = MaybeCreatePapiProfiler();
#endif  // PAPI_PROFILING_ENABLE
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
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
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state)
{
  // Load the TFLite network
  THROW_IF_BACKEND_INSTANCE_ERROR(model_state->LoadModel(
      ArtifactFilename(), &model_path_, model_state->ModelConfig(), &model_));

  // Build interpreter
  THROW_IF_BACKEND_INSTANCE_ERROR(BuildInterpreter());

#ifdef PAPI_PROFILING_ENABLE
  interpreter_->AddProfiler(papi_profiler_.get());
#endif  // PAPI_PROFILING_ENABLE
}

ModelInstanceState::~ModelInstanceState()
{
  // Consider the function ReleaseNonPersistentMemory here for our interpreter
  interpreter_.reset();
}

TRITONSERVER_Error*
ModelInstanceState::BuildInterpreter()
{
  // Build the tflite interpreter
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  tflite::InterpreterBuilder builder(*model_, resolver);
  builder(&interpreter_);
  if (!interpreter_) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        ("failed to build tflite interpreter for model " + Name()).c_str());
  }

  // Tell interpreter to use max threads available to system
  if (interpreter_->SetNumThreads(model_state_->tflite_num_threads_) !=
      kTfLiteOk) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        ("failed to set number of threads for interpreter for model " + Name())
            .c_str());
  }

#ifdef ARMNN_DELEGATE_ENABLE
  bool armnn_gpu_delegate_enabled =
      model_state_->use_armnn_delegate_gpu_ &&
      Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU;
  bool armnn_cpu_delegate_enabled =
      model_state_->use_armnn_delegate_cpu_ &&
      Kind() == TRITONSERVER_INSTANCEGROUPKIND_CPU;
  if (armnn_cpu_delegate_enabled || armnn_gpu_delegate_enabled) {
    armnnDelegate::DelegateOptions armnn_delegate_options =
        armnnDelegate::TfLiteArmnnDelegateOptionsDefault();

    // Set backend prefs based on gpu or cpu selection
    if (armnn_gpu_delegate_enabled) {
      armnn_delegate_options.SetBackends(
          {armnn::Compute::GpuAcc, armnn::Compute::CpuAcc});
      armnn_delegate_options.SetOptimizerOptions(
          model_state_->armnn_optimizer_options_gpu_);
    } else {
      // Set backend pref to Neon ACL backend
      armnn_delegate_options.SetBackends({armnn::Compute::CpuAcc});
      armnn_delegate_options.SetOptimizerOptions(
          model_state_->armnn_optimizer_options_cpu_);
    }

    // Create ArmNN Delegate with options registered in model state
    std::unique_ptr<
        TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)>
        armnn_delegate(
            armnnDelegate::TfLiteArmnnDelegateCreate(armnn_delegate_options),
            armnnDelegate::TfLiteArmnnDelegateDelete);

    // Instruct the Interpreter to use the armnnDelegate
    if (interpreter_->ModifyGraphWithDelegate(std::move(armnn_delegate)) !=
        kTfLiteOk) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("failed to use armnn delegate for model " + Name()).c_str());
    }
    LogDelegation("armnn");
  } else if (
      model_state_->use_xnnpack_delegate_ &&
      Kind() == TRITONSERVER_INSTANCEGROUPKIND_CPU) {
#else
  if (model_state_->use_xnnpack_delegate_ &&
      Kind() == TRITONSERVER_INSTANCEGROUPKIND_CPU) {
#endif  // ARMNN_DELEGATE_ENABLE
    // Create the XNNPack Delegate
    TfLiteXNNPackDelegateOptions options =
        TfLiteXNNPackDelegateOptionsDefault();

    options.num_threads = model_state_->num_threads_xnnpack_;

    tflite::Interpreter::TfLiteDelegatePtr xnnpack_delegate(
        TfLiteXNNPackDelegateCreate(&options),
        [](TfLiteDelegate* xnnpack_delegate) {
          TfLiteXNNPackDelegateDelete(xnnpack_delegate);
        });

    // Instruct the Interpreter to use the xnnpack
    if (interpreter_->ModifyGraphWithDelegate(std::move(xnnpack_delegate)) !=
        kTfLiteOk) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("failed to use xnnpack delegate for model " + Name()).c_str());
    }
    LogDelegation("xnnpack");
  }

  return nullptr;
}

void
ModelInstanceState::LogDelegation(const std::string& delegate_name)
{
  std::unordered_set<uint32_t> checked_node_ids;
  uint32_t num_delegated_kernels = 0;
  for (uint64_t i = 0; i < interpreter_->execution_plan().size(); i++) {
    int32_t node_id = interpreter_->execution_plan()[i];
    if (checked_node_ids.find(node_id) != checked_node_ids.end()) {
      continue;
    }
    const TfLiteNode& node =
        interpreter_->node_and_registration(node_id)->first;

    if (node.delegate != nullptr) {
      num_delegated_kernels++;
      checked_node_ids.insert(node_id);
    }
  }
  bool fully_delegated =
      (num_delegated_kernels == 1 &&
       interpreter_->execution_plan().size() == 1);

  if (fully_delegated) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO, ("Applied " + delegate_name +
                                " delegate, and the model graph will be "
                                "completely executed by the delegate.")
                                   .c_str());
  } else if (num_delegated_kernels > 0) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        ("Applied " + delegate_name +
         " delegate, and the model graph will be paritally executed by the "
         "delegate w/ " +
         std::to_string(num_delegated_kernels) + " delegate kernels.")
            .c_str());
  } else {
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO, ("Though " + delegate_name +
                                " delegate is applied, the model graph will "
                                "not be executed by the delegate.")
                                   .c_str());
  }
}

void
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
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
        RequestsRespondWithError(requests, request_count, err);
        return;
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

  for (size_t i = 0; i < request_count; i++) {
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

  std::vector<BackendMemory*> input_memories;
  BackendInputCollector collector(
      requests, request_count, &responses, model_state_->TritonMemoryManager(),
      false, nullptr);

  // Note here we are copying the triton input buffers to the tflite allocated
  // buffers
  SetInputTensors(
      total_batch_size, requests, request_count, &responses, &collector,
      &input_memories);

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  // Run...
  Execute(&responses, request_count);

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  // Free BackendMemory used for inputs
  for (BackendMemory* mem : input_memories) {
    delete mem;
  }
  input_memories.clear();

  ReadOutputTensors(total_batch_size, requests, request_count, &responses);

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
  for (uint32_t r = 0; r < request_count; ++r) {
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

  // Report the entire batch statistics.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          TritonModelInstance(), total_batch_size, exec_start_ns,
          compute_start_ns, compute_end_ns, exec_end_ns),
      "failed reporting batch request statistics");
}

void
ModelInstanceState::Execute(
    std::vector<TRITONBACKEND_Response*>* responses,
    const uint32_t response_count)
{
  static TfLiteStatus status;
  status = interpreter_->Invoke();
  if (status != kTfLiteOk) {
    SendErrorForResponses(
        responses, response_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, ("TFLite execute failure")));
  }
  first_inference_ = false;
}

void
ModelInstanceState::SetInputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    BackendInputCollector* collector,
    std::vector<BackendMemory*>* input_memories)
{
  const int32_t max_batch_size = model_state_->MaxBatchSize();
  bool allocate_tensors = false;

  // All requests must have equally-sized input tensors so use any
  // request as the representative for the input tensors.
  uint32_t input_count;
  RESPOND_ALL_AND_RETURN_IF_ERROR(
      responses, request_count,
      TRITONBACKEND_RequestInputCount(requests[0], &input_count));
  for (uint32_t input_idx = 0; input_idx < input_count; input_idx++) {
    TRITONBACKEND_Input* input;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input));

    const char* input_name;
    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        TRITONBACKEND_InputProperties(
            input, &input_name, &input_datatype, &input_shape,
            &input_dims_count, nullptr, nullptr));

    // Return an error if the input name within the request DNE in model
    if (model_state_->input_index_map_.count(input_name) == 0) {
      SendErrorForResponses(
          responses, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_NOT_FOUND,
              std::string(
                  "Model input: " + std::string(input_name) +
                  " is not a valid input name for '" + Name() + "'")
                  .c_str()));
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

    // Get the batch input tensor shape and compare against the shape of the
    // input tensor as is registered with the current interpreter. If the size
    // is different from the last call, tell the interpreter to resize the
    // input tensor and note that we are going to have to make another call to
    // AllocateTensors below
    std::vector<int32_t> batchn_tflite_size_vector(
        begin(batchn_shape), end(batchn_shape));
    TfLiteIntArray* tflite_input_tensor_dims =
        interpreter_->tensor(model_state_->input_index_map_[input_name])->dims;
    std::vector<int32_t> tflite_input_shape(
        tflite_input_tensor_dims->data,
        (tflite_input_tensor_dims->data + tflite_input_tensor_dims->size));
    if (batchn_tflite_size_vector != tflite_input_shape) {
      // Resize input tensors based on current total batch size
      allocate_tensors = true;
      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          (std::string(
               "resizing input " + std::string(input_name) +
               " with total batch size: " + std::to_string(total_batch_size) +
               "\n"))
              .c_str());
      interpreter_->ResizeInputTensor(
          model_state_->input_index_map_[input_name],
          batchn_tflite_size_vector);
    }
  }

  // Once we have resized all input tensors in the loop above,
  // now we can allocate the memory plan within the tflite runtime if
  // necessary
  if (allocate_tensors || first_inference_) {
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
      SendErrorForResponses(
          responses, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              "TfLite interpreter failed to allocate tensor inputs"));
    }
  }

  // With the memory now allocated appropriately for all input tensors, we can
  // call process tensor for each
  for (uint32_t input_idx = 0; input_idx < input_count; input_idx++) {
    TRITONBACKEND_Input* input;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input));

    const char* input_name;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        TRITONBACKEND_InputProperties(
            input, &input_name, nullptr, nullptr, nullptr, nullptr, nullptr));

    // Even if running on MALI GPU, we use CPU memory
    std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> alloc_perference;
    alloc_perference = {{TRITONSERVER_MEMORY_CPU, 0}};

    const char* input_buffer;
    size_t batchn_byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    TfLiteTensor* tflite_input_tensor =
        interpreter_->tensor(model_state_->input_index_map_[input_name]);
    char* tflite_input_buffer = tflite_input_tensor->data.raw;

    // Here we use ProcessTensor to copy the data from triton into the buffer
    // allocated by the tflite interpreter. I don't believe the data copy can
    // be avoided using the tflite runtime
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        collector->ProcessTensor(
            input_name, tflite_input_buffer, tflite_input_tensor->bytes,
            alloc_perference, &input_buffer, &batchn_byte_size, &memory_type,
            &memory_type_id));
  }

  // Finalize Backend Input Collector...
  collector->Finalize();
}

void
ModelInstanceState::ReadOutputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  BackendOutputResponder responder(
      requests, request_count, responses, model_state_->MaxBatchSize(),
      model_state_->TritonMemoryManager(), false, nullptr);

  for (const auto& map_entry : model_state_->output_index_map_) {
    std::string output_name = map_entry.first;
    int tensor_index = map_entry.second;

    TfLiteTensor* tflite_output_tensor = interpreter_->tensor(tensor_index);

    // Verify output datatype matches datatype from model config
    TRITONSERVER_DataType output_dtype =
        ConvertTFLiteTypeToDataType(tflite_output_tensor->type);
    TRITONSERVER_DataType config_datatype =
        model_state_->output_dtype_map_[output_name];
    if (config_datatype != output_dtype) {
      RESPOND_ALL_AND_RETURN_IF_ERROR(
          responses, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("unexpected datatype TYPE_") +
               TRITONSERVER_DataTypeString(output_dtype) +
               " for inference output '" + output_name + "', expecting TYPE_" +
               TRITONSERVER_DataTypeString(config_datatype))
                  .c_str()));
    }

    // Assign data pointer to head of data container for output tensor
    const char* output_buffer =
        static_cast<const char*>(tflite_output_tensor->data.raw);

    // Set output shape
    std::vector<int64_t> batchn_shape;
    TfLiteIntArray* dims = tflite_output_tensor->dims;
    for (int32_t i = 0; i < dims->size; i++) {
      batchn_shape.push_back(dims->data[i]);
    }

    responder.ProcessTensor(
        output_name, output_dtype, batchn_shape, output_buffer,
        TRITONSERVER_MEMORY_CPU, 0);
  }

  // Finalize and wait for any pending buffer copies.
  responder.Finalize();
}

/////////////

extern "C" {

int32_t armnn_threads = INT_MAX;

TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
#ifdef PAPI_PROFILING_ENABLE
  // Init PAPI library
  RETURN_ERROR_IF_FALSE(
      PAPI_library_init(PAPI_VER_CURRENT) == PAPI_VER_CURRENT,
      TRITONSERVER_ERROR_UNAVAILABLE, std::string("Failed to init PAPI lib"));
  RETURN_ERROR_IF_FALSE(
      PAPI_thread_init(pthread_self) == PAPI_OK, TRITONSERVER_ERROR_UNAVAILABLE,
      std::string("Failed to init PAPI thread lib"));

  // The backend configuration may contain information needed by the
  // backend, such a command-line arguments.
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));
  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend configuration:\n") + buffer).c_str());
  triton::common::TritonJson::Value backend_config;
  if (byte_size != 0) {
    RETURN_IF_ERROR(backend_config.Parse(buffer, byte_size));
  }
  triton::common::TritonJson::Value cmdline;
  if (backend_config.Find("cmdline", &cmdline)) {
    triton::common::TritonJson::Value value;
    std::string value_str;
    if (cmdline.Find("papi-events", &value)) {
      RETURN_IF_ERROR(value.AsString(&value_str));
      std::stringstream ss(value_str);
      while (ss.good()) {
        std::string substr;
        std::getline(ss, substr, ',');
        // Validate counter is a valid papi counter
        RETURN_ERROR_IF_FALSE(
            PAPIEventValid(substr), TRITONSERVER_ERROR_INVALID_ARG,
            std::string("PAPI event '") + substr +
                "' is requested but invalid");
      }
      // Set environment for papi to do high level op profiling
      RETURN_ERROR_IF_TRUE(
          setenv("PAPI_EVENTS", value_str.c_str(), 1),
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string("Could not set PAPI_EVENTS env variable"));
    }
  }
#endif  // PAPI_PROFILING_ENABLE

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
  RETURN_IF_ERROR(ModelState::Create(model, &model_state, &armnn_threads));
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
      ModelInstanceState::Create(model_state, instance, &instance_state));
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
