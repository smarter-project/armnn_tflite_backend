//
// Copyright © 2021 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <stdint.h>
#include <exception>
#include <fstream>
#include <thread>
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
  virtual ~ModelState() = default;

  // Load a serialized tflite model using 'artifact_name' as the name for the
  // tflite model file. Return in 'model_path' the full path to the
  // tflite model file. Return in 'model' the TFLite network,
  // representing the model.
  TRITONSERVER_Error* LoadModel(
      const std::string& artifact_name, std::string* model_path,
      common::TritonJson::Value& model_config,
      std::unique_ptr<tflite::FlatBufferModel>* model);

  // Validate that model configuration is supported by this backend.
  // TRITONSERVER_Error* ValidateModelConfig();

#ifdef ARMNN_DELEGATE_ENABLE
  // ArmNN Delegate options
  bool use_armnn_delegate_cpu_ = false;
  bool use_armnn_delegate_gpu_ = false;
  armnn::OptimizerOptions armnn_optimizer_options_cpu_;
  armnn::OptimizerOptions armnn_optimizer_options_gpu_;
#endif  // ARMNN_DELEGATE_ENABLE

  // XNNPACK Delegate options
  bool use_xnnpack_delegate_ = false;
  int32_t num_threads_xnnpack_ = std::thread::hardware_concurrency();

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
}

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
                int32_t tuning_level = 0;
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
ModelState::AutoCompleteConfig()
{
  // Auto-complete configuration is not supported since TFLite does not
  // store/capture sufficient model metadata so just log error instead.
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
  TRITONSERVER_Error* ValidateBooleanSequenceControl(
      triton::common::TritonJson::Value& sequence_batching,
      const std::string& control_kind, bool required, bool* have_control);
  TRITONSERVER_Error* ValidateTypedSequenceControl(
      triton::common::TritonJson::Value& sequence_batching,
      const std::string& control_kind, bool required, bool* have_control);
  TRITONSERVER_Error* BuildInterpreter();
  TRITONSERVER_Error* ValidateInputs();
  TRITONSERVER_Error* ValidateOutputs();
  void Execute(
      std::vector<TRITONBACKEND_Response*>* responses,
      const uint32_t response_count);
  void SetInputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      BackendInputCollector* collector, std::vector<const char*>* input_names,
      std::vector<BackendMemory*>* input_memories);
  void ReadOutputTensors(
      size_t total_batch_size, const std::vector<const char*>& output_names,
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);

  ModelState* model_state_;

  // The full path to the TFLite model file.
  std::string model_path_;

  // The pointer to the tflite network
  std::unique_ptr<tflite::FlatBufferModel> model_;

  // The pointer to the tflite interpreter instance
  std::unique_ptr<tflite::Interpreter> interpreter_;

  // Map from configuration name for an input to the index of
  // that input in the model.
  std::unordered_map<std::string, int> input_index_map_;

  // Map from configuration name for an output to the index of
  // that output in the model.
  std::unordered_map<std::string, int> output_index_map_;
  std::unordered_map<std::string, TRITONSERVER_DataType> output_dtype_map_;
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
  // Validate the inputs and outputs from the model configuration
  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateInputs());
  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateOutputs());

  // Load the TFLite network
  THROW_IF_BACKEND_INSTANCE_ERROR(model_state->LoadModel(
      ArtifactFilename(), &model_path_, model_state->ModelConfig(), &model_));

  // Build interpreter
  THROW_IF_BACKEND_INSTANCE_ERROR(BuildInterpreter());

  // inputs/outputs hold the list of tensor indexes in the graph for each
  // respectively
  const std::vector<int> inputs = interpreter_->inputs();
  const std::vector<int> outputs = interpreter_->outputs();
  size_t num_inputs = inputs.size();
  size_t num_outputs = outputs.size();

  // Populate input name map
  for (size_t i = 0; i < num_inputs; i++) {
    input_index_map_[interpreter_->GetInputName(i)] = inputs[i];
  }

  // Populate output name and dtype map
  for (size_t i = 0; i < num_outputs; i++) {
    output_index_map_[interpreter_->GetOutputName(i)] = outputs[i];
    output_dtype_map_[interpreter_->GetOutputName(i)] =
        ConvertTFLiteTypeToDataType(interpreter_->tensor(outputs[i])->type);
  }
}

ModelInstanceState::~ModelInstanceState()
{
  // Consider the function ReleaseNonPersistentMemory here for our interpreter
  // delete &interpreter_;
}

TRITONSERVER_Error*
ModelInstanceState::BuildInterpreter()
{
  // Build the tflite interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model_, resolver);
  builder(&interpreter_);
  if (!interpreter_) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        ("failed to build tflite interpreter for model " + Name()).c_str());
  }

  // Tell interpreter to use max threads available to system
  if (interpreter_->SetNumThreads(std::thread::hardware_concurrency()) !=
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

    // Instruct the Interpreter to use the xnn pack
    if (interpreter_->ModifyGraphWithDelegate(std::move(xnnpack_delegate)) !=
        kTfLiteOk) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("failed to use xnnpack delegate for model " + Name()).c_str());
    }
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::ValidateInputs()
{
  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(model_state_->ModelConfig().MemberAsArray("input", &ios));

  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));

    // Fetch name of input
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));

    // Validate data type
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
    const auto pr = ModelConfigDataTypeToTFLiteType(io_dtype);
    if (!pr.first) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("unsupported datatype " + io_dtype + " for input '" + io_name +
           "' for model '" + model_state_->Name() + "'")
              .c_str());
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ValidateOutputs()
{
  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(model_state_->ModelConfig().MemberAsArray("output", &ios));

  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));

    // Fetch name of output
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));

    // Validate data type
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
    const auto pr = ModelConfigDataTypeToTFLiteType(io_dtype);
    if (!pr.first) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("unsupported datatype " + io_dtype + " for output '" + io_name +
           "' for model '" + model_state_->Name() + "'")
              .c_str());
    }
  }

  return nullptr;  // success
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
  if ((total_batch_size != 1) && (total_batch_size > (size_t)max_batch_size)) {
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

  std::vector<const char*> input_names;
  std::vector<BackendMemory*> input_memories;
  BackendInputCollector collector(
      requests, request_count, &responses, model_state_->TritonMemoryManager(),
      false, nullptr);
  // Note here we are copying the triton input buffers to the tflite allocated
  // buffers
  SetInputTensors(
      total_batch_size, requests, request_count, &responses, &collector,
      &input_names, &input_memories);

  // Request to retrieve all model outputs.
  std::vector<const char*> output_names;
  {
    triton::common::TritonJson::Value ios;
    TRITONSERVER_Error* err =
        model_state_->ModelConfig().MemberAsArray("output", &ios);
    if (err == nullptr) {
      for (size_t i = 0; i < ios.ArraySize(); i++) {
        triton::common::TritonJson::Value io;
        err = ios.IndexAsObject(i, &io);
        if (err != nullptr) {
          break;
        }

        // Use names from ModelConfig by reference since the model
        // config will persist longer than this inference execution.
        const char* io_name;
        size_t io_name_len;
        err = io.MemberAsString("name", &io_name, &io_name_len);
        if (err != nullptr) {
          break;
        }

        output_names.emplace_back(io_name);
      }
    }

    if (err != nullptr) {
      SendErrorForResponses(&responses, request_count, err);
      output_names.clear();
    }
  }

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

  ReadOutputTensors(
      total_batch_size, output_names, requests, request_count, &responses);

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
  if (interpreter_->Invoke() != kTfLiteOk) {
    SendErrorForResponses(
        responses, response_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, ("TFLite execute failure")));
  }
}

void
ModelInstanceState::SetInputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    BackendInputCollector* collector, std::vector<const char*>* input_names,
    std::vector<BackendMemory*>* input_memories)
{
  const int32_t max_batch_size = model_state_->MaxBatchSize();

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

    input_names->emplace_back(input_name);

    // The shape for the entire input patch, [total_batch_size, ...]
    std::vector<int64_t> batchn_shape(
        input_shape, input_shape + input_dims_count);
    if (max_batch_size != 0) {
      batchn_shape[0] = total_batch_size;
    }

    // Allocate memory for tensors
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
      SendErrorForResponses(
          responses, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              "TfLite interpreter failed to allocate tensor inputs"));
    }

    // Even if running on MALI GPU, we use CPU memory
    std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> alloc_perference;
    alloc_perference = alloc_perference = {{TRITONSERVER_MEMORY_CPU, 0}};

    const char* input_buffer;
    size_t batchn_byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    TfLiteTensor* tflite_input_tensor =
        interpreter_->tensor(input_index_map_[input_name]);
    char* tflite_input_buffer = tflite_input_tensor->data.raw;

    // Here we use ProcessTensor to copy the data from triton into the buffer
    // allocated by the tflite interpreter. I don't believe the data copy can be
    // avoided using the tflite runtime
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
    size_t total_batch_size, const std::vector<const char*>& output_names,
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  BackendOutputResponder responder(
      requests, request_count, responses, model_state_->MaxBatchSize(),
      model_state_->TritonMemoryManager(), false, nullptr);

  for (size_t idx = 0; idx < output_names.size(); idx++) {
    std::string name = output_names[idx];
    TfLiteTensor* tflite_output_tensor =
        interpreter_->tensor(output_index_map_[name]);

    // Verify output datatype matches datatype from model config
    TRITONSERVER_DataType output_dtype =
        ConvertTFLiteTypeToDataType(tflite_output_tensor->type);
    TRITONSERVER_DataType config_datatype = output_dtype_map_[name];
    if (config_datatype != output_dtype) {
      RESPOND_ALL_AND_RETURN_IF_ERROR(
          responses, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("unexpected datatype TYPE_") +
               TRITONSERVER_DataTypeString(output_dtype) +
               " for inference output '" + name + "', expecting TYPE_" +
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
        name, output_dtype, batchn_shape, output_buffer,
        TRITONSERVER_MEMORY_CPU, 0);
  }

  // Finalize and wait for any pending buffer copies.
  responder.Finalize();
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
  // RETURN_IF_ERROR(model_state->ValidateModelConfig());

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
