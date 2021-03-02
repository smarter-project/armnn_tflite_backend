// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <stdint.h>
#include <exception>
#include <fstream>
#include "armnn_utils.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonbackend.h"

// Suppress warnings in armnn headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma warning(push, 0)
#include <armnn/ArmNN.hpp>
#include <armnnDeserializer/IDeserializer.hpp>
#pragma warning(pop)
#pragma GCC diagnostic pop

//
// ArmNN Backend that implements the TRITONBACKEND API.
//

namespace triton { namespace backend { namespace armnnimpl {

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

  // Load a serialized armnn model using 'artifact_name' as the name for the
  // armnn model file. Return in 'model_path' the full path to the
  // armnn model file. Return in 'armnn_network' the ArmNN network,
  // representing the model. Return in
  // 'input_bindings_info'/'output_bindings_info' the input/output binding info
  // for the model
  TRITONSERVER_Error* LoadModel(
      const std::string& artifact_name, std::string* model_path,
      common::TritonJson::Value& model_config, armnn::INetworkPtr* network,
      std::vector<armnn::BindingPointInfo>& input_bindings_info,
      std::vector<armnn::BindingPointInfo>& output_bindings_info);

  // Validate that model configuration is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

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
    common::TritonJson::Value& model_config, armnn::INetworkPtr* network,
    std::vector<armnn::BindingPointInfo>& input_bindings_info,
    std::vector<armnn::BindingPointInfo>& output_bindings_info)
{
  // Find the ArmNN model file that describes the model. If the model
  // configuration doesn't have an explicit model file specified then
  // use the default name ("model.armnn").
  std::string cc_model_filename = artifact_name;
  if (cc_model_filename.empty()) {
    cc_model_filename = "model.armnn";
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

  try {
    // Fetch model data byte stream from filepath
    std::ifstream in(*model_path, std::ios::in | std::ios::binary);

    // Deserialize armnn saved model
    armnnDeserializer::IDeserializerPtr deserializer =
        armnnDeserializer::IDeserializer::Create();
    *network = deserializer->CreateNetworkFromBinary(in);

    // Set input tensor bindings
    triton::common::TritonJson::Value ios;
    RETURN_IF_ERROR(model_config.MemberAsArray("input", &ios));
    for (size_t i = 0; i < ios.ArraySize(); i++) {
      triton::common::TritonJson::Value io;
      RETURN_IF_ERROR(ios.IndexAsObject(i, &io));

      std::string io_name;
      RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
      armnnDeserializer::BindingPointInfo input_binding =
          deserializer->GetNetworkInputBindingInfo(0, io_name);
      input_bindings_info.push_back(std::make_pair(
          input_binding.m_BindingId, input_binding.m_TensorInfo));
    }

    // Set output tensor bindings
    RETURN_IF_ERROR(model_config.MemberAsArray("output", &ios));
    for (size_t i = 0; i < ios.ArraySize(); i++) {
      triton::common::TritonJson::Value io;
      RETURN_IF_ERROR(ios.IndexAsObject(i, &io));

      std::string io_name;
      RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
      armnnDeserializer::BindingPointInfo output_binding =
          deserializer->GetNetworkOutputBindingInfo(0, io_name);
      output_bindings_info.push_back(std::make_pair(
          output_binding.m_BindingId, output_binding.m_TensorInfo));
    }
  }
  catch (const std::exception& ex) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        ("failed to load model '" + Name() + "': " + ex.what()).c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  // ArmNN does not support batching for models. Tensor sizes must be fixed in
  // advance of model execution
  RETURN_ERROR_IF_TRUE(
      MaxBatchSize() > 0, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("batch size > 0 not supported by ArmNN"));

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::AutoCompleteConfig()
{
  // Auto-complete configuration is not supported since ArmNN does not
  // store/capture sufficient model metadata so just log error instead.
  LOG_MESSAGE(
      TRITONSERVER_LOG_WARN,
      (std::string("skipping model configuration auto-complete for '") +
       Name() + "': not supported for armnn backend")
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
  TRITONSERVER_Error* ValidateInputs();
  TRITONSERVER_Error* ValidateOutputs();
  void Execute(
      std::vector<TRITONBACKEND_Response*>* responses,
      const uint32_t response_count, armnn::InputTensors& input_tensors,
      armnn::OutputTensors& output_tensors);
  void SetTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      BackendInputCollector* collector, std::vector<const char*>* input_names,
      std::vector<const char*>* output_names,
      armnn::InputTensors& input_tensors, armnn::OutputTensors& output_tensors,
      std::vector<BackendMemory*>* input_memories,
      std::vector<BackendMemory*>* output_memories);
  void ReadOutputTensors(
      size_t total_batch_size, const std::vector<const char*>& output_names,
      armnn::OutputTensors& output_tensors, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);

  ModelState* model_state_;

  // The full path to the ArmNN model file.
  std::string model_path_;

  // The pointer to the armnn network
  armnn::INetworkPtr network_ = armnn::INetworkPtr(nullptr, nullptr);

  // These hold information on input and output tensor
  std::vector<armnn::BindingPointInfo> input_bindings_info_;
  std::vector<armnn::BindingPointInfo> output_bindings_info_;

  // The pointer to the ArmNN runtime for a single network
  armnn::IRuntimePtr runtime_ = armnn::IRuntimePtr(nullptr, nullptr);

  // The pointer to the optimized network for cpu ref, cpu neon, or mali gpu
  armnn::IOptimizedNetworkPtr opt_network_ =
      armnn::IOptimizedNetworkPtr(nullptr, nullptr);

  // Identification number for loaded network
  armnn::NetworkId network_identifier_;

  // Vector holding ArmNN backend execution preferences for model instance
  const std::vector<armnn::BackendId> backend_prefs_;

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
  // Set ArmNN backend prefs depending on triton instance group
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    const std::vector<armnn::BackendId> backend_prefs_ = {armnn::Compute::GpuAcc, armnn::Compute::CpuAcc,
                      armnn::Compute::CpuRef};
  } else {
     const std::vector<armnn::BackendId> backend_prefs_ = {armnn::Compute::CpuAcc, armnn::Compute::CpuRef};
  }

  // Validate the inputs and outputs from the model configuration
  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateInputs());
  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateOutputs());

  // Setup the ArmNN network and io binding point info
  THROW_IF_BACKEND_INSTANCE_ERROR(model_state->LoadModel(
      ArtifactFilename(), &model_path_, model_state->ModelConfig(), &network_,
      input_bindings_info_, output_bindings_info_));

  // TODO: inspect runtime creation options
  // TODO: it seems that one runtime pointer can index
  //       into multiple models (investigate)
  armnn::IRuntime::CreationOptions options;
  runtime_ = armnn::IRuntime::Create(options);

  // Optimize network for target backend
  opt_network_ =
      armnn::Optimize(*network_, backend_prefs_, runtime_->GetDeviceSpec());

  // Load network, which sets the network identifier
  runtime_->LoadNetwork(network_identifier_, std::move(opt_network_));
}

ModelInstanceState::~ModelInstanceState()
{
  runtime_->UnloadNetwork(network_identifier_);
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
    const auto pr = ModelConfigDataTypeToArmNNType(io_dtype);
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
    const auto pr = ModelConfigDataTypeToArmNNType(io_dtype);
    if (!pr.first) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("unsupported datatype " + io_dtype + " for output '" + io_name +
           "' for model '" + model_state_->Name() + "'")
              .c_str());
    }
    output_dtype_map_[io_name] = ConvertArmNNTypeToDataType(pr.second);
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

  const int max_batch_size = model_state_->MaxBatchSize();

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
                  "null request given to PyTorch backend for '" + Name() + "'")
                  .c_str()));
      return;
    }

    total_batch_size += 1;
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

  // Store input and output tensor data and metadata
  armnn::InputTensors input_tensors;
  armnn::OutputTensors output_tensors;

  std::vector<const char*> input_names;
  std::vector<const char*> output_names;

  std::vector<BackendMemory*> input_memories;
  std::vector<BackendMemory*> output_memories;

  BackendInputCollector collector(
      requests, request_count, &responses, model_state_->TritonMemoryManager(),
      false, nullptr);
  SetTensors(
      total_batch_size, requests, request_count, &responses, &collector,
      &input_names, &output_names, input_tensors, output_tensors,
      &input_memories, &output_memories);

  // Wait for any in-flight input tensor copies to complete.
  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  // Run...
  Execute(&responses, request_count, input_tensors, output_tensors);

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  // Free BackendMemory used for inputs
  for (BackendMemory* mem : input_memories) {
    delete mem;
  }
  input_memories.clear();

  // Verify output indices are valid with number of outputs after execution
  bool invalid_index = false;
  int max_index = output_tensors.size() - 1;
  for (const auto& name : output_names) {
    int op_index = output_index_map_[name];
    if ((op_index < 0) || (op_index > max_index)) {
      SendErrorForResponses(
          &responses, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              std::string(
                  "The output " + std::string(name) +
                  " in the model configuration refers to an output index which"
                  " doesn't exist. This model has " +
                  std::to_string(max_index + 1) + " outputs")
                  .c_str()));
      invalid_index = true;
      break;
    }
  }

  if (!invalid_index) {
    ReadOutputTensors(
        total_batch_size, output_names, output_tensors, requests, request_count,
        &responses);
  }

  // Free BackendMemory used for outputs
  for (BackendMemory* mem : output_memories) {
    delete mem;
  }
  output_memories.clear();

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
          "failed to send PyTorch backend response");
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
    const uint32_t response_count, armnn::InputTensors& input_tensors,
    armnn::OutputTensors& output_tensors)
{
  armnn::Status ret = runtime_->EnqueueWorkload(
      network_identifier_, input_tensors, output_tensors);
  if (ret == armnn::Status::Failure) {
    SendErrorForResponses(
        responses, response_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, ("ArmNN execute failure")));
  }
}

void
ModelInstanceState::SetTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    BackendInputCollector* collector, std::vector<const char*>* input_names,
    std::vector<const char*>* output_names, armnn::InputTensors& input_tensors,
    armnn::OutputTensors& output_tensors,
    std::vector<BackendMemory*>* input_memories,
    std::vector<BackendMemory*>* output_memories)
{
  const int max_batch_size = model_state_->MaxBatchSize();

  // All requests must have equally-sized input tensors so use any
  // request as the representative for the input tensors.
  uint32_t input_count;
  RESPOND_ALL_AND_RETURN_IF_ERROR(
      responses, request_count,
      TRITONBACKEND_RequestInputCount(requests[0], &input_count));
  int i = 0;
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

    // The input must be in contiguous CPU memory.
    const int64_t batchn_byte_size = GetByteSize(input_datatype, batchn_shape);

    // Even if running on MALI GPU, we use CPU memory
    std::vector<BackendMemory::AllocationType> alloc_perference;
    alloc_perference = {BackendMemory::AllocationType::CPU};

    // Allocate memory for inputs
    BackendMemory* input_memory;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        BackendMemory::Create(
            model_state_->TritonMemoryManager(), alloc_perference, 0,
            batchn_byte_size, &input_memory));
    input_memories->push_back(input_memory);

    TRITONSERVER_MemoryType memory_type = input_memory->MemoryType();
    int64_t memory_type_id = input_memory->MemoryTypeId();
    char* input_buffer = input_memory->MemoryPtr();

    collector->ProcessTensor(
        input_name, input_buffer, batchn_byte_size, memory_type,
        memory_type_id);

    // Create ArmNN tenor
    // Create input binding info for input tensor
    const armnn::BindingPointInfo& inputBinding = input_bindings_info_[i];

    // Const tensor created from tensor info and data pointer to triton
    // allocated buffer
    armnn::ConstTensor input_tensor(inputBinding.second, input_buffer);
    input_tensors.push_back(std::make_pair(inputBinding.first, input_tensor));

    ++i;
  }

  // Now handle buffer allocation for output tensors

  // Loop through the output tensor config info
  // There may be use for a utility function here the triton backend library
  i = 0;
  triton::common::TritonJson::Value ios;
  RESPOND_ALL_AND_RETURN_IF_ERROR(
      responses, request_count,
      model_state_->ModelConfig().MemberAsArray("output", &ios));
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count, ios.IndexAsObject(i, &io));

    // Use names from ModelConfig by reference since the model
    // config will persist longer than this inference execution.
    const char* io_name;
    size_t io_name_len;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        io.MemberAsString("name", &io_name, &io_name_len));

    // Fetch data type of output
    std::string io_dtype;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count, io.MemberAsString("data_type", &io_dtype));
    TRITONSERVER_DataType output_datatype =
        TRITONSERVER_StringToDataType(io_dtype.c_str());

    output_names->emplace_back(io_name);

    // Create output binding info for output tensor
    const armnn::BindingPointInfo& outputBinding = output_bindings_info_[i];

    // Get output shape
    std::vector<int64_t> batchn_shape;
    armnn::TensorShape shape = outputBinding.second.GetShape();
    for (size_t j = 0; j < shape.GetNumDimensions(); ++j) {
      batchn_shape.push_back(shape[j]);
    }
    
    // The output must be in contiguous CPU memory.
    const int64_t batchn_byte_size = GetByteSize(output_datatype, batchn_shape);

    // Even if running on MALI GPU, we use CPU memory
    std::vector<BackendMemory::AllocationType> alloc_perference;
    alloc_perference = {BackendMemory::AllocationType::CPU};

    // Allocate memory for outputs
    BackendMemory* output_memory;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        BackendMemory::Create(
            model_state_->TritonMemoryManager(), alloc_perference, 0,
            batchn_byte_size, &output_memory));
    output_memories->push_back(output_memory);

    char* output_buffer = output_memory->MemoryPtr();

    // Const tensor created from tensor info and data pointer to triton
    // allocated buffer
    armnn::Tensor output_tensor(outputBinding.second, output_buffer);
    output_tensors.push_back(
        std::make_pair(outputBinding.first, output_tensor));

    ++i;
  }

  // Finalize Backend Input Collector...
  collector->Finalize();
}

void
ModelInstanceState::ReadOutputTensors(
    size_t total_batch_size, const std::vector<const char*>& output_names,
    armnn::OutputTensors& output_tensors, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  BackendOutputResponder responder(
      requests, request_count, responses, model_state_->MaxBatchSize(),
      model_state_->TritonMemoryManager(), false, nullptr);

  for (size_t idx = 0; idx < output_names.size(); idx++) {
    std::string name = output_names[idx];
    armnn::TensorInfo ti;

    try {
      // verify output datatype matches datatype from model config
      ti = output_tensors[idx].second.GetInfo();
    }
    catch (std::exception& ex) {
      RESPOND_ALL_AND_RETURN_IF_ERROR(
          responses, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              (std::string("output tensor '") + name + "' is not found")
                  .c_str()));
    }

    // Verify output datatype matches datatype from model config
    TRITONSERVER_DataType output_dtype =
        ConvertArmNNTypeToDataType(ti.GetDataType());
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
        static_cast<const char*>(output_tensors[idx].second.GetMemoryArea());

    // Set output shape
    std::vector<int64_t> batchn_shape;
    armnn::TensorShape shape = output_tensors[idx].second.GetShape();
    for (size_t i = 0; i < shape.GetNumDimensions(); ++i) {
      batchn_shape.push_back(shape[i]);
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

}}}  // namespace triton::backend::armnnimpl
