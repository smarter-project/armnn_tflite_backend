//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "model_instance.h"

#include <future>
#include <unordered_set>

#include "model_instance_utils.h"

// Triton backend headers
#include "triton/backend/backend_common.h"

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

void
ModelInstance::Finalize()
{
  pipe_->close();
}

void
ModelInstance::Start(const std::string& addr)
{
  pipe_ = context_->connect(addr);
  ReceiveFromPipe();
}

TfLiteStatus
ModelInstance::BuildInterpreter(tensorpipe::Descriptor descriptor)
{
  // Build the tflite interpreter
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  tflite::InterpreterBuilder builder(*model_, resolver);
  builder(&interpreter_);
  if (!interpreter_) {
    return kTfLiteError;
  }

  // Set interpreter threads
  if (interpreter_->SetNumThreads(std::stoi(
          descriptor.payloads[OptimizerOption::TFLITE_NUM_THREADS].metadata)) !=
      kTfLiteOk) {
    return kTfLiteError;
  }

  // Set numa parameters
  numa_alloc_policy_ = AllocationPolicyFromString(
      descriptor.payloads[OptimizerOption::NUMA_ALLOC_POLICY].metadata);

  local_numa_node_id_ = std::stoi(
      descriptor.payloads[OptimizerOption::NUMA_LOCAL_NODE_ID].metadata);

  remote_numa_node_id_ = std::stoi(
      descriptor.payloads[OptimizerOption::NUMA_REMOTE_NODE_ID].metadata);

#ifdef ARMNN_DELEGATE_ENABLE
  armnn::OptimizerOptions armnn_optimizer_options_cpu;
  armnn::OptimizerOptions armnn_optimizer_options_gpu;
  bool armnn_cpu_delegate_enabled =
      descriptor.payloads[OptimizerOption::ARMNN_CPU_ENABLE].metadata ==
      std::string("y");

  bool armnn_gpu_delegate_enabled =
      descriptor.payloads[OptimizerOption::ARMNN_GPU_ENABLE].metadata ==
      std::string("y");

  if (armnn_cpu_delegate_enabled || armnn_gpu_delegate_enabled) {
    armnnDelegate::DelegateOptions armnn_delegate_options =
        armnnDelegate::TfLiteArmnnDelegateOptionsDefault();

    // Set backend prefs based on gpu or cpu selection
    if (armnn_gpu_delegate_enabled) {
      armnn_delegate_options.SetBackends(
          {armnn::Compute::GpuAcc, armnn::Compute::CpuAcc});
      armnn_optimizer_options_gpu.m_ReduceFp32ToFp16 =
          descriptor.payloads[OptimizerOption::ARMNN_GPU_REDUCE_FP32_TO_FP16]
              .metadata == std::string("on");
      armnn_optimizer_options_gpu.m_ReduceFp32ToBf16 =
          descriptor.payloads[OptimizerOption::ARMNN_GPU_REDUCE_FP32_TO_BF16]
              .metadata == std::string("on");
      armnn::BackendOptions gpu_fast_math_option(
          "GpuAcc",
          {{"FastMathEnabled",
            descriptor.payloads[OptimizerOption::ARMNN_GPU_FAST_MATH_ENABLED]
                    .metadata == std::string("on")}});
      armnn_optimizer_options_gpu.m_ModelOptions.push_back(
          gpu_fast_math_option);
      armnn_delegate_options.SetOptimizerOptions(armnn_optimizer_options_gpu);
    } else {
      // Set backend pref to Neon ACL backend
      armnn_delegate_options.SetBackends({armnn::Compute::CpuAcc});
      armnn_optimizer_options_cpu.m_ReduceFp32ToFp16 =
          descriptor.payloads[OptimizerOption::ARMNN_CPU_REDUCE_FP32_TO_FP16]
              .metadata == std::string("on");
      armnn_optimizer_options_cpu.m_ReduceFp32ToBf16 =
          descriptor.payloads[OptimizerOption::ARMNN_CPU_REDUCE_FP32_TO_BF16]
              .metadata == std::string("on");
      armnn::BackendOptions cpu_fast_math_option(
          "CpuAcc",
          {{"FastMathEnabled",
            descriptor.payloads[OptimizerOption::ARMNN_CPU_FAST_MATH_ENABLED]
                    .metadata == std::string("on")}});
      armnn_optimizer_options_cpu.m_ModelOptions.push_back(
          cpu_fast_math_option);
      armnn::BackendOptions num_threads_option(
          "CpuAcc",
          {{"NumberOfThreads",
            static_cast<unsigned int>(std::stoi(
                descriptor.payloads[OptimizerOption::ARMNN_CPU_NUM_THREADS]
                    .metadata))}});
      armnn_optimizer_options_cpu.m_ModelOptions.push_back(num_threads_option);
      armnn_delegate_options.SetOptimizerOptions(armnn_optimizer_options_cpu);
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
      return kTfLiteError;
    }
    LogDelegation("armnn");
  } else if (
      descriptor.payloads[OptimizerOption::XNNPACK_ENABLE].metadata ==
      std::string("y")) {
#else
  if (descriptor.payloads[OptimizerOption::XNNPACK_ENABLE].metadata ==
      std::string("y")) {
#endif  // ARMNN_DELEGATE_ENABLE
    // Create the XNNPack Delegate
    TfLiteXNNPackDelegateOptions options =
        TfLiteXNNPackDelegateOptionsDefault();

    options.num_threads = std::stoi(
        descriptor.payloads[OptimizerOption::XNNPACK_CPU_NUM_THREADS].metadata);

    tflite::Interpreter::TfLiteDelegatePtr xnnpack_delegate(
        TfLiteXNNPackDelegateCreate(&options),
        [](TfLiteDelegate* xnnpack_delegate) {
          TfLiteXNNPackDelegateDelete(xnnpack_delegate);
        });

    // Instruct the Interpreter to use the xnnpack
    if (interpreter_->ModifyGraphWithDelegate(std::move(xnnpack_delegate)) !=
        kTfLiteOk) {
      return kTfLiteError;
    }
    LogDelegation("xnnpack");
  } else {
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, "No delegates used for model execution");
  }

  return kTfLiteOk;
}

void
ModelInstance::LogDelegation(const std::string& delegate_name)
{
  std::unordered_set<unsigned int> checked_node_ids;
  unsigned int num_delegated_kernels = 0;
  for (uint64_t i = 0; i < interpreter_->execution_plan().size(); i++) {
    int node_id = interpreter_->execution_plan()[i];
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
ModelInstance::ReceiveFromPipe()
{
  pipe_->readDescriptor([this](
                            const tensorpipe::Error& error,
                            tensorpipe::Descriptor descriptor) {
    if (error) {
      if (error.isOfType<tensorpipe::PipeClosedError>()) {
        // Expected.
        LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            (std::string("Remote side hungup: ") + error.what()).c_str());
      } else {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("Unexpected error when reading from accepted pipe: ") +
             error.what())
                .c_str());
      }
      return;
    }
    if (descriptor.metadata == "model_load") {
      LoadModelFromPipe(descriptor);
    } else if (descriptor.metadata == "model_input") {
      Infer(descriptor);
    }
  });
}

void
ModelInstance::LoadModelFromPipe(tensorpipe::Descriptor descriptor)
{
  // TODO: Make sure this can only be called once as it loads the model and
  // builds the interpreter
  tensorpipe::Allocation allocation;
  allocation.payloads.resize(descriptor.payloads.size());
  allocation.payloads[OptimizerOption::COUNT].data =
      new char[descriptor.payloads[OptimizerOption::COUNT].length];
  pipe_->read(
      allocation,
      [this, descriptor, allocation](const tensorpipe::Error& error) {
        if (error) {
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              ("Failed to read model from pipe with err:" + error.what())
                  .c_str());
          return;
        }
        // Load the tflite model from the buffer
        tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates
            builtin_op_resolver;
        model_ = tflite::FlatBufferModel::BuildFromBuffer(
            reinterpret_cast<char*>(
                allocation.payloads[OptimizerOption::COUNT].data),
            descriptor.payloads[OptimizerOption::COUNT].length);

        // Initalize the interpreter after loading the flatbuffers model
        tensorpipe::Message tp_msg;
        bool success = BuildInterpreter(descriptor) == kTfLiteOk;
        tp_msg.metadata = success ? "success" : "fail";
        pipe_->write(tp_msg, [this, success](const tensorpipe::Error& error) {
          if (error) {
            LOG_MESSAGE(
                TRITONSERVER_LOG_ERROR,
                ("Failed send model load ack:" + error.what()).c_str());
            return;
          }
#ifdef LIBNUMA_ENABLE
          if (success) {
            // Model is loaded, apply the numa policy
            InitNuma(local_numa_node_id_, remote_numa_node_id_);
          }
#endif  // LIBNUMA_ENABLE
        });

        // Arm for getting more data
        ReceiveFromPipe();
      });
}

void
ModelInstance::Infer(tensorpipe::Descriptor& descriptor)
{
  bool allocate_tensors = first_inference_;
  bool success = true;

  if (first_inference_) {
    allocation_.tensors.resize(descriptor.tensors.size());
    tp_response_msg_.tensors.resize(interpreter_->outputs().size());
  }

  // Get model inputs from request and ready the buffers (Allocation obj) to
  // write tensor data
  for (uint64_t i = 0; i < descriptor.tensors.size(); ++i) {
    // If the size of the incoming tensor
    // is different from the last call, tell the interpreter to resize the
    // input tensor and note that we are going to have to make another call to
    // AllocateTensors below

    // First element of tensor_info is input tensor index, remaining is the dims
    // of the input tensor
    int input_tensor_index = std::stoi(descriptor.tensors[i].metadata);

    // incoming_length holds the num bytes of the incoming vector
    int incoming_length = descriptor.tensors[i].length;

    int tflite_input_tensor_len =
        interpreter_->tensor(input_tensor_index)->bytes;

    if (incoming_length != tflite_input_tensor_len) {
      // Resize input tensors based on current total batch size
      TfLiteIntArray* tflite_input_tensor_dims =
          interpreter_->tensor(input_tensor_index)->dims;
      std::vector<int> tflite_input_shape(
          tflite_input_tensor_dims->data,
          (tflite_input_tensor_dims->data + tflite_input_tensor_dims->size));

      allocate_tensors = true;

      // Set the new batch size
      tflite_input_shape[0] = incoming_length > tflite_input_tensor_len
                                  ? incoming_length / tflite_input_tensor_len
                                  : tflite_input_tensor_len / incoming_length;

      interpreter_->ResizeInputTensor(input_tensor_index, tflite_input_shape);
    }
  }

  // Once we have resized all input tensors in the loop above,
  // now we can allocate the memory plan within the tflite runtime if
  // necessary
  if (allocate_tensors || first_inference_) {
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
      success = false;
    }

    // Assign Cpu buffers to read incoming tensor bytes into after allocate
    // tensors is called
    for (uint64_t i = 0; i < descriptor.tensors.size(); ++i) {
      allocation_.tensors[i].buffer = tensorpipe::CpuBuffer{
          .ptr = interpreter_->tensor(std::stoi(descriptor.tensors[i].metadata))
                     ->data.raw};
    }
  }

  pipe_->read(
      allocation_,
      [this, &success, allocate_tensors](const tensorpipe::Error& error) {
        success = !error;

        // At this point our input tensors should be written to by the read
        // function, now we invoke the interpreter and read the output
        if (interpreter_->Invoke() != kTfLiteOk) {
          success = false;
        } else {
#ifdef PAPI_PROFILING_ENABLE
          // After the first inference, all threads should be alive to profile
          if (first_inference_) {
            papi_profiler_ = MaybeCreatePapiProfiler();
            interpreter_->AddProfiler(papi_profiler_.get());
          }
#endif  // PAPI_PROFILING_ENABLE
        }

        first_inference_ = false;

        // Write output back to client
        if (!success) {
          tp_response_msg_.metadata = "f";
        } else if (allocate_tensors) {
          // If we (re)allocated tensors then we need to update response message
          for (uint64_t i = 0; i < interpreter_->outputs().size(); ++i) {
            int output_index = interpreter_->outputs()[i];
            TfLiteTensor* output_tensor = interpreter_->tensor(output_index);
            tensorpipe::Message::Tensor tensor;
            // We use the output tensor name as the metadata in the request
            tensor.metadata = std::string(output_tensor->name);
            tensor.length = output_tensor->bytes;
            tensor.buffer =
                tensorpipe::CpuBuffer{.ptr = output_tensor->data.raw};
            tp_response_msg_.tensors[i] = tensor;
          }
        }
        pipe_->write(tp_response_msg_, [](const tensorpipe::Error& error) {
          if (error) {
            LOG_MESSAGE(
                TRITONSERVER_LOG_ERROR,
                ("Failed to send inference response to client. Details:" +
                 error.what())
                    .c_str());
          }
        });
        // Arm for getting more data
        ReceiveFromPipe();
      });
}

#ifdef LIBNUMA_ENABLE
void
ModelInstance::InitNuma(int local_node_id, int remote_node_id)
{
  if (numa_alloc_policy_ == AllocationPolicy::NONE) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN, "Allocation policy ignored, policy is NONE");
    return;
  }

  if (numa_available() < 0) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN,
        "System does not support NUMA API, Allocation policy ignored");
    return;
  } else if (num_numa_nodes_ < 2) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN,
        "Only one numa node available to system. Allocation policy "
        "ignored\n");
    return;
  }

  // Set numa mem pollicies
  // In the case of the split policies, we need to explictly move the pages of
  // the weights to the target numa node, as it goes against the set memory
  // policy this process was launched with
  switch (numa_alloc_policy_) {
    case AllocationPolicy::WEIGHT_LOCAL_RESULT_REMOTE: {
      MoveModelWeights(local_node_id);
      break;
    }
    case AllocationPolicy::WEIGHT_REMOTE_RESULT_LOCAL: {
      MoveModelWeights(remote_node_id);
      break;
    }
    default: {
      break;
    }
  }
}

void
ModelInstance::MoveModelWeights(int numa_node_id)
{
  // Get pointer to base of mmapped model file
  const void* model_file_base = model_->allocation()->base();
  int page_size = getpagesize();

  std::vector<void*> pages(model_->allocation()->bytes() / page_size + 1);

  char* begin = (char*)model_file_base;
  char* end = begin + model_->allocation()->bytes();

  int i = 0;
  for (char* piter = (char*)AlignPage(model_file_base); piter < end;
       piter += page_size) {
    pages[i++] = (void*)piter;
  }

  std::vector<int> dst(pages.size(), numa_node_id);
  std::vector<int> status(pages.size(), 0);

  // Touch all pages of the file to force mapping to phys mem
  volatile char c;
  for (char* piter = (char*)AlignPage(model_file_base); piter < end;
       piter += page_size) {
    c = *piter;
  }
  // This is just to avoid the unused var compiler warning
  (void)c;

  // With all pages mapped, now move them to target numa node
  int ret = numa_move_pages(
      0, pages.size(), pages.data(), dst.data(), status.data(),
      MPOL_MF_MOVE_ALL);

  if (ret < 0) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_ERROR,
        (std::string("Numa move page error: ") + strerror(errno)).c_str());
    for (auto& i : status) {
      if (i < 0) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("Page error status: ") + strerror(i)).c_str());
      }
    }
  }
}
#endif  // LIBNUMA_ENABLE
