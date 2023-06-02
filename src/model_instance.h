//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorpipe/tensorpipe.h"

#ifdef PAPI_PROFILING_ENABLE
#include "papi.h"
#include "papi_profiler.h"
#endif  // PAPI_PROFILING_ENABLE

// ModelInstance for backend end execution of model
class ModelInstance {
 public:
  ModelInstance()
  {
    context_ = std::make_shared<tensorpipe::Context>();
    auto transportContext = tensorpipe::transport::shm::create();
    context_->registerTransport(0 /* priority */, "shm", transportContext);
    // Register cma shm channel
    auto cmaChannel = tensorpipe::channel::cma::create();
    context_->registerChannel(0 /* low priority */, "cma", cmaChannel);
  }

  ~ModelInstance() { Finalize(); }

  // Start model instance and attempt to connect to passed address
  void Start(const std::string& addr);

  // Cleanup
  void Finalize();

  // Issue a receive request pipe
  void ReceiveFromPipe();

 private:
  // Callback for new connection is accepted.
  void OnAccepted(const tensorpipe::Error&, std::shared_ptr<tensorpipe::Pipe>);

  // Callback for loading a tflite model.
  void LoadModelFromPipe(tensorpipe::Descriptor descriptor);

  // Builds the tflite interpreter based on passed descriptor
  TfLiteStatus BuildInterpreter(tensorpipe::Descriptor descriptor);

  void LogDelegation(const std::string& delegate_name);

  // Callback for inferencing on a loaded tflite model.
  void Infer(tensorpipe::Descriptor& descriptor);

  // Global tensorpipe context
  std::shared_ptr<tensorpipe::Context> context_;

  // Pipe for client connection
  std::shared_ptr<tensorpipe::Pipe> pipe_;

  // Tflite interpreter
  std::unique_ptr<tflite::Interpreter> interpreter_;

  // Tflite model
  std::unique_ptr<tflite::FlatBufferModel> model_;

  // Unique model instance name
  std::string model_instance_name_;

  // State variable to register whether inference has been called at least once
  bool first_inference_ = true;

  // Tensorpipe allocation that we can reuse to write inputs into
  tensorpipe::Allocation allocation_;

  // Tensorpipe response message we can reuse to write outputs into
  tensorpipe::Message tp_response_msg_;

#ifdef PAPI_PROFILING_ENABLE
  std::unique_ptr<tflite::Profiler> papi_profiler_ = MaybeCreatePapiProfiler();
#endif  // PAPI_PROFILING_ENABLE
};