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

/*!
 * \brief ModelInstance for backend end execution of model.
 *
 * Tensorpipe Receiver is the communicator implemented by tcp.
 */
class ModelInstance {
 public:
  /*!
   * \brief Receiver constructor
   */
  ModelInstance()
  {
    context_ = std::make_shared<tensorpipe::Context>();
    auto transportContext = tensorpipe::transport::shm::create();
    context_->registerTransport(0 /* priority */, "shm", transportContext);
    // Register basic shm channel
    auto basicChannel = tensorpipe::channel::basic::create();
    context_->registerChannel(0 /* low priority */, "basic", basicChannel);
  }

  /*!
   * \brief ModelInstance destructor
   */
  ~ModelInstance() { Finalize(); }

  /*!
   * \brief Start server
   * \param addr Networking address, e.g., 'tcp://127.0.0.1:50051'
   */
  void Start(const std::string& addr);

  /*!
   * \brief Finalize ModelInstance
   *
   * Finalize() is not thread-safe and only one thread can invoke this API.
   */
  void Finalize();

  /*!
   * \brief Issue a receive request pipe
   */
  void ReceiveFromPipe();

 private:
  /*!
   * \brief Callback for new connection is accepted.
   */
  void OnAccepted(const tensorpipe::Error&, std::shared_ptr<tensorpipe::Pipe>);

  /*!
   * \brief Callback for loading a tflite model.
   */
  void LoadModelFromPipe(tensorpipe::Descriptor descriptor);

  TfLiteStatus BuildInterpreter(tensorpipe::Descriptor descriptor);

  void LogDelegation(const std::string& delegate_name);

  /*!
   * \brief Callback for inferencing on a loaded tflite model.
   */
  void Infer(tensorpipe::Descriptor& descriptor);

  /*!
   * \brief global context of tensorpipe
   */
  std::shared_ptr<tensorpipe::Context> context_;

  /*!
   * \brief pipe for client connection
   */
  std::shared_ptr<tensorpipe::Pipe> pipe_;

  /*!
   * \brief listener to build pipe
   */
  std::shared_ptr<tensorpipe::Listener> listener_{nullptr};

  /*!
   * \brief tflite interpreter
   */
  std::unique_ptr<tflite::Interpreter> interpreter_;

  /*!
   * \brief tflite model
   */
  std::unique_ptr<tflite::FlatBufferModel> model_;

  // Unique model instance name
  std::string model_instance_name_;

  // State variable to register whether inference has been called at least once
  bool first_inference_ = true;

#ifdef PAPI_PROFILING_ENABLE
  std::unique_ptr<tflite::Profiler> papi_profiler_ = MaybeCreatePapiProfiler();
#endif  // PAPI_PROFILING_ENABLE
};