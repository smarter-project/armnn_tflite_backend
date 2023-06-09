//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <signal.h>

#include <atomic>
#include <future>

#include "model_instance.h"

// Triton backend headers
#include "triton/backend/backend_common.h"

#ifdef PAPI_PROFILING_ENABLE
#include "papi.h"
#endif  // PAPI_PROFILING_ENABLE

int
main(int argc, char* argv[])
{
#ifdef PAPI_PROFILING_ENABLE
  // Init PAPI library
  if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Failed to init PAPI lib");
    return 1;
  }
  if (PAPI_thread_init(pthread_self) != PAPI_OK) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Failed to init PAPI thread lib");
    return 1;
  }
#endif  // PAPI_PROFILING_ENABLE

  // Parse listen address
  if (argc != 2) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_ERROR,
        "Args should be model_instance <bind "
        "address>");

    return 1;
  }
  const char* addr = argv[1];

  // block signals in this thread and subsequently
  // spawned threads
  sigset_t sigset;
  sigemptyset(&sigset);
  sigaddset(&sigset, SIGINT);
  sigaddset(&sigset, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &sigset, nullptr);

  std::atomic<bool> shutdown_requested(false);
  std::mutex cv_mutex;
  std::condition_variable cv;

  auto signal_handler = [&shutdown_requested, &cv, &sigset]() {
    int signum = 0;
    // wait until a signal is delivered:
    sigwait(&sigset, &signum);
    shutdown_requested.store(true);
    // notify all waiting workers to check their predicate:
    cv.notify_all();
    return signum;
  };

  auto ft_signal_handler = std::async(std::launch::async, signal_handler);

  ModelInstance model_instance;

  // Will connect to the address provided as the first argument in the list
  model_instance.Start(std::string(addr));

  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "Model instance running...");

  // wait for signal handler to complete
  int signal = ft_signal_handler.get();
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Received signal: ") + std::to_string(signal)).c_str());

  return 0;
}