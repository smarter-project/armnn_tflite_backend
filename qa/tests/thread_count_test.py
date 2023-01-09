# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
from math import pow, log2
import multiprocessing

import psutil

import tritonclient.http as httpclient

from itertools import product

from helpers.triton_model_config import Model, TFLiteTritonModel
from helpers.helper_functions import load_model, get_random_triton_inputs


@pytest.mark.parametrize(
    "model_config",
    [
        TFLiteTritonModel(
            "mobilenet_v1_1.0_224",
            [Model.TensorIO("input", "TYPE_FP32", [1, 224, 224, 3])],
            [
                Model.TensorIO(
                    "MobilenetV1/Predictions/Reshape_1",
                    "TYPE_FP32",
                    [1001],
                    label_filename="labels.txt",
                )
            ],
            max_batch_size=0,
            armnn_cpu=armnn_on,
            xnnpack=xnnpack_on,
        )
        for armnn_on, xnnpack_on in list(product([True, False], repeat=2))
    ],
)
@pytest.mark.parametrize(
    "num_threads",
    [int(pow(2, i)) for i in range(int(log2(multiprocessing.cpu_count())) + 1)],
)
def test_single_model(
    tritonserver_client,
    request,
    model_config,
    num_threads,
):
    if tritonserver_client.module != httpclient:
        pytest.skip("Thread count test only runs for http client")

    triton_process = psutil.Process(tritonserver_client.triton_pid)

    base_threads = triton_process.num_threads()

    model_config.armnn_cpu_parameters["num_threads"] = num_threads
    model_config.xnnpack_parameters["num_threads"] = num_threads
    model_config.tflite_num_threads = num_threads

    load_model(
        tritonserver_client.client,
        model_config,
    )

    assert tritonserver_client.client.is_model_ready(model_config.name)

    request_inputs = get_random_triton_inputs(
        model_config.inputs,
        None if model_config.max_batch_size == 0 else model_config.max_batch_size,
        tritonserver_client.module,
    )

    request_outputs = []
    for output in model_config.outputs:
        request_outputs.append(
            tritonserver_client.module.InferRequestedOutput(output.name)
        )

    results = tritonserver_client.client.infer(
        model_config.name,
        request_inputs,
        model_version="1",
        outputs=request_outputs,
    )

    post_inf_threads = triton_process.num_threads()

    assert (post_inf_threads - base_threads) == num_threads
