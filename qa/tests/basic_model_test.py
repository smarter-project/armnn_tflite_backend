# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest

import numpy as np

import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient

from itertools import product

from helpers.triton_model_config import Model, TFLiteTritonModel


def basic_test(model_config, inference_client, client_type, input_value, expected):
    assert inference_client.is_model_ready(model_config.name)
    
    request_inputs = []
    for input in model_config.inputs:
        request_input = client_type.InferInput(
            input.name, input.dims, input.datatype.split("TYPE_", 1)[1]
        )
        request_input.set_data_from_numpy(
            np.array(input_value, dtype=np.float32).reshape(input.dims)
        )
        request_inputs.append(request_input)

    request_outputs = []
    for output in model_config.outputs:
        request_outputs.append(client_type.InferRequestedOutput(output.name))

    results = inference_client.infer(
        model_config.name,
        request_inputs,
        model_version="1",
        outputs=request_outputs,
    )

    assert np.array_equal(
        results.as_numpy(model_config.outputs[0].name),
        np.array(expected, dtype=np.float32).reshape(model_config.outputs[0].dims),
    )


@pytest.mark.parametrize(
    "model_config",
    [
        TFLiteTritonModel(
            "add",
            [Model.TensorIO("X_input", "TYPE_FP32", [1])],
            [Model.TensorIO("ADD_TOP", "TYPE_FP32", [1])],
            armnn_cpu=armnn_on,
            xnnpack=xnnpack_on,
        )
        for armnn_on, xnnpack_on in list(product([True, False], repeat=2))
    ],
)
@pytest.mark.parametrize("client_type", [httpclient, grpcclient])
@pytest.mark.parametrize(
    "input_value,expected",
    [
        ([5], [7]),
        ([0], [2]),
        ([-1], [1]),
    ],
)
def test_add(
    tritonserver,
    load_model_with_config,
    inference_client,
    client_type,
    input_value,
    expected,
    model_config,
):
    basic_test(model_config, inference_client, client_type, input_value, expected)


@pytest.mark.parametrize(
    "model_config",
    [
        TFLiteTritonModel(
            "conv2d",
            [Model.TensorIO("input", "TYPE_FP32", [1, 5, 5, 1])],
            [Model.TensorIO("output", "TYPE_FP32", [1, 3, 3, 1])],
            armnn_cpu=armnn_on,
            xnnpack=xnnpack_on,
        )
        for armnn_on, xnnpack_on in list(product([True, False], repeat=2))
    ],
)
@pytest.mark.parametrize("client_type", [httpclient, grpcclient])
@pytest.mark.parametrize(
    "input_value,expected",
    [
        (
            [
                1,
                5,
                2,
                3,
                5,
                8,
                7,
                3,
                6,
                3,
                3,
                3,
                9,
                1,
                9,
                4,
                1,
                8,
                1,
                3,
                6,
                8,
                1,
                9,
                2,
            ],
            [28, 38, 29, 96, 104, 53, 31, 55, 24],
        ),
    ],
)
def test_conv2d(
    tritonserver,
    load_model_with_config,
    inference_client,
    client_type,
    input_value,
    expected,
    model_config,
):
    basic_test(model_config, inference_client, client_type, input_value, expected)
