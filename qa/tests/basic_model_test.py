# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest

import numpy as np
from typing import Union

import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
from tritonclient.utils import triton_to_np_dtype
from typing import List

from itertools import product

from helpers.triton_model_config import Model, TFLiteTritonModel


def basic_test(
    model_config: TFLiteTritonModel,
    inference_client: Union[
        httpclient.InferenceServerClient, grpcclient.InferenceServerClient
    ],
    client_type: str,
    input_value: List,
    expected: List,
):
    assert inference_client.is_model_ready(model_config.name)

    if client_type == "http":
        client_module = httpclient
    else:
        client_module = grpcclient

    request_inputs = []
    for input in model_config.inputs:
        input_dtype_name = input.datatype.split("TYPE_", 1)[1]
        request_input = client_module.InferInput(
            input.name, input.dims, input_dtype_name
        )
        request_input.set_data_from_numpy(
            np.array(input_value, dtype=triton_to_np_dtype(input_dtype_name)).reshape(
                input.dims
            )
        )
        request_inputs.append(request_input)

    request_outputs = []
    for output in model_config.outputs:
        request_outputs.append(client_module.InferRequestedOutput(output.name))

    results = inference_client.infer(
        model_config.name,
        request_inputs,
        model_version="1",
        outputs=request_outputs,
    )

    for output in model_config.outputs:
        output_dtype_name = output.datatype.split("TYPE_", 1)[1]
        assert np.array_equal(
            results.as_numpy(output.name),
            np.array(expected, dtype=triton_to_np_dtype(output_dtype_name)).reshape(
                output.dims
            ),
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
@pytest.mark.parametrize("client_type", ["http", "grpc"])
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
@pytest.mark.parametrize("client_type", ["http", "grpc"])
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
