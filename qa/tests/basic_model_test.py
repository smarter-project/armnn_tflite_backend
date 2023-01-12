# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest

import numpy as np

from tritonclient.utils import triton_to_np_dtype
from typing import List

from itertools import product

from helpers.triton_model_config import Model, TFLiteTritonModel
from helpers.helper_functions import is_server_ready, send_inference_request


def basic_test(
    model_config: TFLiteTritonModel,
    tritonserver_client,
    input_value: List,
    expected: List,
):
    assert is_server_ready(tritonserver_client.client)

    results = send_inference_request(
        tritonserver_client.client,
        tritonserver_client.module,
        model_config,
        input_tensors={model_config.inputs[0].name: input_value},
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
@pytest.mark.parametrize(
    "input_value,expected",
    [
        ([5], [7]),
        ([0], [2]),
        ([-1], [1]),
    ],
)
def test_add(
    load_model_with_config,
    tritonserver_client,
    input_value,
    expected,
    model_config,
):
    basic_test(model_config, tritonserver_client, input_value, expected)


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
    load_model_with_config,
    tritonserver_client,
    input_value,
    expected,
    model_config,
):
    basic_test(model_config, tritonserver_client, input_value, expected)
