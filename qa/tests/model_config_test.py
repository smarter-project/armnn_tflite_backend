# Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest

from tritonclient.utils import InferenceServerException

from helpers.triton_model_config import Model, TFLiteTritonModel
from helpers.helper_functions import (
    load_model,
)


@pytest.mark.parametrize(
    "model_config",
    [
        TFLiteTritonModel(
            "add",
            [Model.TensorIO("X_input_xxx", "TYPE_FP32", [1])],
            [Model.TensorIO("ADD_TOP", "TYPE_FP32", [1])],
        ),
    ],
)
def test_incorrect_input_name(
    tritonserver_client,
    model_config: TFLiteTritonModel,
):
    with pytest.raises(InferenceServerException):
        load_model(
            tritonserver_client.client,
            model_config,
        )


@pytest.mark.parametrize(
    "model_config",
    [
        TFLiteTritonModel(
            "add",
            [Model.TensorIO("X_input", "TYPE_FP32", [1])],
            [Model.TensorIO("ADD_TOP_xxx", "TYPE_FP32", [1])],
        ),
    ],
)
def test_incorrect_output_name(
    tritonserver_client,
    model_config: TFLiteTritonModel,
):
    with pytest.raises(InferenceServerException):
        load_model(
            tritonserver_client.client,
            model_config,
        )


@pytest.mark.parametrize(
    "model_config",
    [
        TFLiteTritonModel(
            "add",
            [
                Model.TensorIO("X_input", "TYPE_FP32", [1]),
                Model.TensorIO("extra_input", "TYPE_FP32", [1]),
            ],
            [Model.TensorIO("ADD_TOP", "TYPE_FP32", [1])],
        ),
    ],
)
def test_extra_input(
    tritonserver_client,
    model_config: TFLiteTritonModel,
):
    with pytest.raises(InferenceServerException):
        load_model(
            tritonserver_client.client,
            model_config,
        )


@pytest.mark.parametrize(
    "model_config",
    [
        TFLiteTritonModel(
            "add",
            [Model.TensorIO("X_input", "TYPE_FP32", [1])],
            [
                Model.TensorIO("ADD_TOP", "TYPE_FP32", [1]),
                Model.TensorIO("extra_output", "TYPE_FP32", [1]),
            ],
        ),
    ],
)
def test_extra_output(
    tritonserver_client,
    model_config: TFLiteTritonModel,
):
    with pytest.raises(InferenceServerException):
        load_model(
            tritonserver_client.client,
            model_config,
        )


@pytest.mark.parametrize(
    "model_config",
    [
        TFLiteTritonModel(
            "add",
            [Model.TensorIO("X_input", "TYPE_INT32", [1])],
            [Model.TensorIO("ADD_TOP", "TYPE_FP32", [1])],
        ),
        TFLiteTritonModel(
            "add",
            [Model.TensorIO("X_input", "TYPE_FP32", [1])],
            [Model.TensorIO("ADD_TOP", "TYPE_INT32", [1])],
        ),
    ],
)
def test_incorrect_datatypes(
    tritonserver_client,
    model_config: TFLiteTritonModel,
):
    with pytest.raises(InferenceServerException):
        load_model(
            tritonserver_client.client,
            model_config,
        )


@pytest.mark.parametrize(
    "model_config",
    [
        TFLiteTritonModel(
            "add",
            [Model.TensorIO("X_input", "TYPE_FP32", [5])],
            [Model.TensorIO("ADD_TOP", "TYPE_FP32", [5])],
        ),
    ],
)
def test_incorrect_shapes(
    tritonserver_client,
    model_config: TFLiteTritonModel,
):
    with pytest.raises(InferenceServerException):
        load_model(
            tritonserver_client.client,
            model_config,
        )


@pytest.mark.parametrize(
    "model_config",
    [
        TFLiteTritonModel(
            "add",
            [Model.TensorIO("X_input", "TYPE_FP32", [1])],
            [Model.TensorIO("ADD_TOP", "TYPE_FP32", [1])],
            tflite_num_threads=-2,
        ),
        TFLiteTritonModel(
            "add",
            [Model.TensorIO("X_input", "TYPE_FP32", [1])],
            [Model.TensorIO("ADD_TOP", "TYPE_FP32", [1])],
            armnn_cpu=True,
            armnn_cpu_parameters={"invalid_name": 3},
        ),
        TFLiteTritonModel(
            "add",
            [Model.TensorIO("X_input", "TYPE_FP32", [1])],
            [Model.TensorIO("ADD_TOP", "TYPE_FP32", [1])],
            armnn_cpu=True,
            armnn_cpu_parameters={"num_threads": -2},
        ),
        TFLiteTritonModel(
            "add",
            [Model.TensorIO("X_input", "TYPE_FP32", [1])],
            [Model.TensorIO("ADD_TOP", "TYPE_FP32", [1])],
            armnn_cpu=True,
            armnn_cpu_parameters={"reduce_fp32_to_fp16": -2},
        ),
        TFLiteTritonModel(
            "add",
            [Model.TensorIO("X_input", "TYPE_FP32", [1])],
            [Model.TensorIO("ADD_TOP", "TYPE_FP32", [1])],
            armnn_cpu=True,
            armnn_cpu_parameters={"reduce_fp32_to_bf16": -2},
        ),
        TFLiteTritonModel(
            "add",
            [Model.TensorIO("X_input", "TYPE_FP32", [1])],
            [Model.TensorIO("ADD_TOP", "TYPE_FP32", [1])],
            armnn_cpu=True,
            armnn_cpu_parameters={"fast_math_enabled": -2},
        ),
        TFLiteTritonModel(
            "add",
            [Model.TensorIO("X_input", "TYPE_FP32", [1])],
            [Model.TensorIO("ADD_TOP", "TYPE_FP32", [1])],
            xnnpack=True,
            xnnpack_parameters={"invalid_name": 3},
        ),
    ],
)
def test_invalid_runtime_params(
    tritonserver_client,
    model_config: TFLiteTritonModel,
):
    with pytest.raises(InferenceServerException):
        load_model(
            tritonserver_client.client,
            model_config,
        )
