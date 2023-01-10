# Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
from typing import List
from time import sleep

from helpers.triton_model_config import Model, TFLiteTritonModel
from helpers.helper_functions import load_model, get_random_triton_inputs, unload_model


@pytest.mark.parametrize(
    "model_configs",
    [
        [
            TFLiteTritonModel(
                "mobilenet_v1_1.0_224",
                [Model.TensorIO("input", "TYPE_FP32", [1, 224, 224, 3])],
                [
                    Model.TensorIO(
                        "MobilenetV1/Predictions/Reshape_1",
                        "TYPE_FP32",
                        [1001],
                    )
                ],
                max_batch_size=0,
                armnn_cpu=True,
                armnn_cpu_parameters={"num_threads": 1},
            ),
            TFLiteTritonModel(
                "add",
                [Model.TensorIO("X_input", "TYPE_FP32", [1])],
                [Model.TensorIO("ADD_TOP", "TYPE_FP32", [1])],
                armnn_cpu=True,
                armnn_cpu_parameters={"num_threads": 3},
            ),
        ],
        [
            TFLiteTritonModel(
                "mobilenet_v1_1.0_224",
                [Model.TensorIO("input", "TYPE_FP32", [1, 224, 224, 3])],
                [
                    Model.TensorIO(
                        "MobilenetV1/Predictions/Reshape_1",
                        "TYPE_FP32",
                        [1001],
                    )
                ],
                max_batch_size=0,
                armnn_cpu=True,
                armnn_cpu_parameters={"num_threads": 2},
            ),
            TFLiteTritonModel(
                "mobilenet_v2_1.0_224",
                [Model.TensorIO("input", "TYPE_FP32", [1, 224, 224, 3])],
                [
                    Model.TensorIO(
                        "MobilenetV2/Predictions/Reshape_1",
                        "TYPE_FP32",
                        [1001],
                    )
                ],
                armnn_cpu=True,
                armnn_cpu_parameters={"num_threads": 3},
            ),
        ],
        [
            TFLiteTritonModel(
                "inceptionv3",
                [Model.TensorIO("input", "TYPE_FP32", [1, 299, 299, 3])],
                [
                    Model.TensorIO(
                        "InceptionV3/Predictions/Reshape_1",
                        "TYPE_FP32",
                        [1001],
                        label_filename="labels.txt",
                    )
                ],
                armnn_cpu=True,
                armnn_cpu_parameters={"num_threads": 2},
            ),
            TFLiteTritonModel(
                "mobilenet_v2_1.0_224",
                [Model.TensorIO("input", "TYPE_FP32", [1, 224, 224, 3])],
                [
                    Model.TensorIO(
                        "MobilenetV2/Predictions/Reshape_1",
                        "TYPE_FP32",
                        [1001],
                    )
                ],
                armnn_cpu=True,
                armnn_cpu_parameters={"num_threads": 4},
            ),
            TFLiteTritonModel(
                "add",
                [Model.TensorIO("X_input", "TYPE_FP32", [1])],
                [Model.TensorIO("ADD_TOP", "TYPE_FP32", [1])],
                armnn_cpu=True,
                armnn_cpu_parameters={"num_threads": 5},
            ),
        ],
    ],
)
def test_increasing_thread_counts_armnn(
    tritonserver_client,
    model_configs: List[TFLiteTritonModel],
):
    # Load each model in sequence
    for model_config in model_configs:
        load_model(
            tritonserver_client.client,
            model_config,
        )

    assert tritonserver_client.client.is_server_ready()

    # Request inference from each model in sequence
    for model_config in model_configs:
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

    # After both models have inference requested, assert triton has not segfaulted
    assert tritonserver_client.client.is_server_ready()

    # Now unload models and make sure everything still behaves
    for model_config in model_configs:
        unload_model(tritonserver_client.client, model_config.name)

    retries = 5
    while retries > 0:
        if tritonserver_client.client.is_server_ready():
            break
        sleep(1)
        retries -= 1

    assert retries != 0, "Triton never became ready after unload"
