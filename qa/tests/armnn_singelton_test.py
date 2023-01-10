# Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import os
import psutil
from typing import List

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
                armnn_cpu_parameters={"num_threads": 3},
            ),
            TFLiteTritonModel(
                "add",
                [Model.TensorIO("X_input", "TYPE_FP32", [1])],
                [Model.TensorIO("ADD_TOP", "TYPE_FP32", [1])],
                armnn_cpu=True,
                armnn_cpu_parameters={"num_threads": 1},
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
        ],
    ],
)
def test_differing_thread_counts(
    tritonserver_client,
    model_configs: List[TFLiteTritonModel],
):
    if os.uname().machine != "aarch64":
        pytest.skip("ArmNN acceleration only supported on aarch64")

    triton_process = psutil.Process(tritonserver_client.triton_pid)

    for model_config in model_configs:
        load_model(
            tritonserver_client.client,
            model_config,
        )

        assert tritonserver_client.client.is_server_ready()

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
    assert triton_process.is_running()
    assert tritonserver_client.client.is_server_ready()

    # Now unload models and make sure everything still behaves
    for model_config in model_configs:
        unload_model(tritonserver_client.client, model_config.name)

    assert triton_process.is_running()
    assert tritonserver_client.client.is_server_ready()
