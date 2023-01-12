# Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
from typing import List
from itertools import product


from helpers.triton_model_config import Model, TFLiteTritonModel
from helpers.helper_functions import (
    load_model,
    send_inference_request,
    unload_model,
    is_server_ready,
)


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
                        [1, 1001],
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
                        [1, 1001],
                    )
                ],
                max_batch_size=0,
                tflite_num_threads=1,
            ),
            TFLiteTritonModel(
                "add",
                [Model.TensorIO("X_input", "TYPE_FP32", [1])],
                [Model.TensorIO("ADD_TOP", "TYPE_FP32", [1])],
                tflite_num_threads=3,
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
                        [1, 1001],
                    )
                ],
                max_batch_size=0,
                tflite_num_threads=3,
            ),
            TFLiteTritonModel(
                "add",
                [Model.TensorIO("X_input", "TYPE_FP32", [1])],
                [Model.TensorIO("ADD_TOP", "TYPE_FP32", [1])],
                tflite_num_threads=1,
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
                        [1, 1001],
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
                        [1, 1001],
                    )
                ],
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
                        [1, 1001],
                    )
                ],
                max_batch_size=0,
                xnnpack=True,
                xnnpack_parameters={"num_threads": 2},
            ),
            TFLiteTritonModel(
                "mobilenet_v2_1.0_224",
                [Model.TensorIO("input", "TYPE_FP32", [1, 224, 224, 3])],
                [
                    Model.TensorIO(
                        "MobilenetV2/Predictions/Reshape_1",
                        "TYPE_FP32",
                        [1, 1001],
                    )
                ],
                xnnpack=True,
                xnnpack_parameters={"num_threads": 3},
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
                        [1, 1001],
                    )
                ],
                max_batch_size=0,
                xnnpack=True,
                xnnpack_parameters={"num_threads": 3},
            ),
            TFLiteTritonModel(
                "mobilenet_v2_1.0_224",
                [Model.TensorIO("input", "TYPE_FP32", [1, 224, 224, 3])],
                [
                    Model.TensorIO(
                        "MobilenetV2/Predictions/Reshape_1",
                        "TYPE_FP32",
                        [1, 1001],
                    )
                ],
                xnnpack=True,
                xnnpack_parameters={"num_threads": 2},
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
                        [1, 1001],
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
                        [1, 1001],
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
        [
            TFLiteTritonModel(
                "inceptionv3",
                [Model.TensorIO("input", "TYPE_FP32", [1, 299, 299, 3])],
                [
                    Model.TensorIO(
                        "InceptionV3/Predictions/Reshape_1",
                        "TYPE_FP32",
                        [1, 1001],
                        label_filename="labels.txt",
                    )
                ],
                armnn_cpu=True,
                armnn_cpu_parameters={
                    "num_threads": 2,
                    "fast_math_enabled": "on",
                    "reduce_fp32_to_fp16": "off",
                },
            ),
            TFLiteTritonModel(
                "mobilenet_v2_1.0_224",
                [Model.TensorIO("input", "TYPE_FP32", [1, 224, 224, 3])],
                [
                    Model.TensorIO(
                        "MobilenetV2/Predictions/Reshape_1",
                        "TYPE_FP32",
                        [1, 1001],
                    )
                ],
                armnn_cpu=True,
                armnn_cpu_parameters={"num_threads": 4, "reduce_fp32_to_fp16": "on"},
            ),
            TFLiteTritonModel(
                "add",
                [Model.TensorIO("X_input", "TYPE_FP32", [1])],
                [Model.TensorIO("ADD_TOP", "TYPE_FP32", [1])],
                xnnpack=True,
                xnnpack_parameters={"num_threads": 5},
            ),
        ],
        [
            TFLiteTritonModel(
                "inceptionv3_dynamic",
                [Model.TensorIO("serving_default_inputs:0", "TYPE_FP32", [-1, -1, 3])],
                [
                    Model.TensorIO(
                        "StatefulPartitionedCall:0",
                        "TYPE_FP32",
                        [1001],
                    )
                ],
                armnn_cpu=armnn_on,
                xnnpack=xnnpack_on,
            )
            for (armnn_on, xnnpack_on) in list(product([True, False], [True, False]))
            if not (xnnpack_on and armnn_on)
        ],
    ],
)
def test_differing_thread_counts(
    tritonserver_client,
    model_configs: List[TFLiteTritonModel],
):
    # Load each model in sequence
    for model_config in model_configs:
        load_model(
            tritonserver_client.client,
            model_config,
        )

    assert is_server_ready(tritonserver_client.client)

    # Request inference from each model in sequence
    for model_config in model_configs:
        send_inference_request(
            tritonserver_client.client, tritonserver_client.module, model_config
        )

    # After both models have inference requested, assert triton has not segfaulted
    assert is_server_ready(tritonserver_client.client)

    # Now unload models and make sure everything still behaves
    for model_config in model_configs:
        unload_model(tritonserver_client.client, model_config.name)

    assert is_server_ready(
        tritonserver_client.client
    ), "Triton never became ready after unload"
