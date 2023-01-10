# Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import os

from helpers.triton_model_config import Model, TFLiteTritonModel
from helpers.helper_functions import load_model, get_random_triton_inputs


def is_alive(pid: int):
    try:
        os.kill(pid, 0)
    except OSError:
        return True
    else:
        return False


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
                        label_filename="labels.txt",
                    )
                ],
                max_batch_size=0,
                armnn_cpu=True,
            ),
            TFLiteTritonModel(
                "add",
                [Model.TensorIO("X_input", "TYPE_FP32", [1])],
                [Model.TensorIO("ADD_TOP", "TYPE_FP32", [1])],
                armnn_cpu=True,
            ),
        ]
    ],
)
def test_differing_thread_counts(
    tritonserver_client,
    model_configs,
):
    if os.uname().machine != "aarch64":
        pytest.skip("ArmNN acceleration only supported on aarch64")

    model_configs[0].set_thread_count(2)
    model_configs[1].set_thread_count(1)

    for model_config in model_configs:
        load_model(
            tritonserver_client.client,
            model_config,
        )

        assert tritonserver_client.client.is_server_ready()

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
    assert is_alive(tritonserver_client.triton_pid)
