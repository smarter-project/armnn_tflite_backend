# Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest

from itertools import product

from helpers.triton_model_config import Model, TFLiteTritonModel
from helpers.helper_functions import load_model


@pytest.mark.parametrize(
    "model_config",
    [
        TFLiteTritonModel(
            "inceptionv3_dynamic",
            [Model.TensorIO("serving_default_inputs:0", "TYPE_FP32", [299, 299, 3])],
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
)
def test_inceptionv3_dynamic(tritonserver_client, model_config):
    if model_config.xnnpack or model_config.armnn_cpu:
        pytest.xfail(
            "XNNPACK/ArmNN not supported on dynamic sized non-batch dimensions for input tensor shapes"
        )
    load_model(
        tritonserver_client.client,
        model_config,
    )
    assert tritonserver_client.client.is_model_ready(model_config.name)
