# Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest

import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient

from itertools import product

from helpers.triton_model_config import Model, TFLiteTritonModel, load_model


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
@pytest.mark.parametrize("client_type", [httpclient, grpcclient])
def test_inceptionv3_dynamic(tritonserver, inference_client, model_config, request):
    if model_config.xnnpack or model_config.armnn_cpu:
        pytest.xfail(
            "XNNPACK/ArmNN not supported on dynamic sized non-batch dimensions for input tensor shapes"
        )
    load_model(
        inference_client, model_config, request.config.getoption("model_repo_path")
    )
    assert inference_client.is_model_ready(model_config.name)
