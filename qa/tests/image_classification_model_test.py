# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest

import numpy as np

from itertools import product, combinations

from helpers.triton_model_config import Model, TFLiteTritonModel
from helpers.image_helper import extract_photo, image_set_generator


def classification_net(
    inference_client,
    client_type,
    test_image_set,
    model_config,
    scaling,
    batching,
):
    assert inference_client.is_model_ready(model_config.name)

    image_input = model_config.inputs[0]

    if (
        len(test_image_set) > model_config.max_batch_size
        and model_config.max_batch_size != 0
    ):
        pytest.xfail("Test image set larger than max batch size for this test")

    if batching:
        image_data = []

        for idx in range(len(test_image_set)):
            image_data.append(
                np.squeeze(
                    extract_photo(
                        test_image_set[idx][0],
                        image_input.dims[1],
                        image_input.dims[1],
                        scaling,
                    ),
                    axis=0,
                )
            )

        batched_image_data = np.stack(image_data, axis=0)

        request_input = client_type.InferInput(
            image_input.name,
            [batched_image_data.shape[0], image_input.dims[1], image_input.dims[1], 3],
            "FP32",
        )

        request_input.set_data_from_numpy(batched_image_data)

    else:
        request_input = client_type.InferInput(
            image_input.name,
            [1, image_input.dims[1], image_input.dims[1], 3],
            "FP32",
        )

        request_input.set_data_from_numpy(
            extract_photo(
                test_image_set[0][0],
                image_input.dims[1],
                image_input.dims[1],
                scaling,
            )
        )

    classification_output = model_config.outputs[0]
    request_output = client_type.InferRequestedOutput(
        classification_output.name, class_count=3
    )

    results = inference_client.infer(
        model_config.name,
        (request_input,),
        model_version="1",
        outputs=(request_output,),
    )

    output_array = results.as_numpy(classification_output.name)

    cls_ids = []
    # If we are working on batched inference validate size output array
    if batching:
        assert len(output_array) == len(test_image_set)
        i = 0
        for output in output_array:
            for result in output:
                cls = "".join(chr(x) for x in result).split(":")
                cls_ids.append(cls[1])

            expected = test_image_set[i][1]
            i += 1
            assert expected in cls_ids
    else:
        for result in output_array:
            cls = "".join(chr(x) for x in result).split(":")
            cls_ids.append(cls[1])

        expected = test_image_set[[0][0]][1]
        assert expected in cls_ids


@pytest.mark.parametrize(
    "model_config",
    [
        TFLiteTritonModel(
            "mobilenet_v3_large_100_224",
            [Model.TensorIO("serving_default_inputs:0", "TYPE_FP32", [224, 224, 3])],
            [
                Model.TensorIO(
                    "StatefulPartitionedCall:0",
                    "TYPE_FP32",
                    [1001],
                    label_filename="labels.txt",
                )
            ],
            max_batch_size=max_batch_size,
            armnn_cpu=armnn_on,
            xnnpack=xnnpack_on,
        )
        for (armnn_on, xnnpack_on, max_batch_size) in list(
            product([True, False], [True, False], range(1, 4))
        )
        if not (xnnpack_on and armnn_on)
    ],
)
@pytest.mark.parametrize("client_type", ["http", "grpc"])
@pytest.mark.parametrize(
    "test_image_set",
    image_set_generator(
        [
            ("images/mug.jpg", "505"),
            ("images/shark.jpg", "3"),
            ("images/ostrich.jpg", "10"),
            ("images/dog.jpg", "209"),
            ("images/goldfish.jpg", "2"),
        ],
        5,
    ),
)
def test_mobilenetv3(
    tritonserver,
    load_model_with_config,
    inference_client,
    client_type,
    test_image_set,
    model_config,
):
    classification_net(
        inference_client,
        client_type,
        test_image_set,
        model_config,
        "mobilenetv3",
        True,
    )


@pytest.mark.parametrize(
    "model_config",
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
            armnn_cpu=armnn_on,
            xnnpack=xnnpack_on,
        )
        for (armnn_on, xnnpack_on) in list(product([True, False], [True, False]))
        if not (xnnpack_on and armnn_on)
    ],
)
@pytest.mark.parametrize("client_type", ["http", "grpc"])
@pytest.mark.parametrize(
    "test_image_set",
    combinations(
        [
            ("images/mug.jpg", "505"),
            ("images/shark.jpg", "3"),
            ("images/ostrich.jpg", "10"),
            ("images/dog.jpg", "209"),
            ("images/goldfish.jpg", "2"),
        ],
        r=1,
    ),
)
def test_mobilenetv1(
    tritonserver,
    load_model_with_config,
    inference_client,
    client_type,
    test_image_set,
    model_config,
):
    classification_net(
        inference_client,
        client_type,
        test_image_set,
        model_config,
        "mobilenet",
        False,
    )


@pytest.mark.parametrize(
    "model_config",
    [
        TFLiteTritonModel(
            "mobilenet_v2_1.0_224",
            [Model.TensorIO("input", "TYPE_FP32", [1, 224, 224, 3])],
            [
                Model.TensorIO(
                    "MobilenetV2/Predictions/Reshape_1",
                    "TYPE_FP32",
                    [1001],
                    label_filename="labels.txt",
                )
            ],
            armnn_cpu=armnn_on,
            xnnpack=xnnpack_on,
        )
        for (armnn_on, xnnpack_on) in list(product([True, False], [True, False]))
        if not (xnnpack_on and armnn_on)
    ],
)
@pytest.mark.parametrize("client_type", ["http", "grpc"])
@pytest.mark.parametrize(
    "test_image_set",
    combinations(
        [
            ("images/mug.jpg", "505"),
            ("images/shark.jpg", "3"),
            ("images/ostrich.jpg", "10"),
            ("images/dog.jpg", "209"),
            ("images/goldfish.jpg", "2"),
        ],
        r=1,
    ),
)
def test_mobilenetv2(
    tritonserver,
    load_model_with_config,
    inference_client,
    client_type,
    test_image_set,
    model_config,
):
    classification_net(
        inference_client,
        client_type,
        test_image_set,
        model_config,
        "mobilenet",
        False,
    )


@pytest.mark.parametrize(
    "model_config",
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
            armnn_cpu=armnn_on,
            xnnpack=xnnpack_on,
        )
        for (armnn_on, xnnpack_on) in list(product([True, False], [True, False]))
        if not (xnnpack_on and armnn_on)
    ],
)
@pytest.mark.parametrize("client_type", ["http", "grpc"])
@pytest.mark.parametrize(
    "test_image_set",
    combinations(
        [
            ("images/mug.jpg", "505"),
            ("images/shark.jpg", "3"),
            ("images/ostrich.jpg", "10"),
            ("images/dog.jpg", "209"),
            ("images/goldfish.jpg", "2"),
        ],
        r=1,
    ),
)
def test_inceptionv3(
    tritonserver,
    load_model_with_config,
    inference_client,
    client_type,
    test_image_set,
    model_config,
):
    classification_net(
        inference_client,
        client_type,
        test_image_set,
        model_config,
        "inception",
        False,
    )


@pytest.mark.parametrize(
    "model_config",
    [
        TFLiteTritonModel(
            "resnet_v2_101_fp32",
            [Model.TensorIO("input", "TYPE_FP32", [1, 299, 299, 3])],
            [
                Model.TensorIO(
                    "output", "TYPE_FP32", [1001], label_filename="labels.txt"
                )
            ],
            armnn_cpu=armnn_on,
            xnnpack=xnnpack_on,
        )
        for (armnn_on, xnnpack_on) in list(product([True, False], [True, False]))
        if not (xnnpack_on and armnn_on)
    ],
)
@pytest.mark.parametrize("client_type", ["http", "grpc"])
@pytest.mark.parametrize(
    "test_image_set",
    combinations(
        [
            ("images/mug.jpg", "505"),
            ("images/shark.jpg", "3"),
            ("images/ostrich.jpg", "10"),
            ("images/dog.jpg", "209"),
            ("images/goldfish.jpg", "2"),
        ],
        r=1,
    ),
)
def test_resnetv2_101(
    tritonserver,
    load_model_with_config,
    inference_client,
    client_type,
    test_image_set,
    model_config,
):
    classification_net(
        inference_client,
        client_type,
        test_image_set,
        model_config,
        "resnetv2",
        False,
    )
