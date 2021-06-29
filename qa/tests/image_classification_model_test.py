# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import numpy as np

import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient

from PIL import Image

from itertools import product

from helpers.triton_model_config import Model, TFLiteTritonModel


def extract_photo(filename, width, height, scaling=None):
    img = Image.open(filename)
    resized_img = img.resize((width, height), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    expanded = np.expand_dims(resized, axis=0)

    typed = expanded.astype(np.float32)

    if scaling:
        if scaling.lower() == "inception":
            scaled = (typed / 127.5) - 1.0
        elif scaling.lower() == "resnetv2":
            scaled = 2 * (typed / 255.0) - 1.0
        elif scaling.lower() == "mobilenet":
            scaled = (typed / 255.0) - 1.0
        else:
            scaled = typed
    else:
        scaled = typed

    return scaled


def classification_net(
    inference_client,
    client_type,
    test_image,
    expected,
    model_config,
    scaling,
):
    image_input = model_config.inputs[0]
    request_input = client_type.InferInput(
        image_input.name, [1, image_input.dims[1], image_input.dims[1], 3], "FP32"
    )
    request_input.set_data_from_numpy(
        extract_photo(test_image, image_input.dims[1], image_input.dims[1], scaling)
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
    for result in output_array:
        cls = "".join(chr(x) for x in result).split(":")
        cls_ids.append(cls[1])

    assert expected in cls_ids


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
        for armnn_on, xnnpack_on in list(product([True, False], repeat=2))
    ],
)
@pytest.mark.parametrize("client_type", [httpclient, grpcclient])
@pytest.mark.parametrize(
    "test_image,expected",
    [
        ("images/mug.jpg", "505"),
        ("images/shark.jpg", "3"),
        ("images/ostrich.jpg", "10"),
        ("images/dog.jpg", "209"),
        ("images/goldfish.jpg", "2"),
    ],
)
def test_mobilenetv1(
    generate_model_config,
    inference_client,
    client_type,
    test_image,
    expected,
    model_config,
):
    classification_net(
        inference_client, client_type, test_image, expected, model_config, "mobilenet"
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
        for armnn_on, xnnpack_on in list(product([True, False], repeat=2))
    ],
)
@pytest.mark.parametrize("client_type", [httpclient, grpcclient])
@pytest.mark.parametrize(
    "test_image,expected",
    [
        ("images/mug.jpg", "505"),
        ("images/shark.jpg", "3"),
        ("images/ostrich.jpg", "10"),
        ("images/dog.jpg", "209"),
        ("images/goldfish.jpg", "2"),
    ],
)
def test_mobilenetv2(
    generate_model_config,
    inference_client,
    client_type,
    test_image,
    expected,
    model_config,
):
    classification_net(
        inference_client, client_type, test_image, expected, model_config, "mobilenet"
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
        for armnn_on, xnnpack_on in list(product([True, False], repeat=2))
    ],
)
@pytest.mark.parametrize("client_type", [httpclient, grpcclient])
@pytest.mark.parametrize(
    "test_image,expected",
    [
        ("images/mug.jpg", "505"),
        ("images/shark.jpg", "3"),
        ("images/ostrich.jpg", "10"),
        ("images/dog.jpg", "209"),
        ("images/goldfish.jpg", "2"),
    ],
)
def test_inceptionv3(
    generate_model_config,
    inference_client,
    client_type,
    test_image,
    expected,
    model_config,
):
    classification_net(
        inference_client, client_type, test_image, expected, model_config, "inception"
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
        for armnn_on, xnnpack_on in list(product([True, False], repeat=2))
    ],
)
@pytest.mark.parametrize("client_type", [httpclient, grpcclient])
@pytest.mark.parametrize(
    "test_image,expected",
    [
        ("images/mug.jpg", "505"),
        ("images/shark.jpg", "3"),
        ("images/ostrich.jpg", "10"),
        ("images/dog.jpg", "209"),
        ("images/goldfish.jpg", "2"),
    ],
)
def test_resnetv2_101(
    generate_model_config,
    inference_client,
    client_type,
    test_image,
    expected,
    model_config,
):
    classification_net(
        inference_client, client_type, test_image, expected, model_config, "resnetv2"
    )
