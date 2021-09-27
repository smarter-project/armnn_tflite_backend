# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest

from jinja2 import Environment, Template

import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient

from time import sleep


@pytest.fixture
def generate_model_config(inference_client, model_config, model_repo_path):
    inference_client.unload_model(model_config.name)

    with open("config-template.pbtxt") as file_:
        template = Template(
            file_.read(),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
    output_config = template.render(model=model_config)
    with open(
        model_repo_path + "/" + model_config.name + "/config.pbtxt", "w+"
    ) as output_file_:
        output_file_.write(output_config)

    inference_client.load_model(model_config.name)

    while not inference_client.is_model_ready(model_config.name):
        sleep(1)

    yield

    inference_client.unload_model(model_config.name)


@pytest.fixture
def inference_client(client_type, host):
    if client_type == httpclient:
        client = httpclient.InferenceServerClient(url=str(host) + ":8000")
    else:
        client = grpcclient.InferenceServerClient(url=str(host) + ":8001")

    return client


def pytest_addoption(parser):
    """
    Adds the program option 'url' to pytest
    """
    parser.addoption(
        "--host",
        action="store",
        default="localhost",
        required=False,
        help="Inference server URL. Default is localhost:8000.",
    )
    parser.addoption(
        "--model-repo-path",
        action="store",
        default="test_model_repo",
        required=False,
        help="Path to top level of triton model repository",
    )
    parser.addoption(
        "--backend-directory",
        action="store",
        default="/tmp/citritonbuild/opt/tritonserver/backends",
        required=False,
        help="Path to triton backends",
    )


def pytest_generate_tests(metafunc):
    """
    Makes the program option 'host' available to all tests as a function fixture
    """
    if "host" in metafunc.fixturenames:
        metafunc.parametrize("host", [metafunc.config.getoption("host")])
    if "model_repo_path" in metafunc.fixturenames:
        metafunc.parametrize(
            "model_repo_path", [metafunc.config.getoption("model_repo_path")]
        )
    if "backend_directory" in metafunc.fixturenames:
        metafunc.parametrize(
            "backend_directory", [metafunc.config.getoption("backend_directory")]
        )
