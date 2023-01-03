# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
from xprocess import ProcessStarter

import os

import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient

import requests

from helpers.triton_model_config import load_model
    
@pytest.fixture(autouse=True)
def validate_arch(model_config):
    if os.uname().machine != "aarch64" and model_config.armnn_cpu:
        pytest.skip("ArmNN acceleration only supported on aarch64")


@pytest.fixture
def load_model_with_config(inference_client, model_config, request):
    load_model(inference_client, model_config, request.config.getoption('model_repo_path'))

    yield

    inference_client.unload_model(model_config.name)


@pytest.fixture
def inference_client(client_type, request):
    host = request.config.getoption("host")
    if client_type == httpclient:
        client = httpclient.InferenceServerClient(url=str(host) + ":8000")
    else:
        client = grpcclient.InferenceServerClient(url=str(host) + ":8001")

    return client


@pytest.fixture(scope="module")
def tritonserver(xprocess, request):
    """
    Starts an instance of the triton server
    """

    class Starter(ProcessStarter):
        pattern = "Started \w* at \d.\d.\d.\d:\d*"
        timeout = 15

        # checks if triton is ready with request to health endpoint
        def startup_check(self):
            try:
                response = requests.get(
                    f"http://{request.config.getoption('host')}:8000/v2/health/ready"
                )
                return response.status_code == 200
            except requests.exceptions.RequestException:
                return False

        # command to start process
        args = [
            request.config.getoption("triton_path"),
            "--model-repository",
            request.config.getoption("model_repo_path"),
            "--model-control-mode",
            "explicit",
            "--backend-directory",
            request.config.getoption("backend_directory"),
        ]

        terminate_on_interrupt = True

    # ensure process is running and return its logfile
    logfile = xprocess.ensure("tritonserver", Starter)

    yield

    # clean up whole process tree afterwards
    xprocess.getinfo("tritonserver").terminate()


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
        default="accuracy_test_model_repo",
        required=False,
        help="Path to top level of triton model repository",
    )
    parser.addoption(
        "--triton-path",
        action="store",
        default="/tmp/citritonbuild/opt/tritonserver/bin/tritonserver",
        required=False,
        help="Path to triton executable",
    )
    parser.addoption(
        "--backend-directory",
        action="store",
        default="/tmp/citritonbuild/opt/tritonserver/backends",
        required=False,
        help="Path to triton backends",
    )
