# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
from xprocess import ProcessStarter

import os
import socket

import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient

from collections import namedtuple

import requests

from helpers.helper_functions import load_model


def get_free_port():
    s = socket.socket()
    s.bind(("localhost", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(autouse=True)
def validate_arch(model_config):
    if os.uname().machine != "aarch64" and model_config.armnn_cpu:
        pytest.skip("ArmNN acceleration only supported on aarch64")


@pytest.fixture(scope="function")
def load_model_with_config(tritonserver_client, model_config, request):
    load_model(
        tritonserver_client.client,
        model_config,
        request.config.getoption("model_repo_path"),
    )


@pytest.fixture(scope="function")
def tritonserver_client(
    xprocess, request, http_port=get_free_port(), grpc_port=get_free_port()
):
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
                    f"http://{request.config.getoption('host')}:{http_port}/v2/health/ready"
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
            "--http-port",
            str(http_port),
            "--grpc-port",
            str(grpc_port),
        ]

        terminate_on_interrupt = True

    # ensure process is running and return its logfile
    logfile = xprocess.ensure("tritonserver", Starter)

    # Create tritonserver client
    host = request.config.getoption("host")

    if request.config.getoption("client_type") == "http":
        client = httpclient.InferenceServerClient(url=f"{host}:{http_port}")
        client_module = httpclient
    else:
        client = grpcclient.InferenceServerClient(url=f"{host}:{grpc_port}")
        client_module = grpcclient

    TritonServerClient = namedtuple(
        "TritonServerClient", ["client", "module", "triton_pid"]
    )

    yield TritonServerClient(
        client,
        client_module,
        xprocess.getinfo("tritonserver").pid,
    )

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
    parser.addoption(
        "--client-type",
        default="http",
        choices=["http", "grpc"],
        required=False,
        help="Type of client to test triton with",
    )
