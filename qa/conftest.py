# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import subprocess
import os
import socket
import psutil
from time import sleep
import json
from filelock import FileLock
import re
import sys
from multiprocessing import cpu_count

import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from collections import namedtuple

import requests

from helpers.helper_functions import load_model


def gen_free_ports(count: int):
    """Generate list of free ports with size count

    Args:
        count (int): number of open ports to include in list
    """
    free_ports = set()
    while len(free_ports) != count:
        s = socket.socket()
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
        s.close()
        free_ports.add(port)
    return list(free_ports)


def is_port_in_use(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


@pytest.fixture(scope="session")
def free_ports(tmp_path_factory, worker_id):
    if worker_id == "master":
        return gen_free_ports(count=int(os.getenv("PYTEST_XDIST_WORKER_COUNT", 1)) * 2)

    # get the temp directory shared by all workers
    root_tmp_dir = tmp_path_factory.getbasetemp().parent

    fn = root_tmp_dir / "data.json"
    with FileLock(str(fn) + ".lock"):
        if fn.is_file():
            free_ports = json.loads(fn.read_text())
        else:
            free_ports = gen_free_ports(
                count=int(os.getenv("PYTEST_XDIST_WORKER_COUNT", 1)) * 2
            )
            fn.write_text(json.dumps(free_ports))
    return free_ports


@pytest.fixture(scope="function")
def http_port(worker_id, free_ports):
    if worker_id == "master":
        return free_ports[0]
    temp = re.findall(r"\d+", worker_id)
    worker_num = list(map(int, temp))[0]
    return free_ports[worker_num]


@pytest.fixture(scope="function")
def grpc_port(worker_id, free_ports):
    if worker_id == "master":
        return free_ports[int(len(free_ports) / 2)]
    temp = re.findall(r"\d+", worker_id)
    worker_num = list(map(int, temp))[0]
    return free_ports[worker_num + int(len(free_ports) / 2)]


@pytest.fixture(autouse=True)
def validate_arch(request):
    if "model_config" in request.fixturenames:
        if os.uname().machine != "aarch64" and request.getfixturevalue("model_config").armnn_cpu:
            pytest.skip("ArmNN acceleration only supported on aarch64")
            
    if "model_configs" in request.fixturenames:
        for model_config in request.getfixturevalue("model_configs"):
            if os.uname().machine != "aarch64" and model_config.armnn_cpu:
                pytest.skip("ArmNN acceleration only supported on aarch64")


@pytest.fixture(scope="function")
def load_model_with_config(tritonserver_client, model_config, request):
    model_config.set_thread_count(request.config.getoption("model_threads_default"))
    retries = 5
    while retries > 0:
        try:
            load_model(
                tritonserver_client.client,
                model_config,
            )
            break
        except InferenceServerException:
            retries -= 1
            if retries == 0:
                sys.exit(1)
            sleep(1)


@pytest.fixture(scope="function")
def tritonserver_client(request, http_port, grpc_port):
    """
    Starts an instance of the triton server
    """

    # checks if triton is ready with request to health endpoint
    def startup_check():
        try:
            response = requests.get(
                f"http://{request.config.getoption('host')}:{http_port}/v2/health/ready"
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    # command to start process
    cmd = [
        request.config.getoption("triton_path"),
        "--log-verbose",
        "2",
        "--model-repository",
        request.config.getoption("model_repo_path"),
        "--model-control-mode",
        "explicit",
        "--allow-metrics",
        "false",
        "--backend-directory",
        request.config.getoption("backend_directory"),
        "--http-port",
        str(http_port),
        "--allow-grpc",
        str(request.config.getoption("client_type") == "grpc"),
        "--grpc-keepalive-timeout",
        "60000",
        "--strict-readiness",
        "True",
    ]

    if request.config.getoption("client_type") == "grpc":
        cmd.extend(["--grpc-port", str(grpc_port)])

    retries = 20
    while not (startup_check()):
        tritonserver = subprocess.Popen(cmd)
        sleep(1)
        retries -= 1
        if retries == 0:
            raise requests.exceptions.RequestException("Triton never became ready")

    tritonserver_process = psutil.Process(tritonserver.pid)

    # Create tritonserver client
    host = request.config.getoption("host")

    if request.config.getoption("client_type") == "http":
        client = httpclient.InferenceServerClient(
            url=f"{host}:{http_port}", connection_timeout=300, network_timeout=300
        )
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
        tritonserver_process.pid,
    )

    tritonserver.terminate()
    try:
        tritonserver.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        tritonserver.kill()
        tritonserver.communicate()

    # There seems to be a race condition in ensuring that triton is actually dead
    # and not using any of the ports, so let's busy-wait for the ports used to report
    # as free
    while is_port_in_use(http_port):
        print("HTTP port still in use")
        sleep(1)
    while is_port_in_use(grpc_port):
        print("GRPC port still in use")
        sleep(1)


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
    parser.addoption(
        "--model-threads-default",
        action="store",
        type=int,
        default=max(
            int(cpu_count() / int(os.getenv("PYTEST_XDIST_WORKER_COUNT", 1))), 1
        ),
        required=False,
        help="If not specified, number of threads to use for each tflite model",
    )
