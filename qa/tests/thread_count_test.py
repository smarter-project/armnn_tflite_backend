# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest

import psutil

import tritonclient.http as httpclient
from typing import List
from statistics import stdev, mean
import numpy as np

from helpers.triton_model_config import Model, TFLiteTritonModel
from helpers.helper_functions import (
    load_model,
    send_inference_request,
    is_server_ready,
)


def get_inference_threads_cpu_percent(
    p: psutil.Process,
    base_time: float,
    exclude_threads: List[int],
    previous_thread_id_times: dict,
):
    total_time = sum(p.cpu_times()) - base_time

    # Populate dict with current thread user cpu times indexed by thread id
    non_triton_threads = {
        t.id: t.user_time for t in p.threads() if t.id not in exclude_threads
    }

    # Subtract shared thread id times where applicable and update prev thread dict
    for tid in non_triton_threads:
        if tid in previous_thread_id_times:
            delta = non_triton_threads[tid] - previous_thread_id_times[tid]
            non_triton_threads[tid] = delta
            previous_thread_id_times[tid] += delta
        else:
            previous_thread_id_times[tid] = non_triton_threads[tid]

    return {
        tid: ((non_triton_threads[tid]) / total_time) * 100
        for tid in non_triton_threads
    }


@pytest.mark.parametrize(
    "model_configs",
    [
        [
            TFLiteTritonModel(
                "resnet_v2_101_fp32",
                [Model.TensorIO("input", "TYPE_FP32", [1, 299, 299, 3])],
                [Model.TensorIO("output", "TYPE_FP32", [1001])],
            )
        ],
        [
            TFLiteTritonModel(
                "resnet_v2_101_fp32",
                [Model.TensorIO("input", "TYPE_FP32", [1, 299, 299, 3])],
                [Model.TensorIO("output", "TYPE_FP32", [1001])],
                armnn_cpu=True,
            )
        ],
        [
            TFLiteTritonModel(
                "resnet_v2_101_fp32",
                [Model.TensorIO("input", "TYPE_FP32", [1, 299, 299, 3])],
                [Model.TensorIO("output", "TYPE_FP32", [1001])],
                xnnpack=True,
            )
        ],
        [
            TFLiteTritonModel(
                "resnet_v2_101_fp32",
                [Model.TensorIO("input", "TYPE_FP32", [1, 299, 299, 3])],
                [Model.TensorIO("output", "TYPE_FP32", [1001])],
            ),
            TFLiteTritonModel(
                "mobilenet_v2_1.0_224",
                [Model.TensorIO("input", "TYPE_FP32", [1, 224, 224, 3])],
                [
                    Model.TensorIO(
                        "MobilenetV2/Predictions/Reshape_1",
                        "TYPE_FP32",
                        [1001],
                    )
                ],
                armnn_cpu=True,
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
                        [1001],
                    ),
                ],
                xnnpack=True,
            ),
            TFLiteTritonModel(
                "resnet_v2_101_fp32",
                [Model.TensorIO("input", "TYPE_FP32", [1, 299, 299, 3])],
                [Model.TensorIO("output", "TYPE_FP32", [1001])],
            ),
        ],
        [
            TFLiteTritonModel(
                "resnet_v2_101_fp32",
                [Model.TensorIO("input", "TYPE_FP32", [1, 299, 299, 3])],
                [Model.TensorIO("output", "TYPE_FP32", [1001])],
                armnn_cpu=True,
            ),
            TFLiteTritonModel(
                "inceptionv3",
                [Model.TensorIO("input", "TYPE_FP32", [1, 299, 299, 3])],
                [
                    Model.TensorIO(
                        "InceptionV3/Predictions/Reshape_1",
                        "TYPE_FP32",
                        [1001],
                    ),
                ],
                armnn_cpu=True,
            ),
        ],
    ],
)
@pytest.mark.parametrize(
    "num_threads",
    [1, 2, 4, 8],
)
def test_correct_thread_count(
    tritonserver_client,
    model_configs,
    num_threads,
):
    if tritonserver_client.module != httpclient:
        pytest.skip("Thread count test only runs for http client")

    if len(model_configs) > 1 and all(
        [model_config.armnn_cpu for model_config in model_configs]
    ):
        pytest.xfail("Multiple ArmNN models will end up sharing same threads")

    triton_process = psutil.Process(tritonserver_client.triton_pid)

    triton_base_thread_ids = [t.id for t in triton_process.threads()]

    for model_config in model_configs:
        model_config.set_thread_count(num_threads)

        load_model(
            tritonserver_client.client,
            model_config,
        )

        assert is_server_ready(tritonserver_client.client)

    previous_thread_id_times = {}
    for i, model_config in enumerate(model_configs):
        # Prime the cpu_percent counter and base time before inference
        base_time = sum(triton_process.cpu_times())
        for _ in range(10):
            send_inference_request(
                tritonserver_client.client, tritonserver_client.module, model_config
            )
        percents = get_inference_threads_cpu_percent(
            triton_process, base_time, triton_base_thread_ids, previous_thread_id_times
        )
        assert (
            len(percents) >= num_threads
        ), "Not enough thread ids found during measurement"
        sorted_percents = dict(
            sorted(percents.items(), reverse=True, key=lambda item: item[1])
        )
        print(sorted_percents)

        # Assert the proportional cpu usage is mostly spent on the tflite inference threads
        top_percents = [*sorted_percents.values()][:num_threads]
        if num_threads > 1:
            relative_std_err = (
                100 * (stdev(top_percents) / np.sqrt(num_threads)) / mean(top_percents)
            )
            # Cpu usage should be spread relatively evenly across the threads
            assert relative_std_err < 20.0
        assert sum(top_percents) > 75.0

        # At this point we have verified the number of threads was correct, so we can add
        # the thread ids used for inference directly to the threads to ignore next time
        # we measure relative cpu usage
        triton_base_thread_ids.extend([*sorted_percents][:num_threads])
