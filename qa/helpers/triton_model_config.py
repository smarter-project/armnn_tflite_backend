# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

from __future__ import annotations
from typing import List


class Model:
    def __init__(
        self,
        name: str,
        inputs: List[TensorIO],
        outputs: List[TensorIO],
        gpu: int = 0,
        cpu: int = 1,
        max_batch_size: int = 0,
        warm_up: bool = False,
    ):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.gpu = gpu
        self.cpu = cpu
        self.max_batch_size = max_batch_size
        self.warm_up = warm_up

    class TensorIO:
        def __init__(
            self, name: str, datatype: str, dims: List, label_filename: str = None
        ):
            self.name = name
            self.datatype = datatype
            self.dims = dims
            self.label_filename = label_filename


class TFLiteTritonModel(Model):
    def __init__(
        self,
        name: str,
        inputs: List[Model.TensorIO],
        outputs: List[Model.TensorIO],
        tflite_num_threads: int = None,
        papi_events: str = None,
        gpu: int = 0,
        cpu: int = 1,
        max_batch_size: int = 0,
        warm_up: bool = False,
        armnn_cpu: bool = False,
        armnn_gpu: bool = False,
        armnn_cpu_parameters: dict = {},
        armnn_gpu_parameters: dict = {},
        xnnpack: bool = False,
        xnnpack_parameters: dict = {},
    ):
        super().__init__(
            name,
            inputs,
            outputs,
            gpu=gpu,
            cpu=cpu,
            max_batch_size=max_batch_size,
            warm_up=warm_up,
        )
        self.tflite_num_threads = tflite_num_threads
        self.papi_events = papi_events
        self.armnn_cpu = armnn_cpu
        self.armnn_gpu = armnn_gpu
        self.armnn_cpu_parameters = armnn_cpu_parameters
        self.armnn_gpu_parameters = armnn_gpu_parameters
        self.xnnpack = xnnpack
        self.xnnpack_parameters = xnnpack_parameters

    def set_thread_count(self, num_threads: int):
        if self.armnn_cpu:
            self.armnn_cpu_parameters["num_threads"] = num_threads
        if self.xnnpack:
            self.xnnpack_parameters["num_threads"] = num_threads
        self.tflite_num_threads = num_threads
