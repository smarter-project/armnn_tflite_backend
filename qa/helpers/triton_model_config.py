# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

from jinja2 import Template

class Model:
    def __init__(self, name, inputs, outputs, gpu=0, cpu=1, max_batch_size=0):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.gpu = gpu
        self.cpu = cpu
        self.max_batch_size = max_batch_size

    class TensorIO:
        def __init__(self, name, datatype, dims, label_filename=None):
            self.name = name
            self.datatype = datatype
            self.dims = dims
            self.label_filename = label_filename


class TFLiteTritonModel(Model):
    def __init__(
        self,
        name,
        inputs,
        outputs,
        gpu=0,
        cpu=1,
        max_batch_size=0,
        armnn_cpu=False,
        armnn_gpu=False,
        armnn_cpu_parameters={},
        armnn_gpu_parameters={},
        xnnpack=False,
        xnnpack_parameters={},
    ):
        super(TFLiteTritonModel, self).__init__(
            name, inputs, outputs, gpu, cpu, max_batch_size
        )
        self.armnn_cpu = armnn_cpu
        self.armnn_gpu = armnn_gpu
        self.armnn_cpu_parameters = armnn_cpu_parameters
        self.armnn_gpu_parameters = armnn_gpu_parameters
        self.xnnpack = xnnpack
        self.xnnpack_parameters = xnnpack_parameters
      
        
def load_model(inference_client, model_config, model_repo_path):
    if inference_client.is_model_ready(model_config.name):
        inference_client.unload_model(model_config.name)
        
    if model_config and model_repo_path:

        with open("config-template.pbtxt") as file_:
            template = Template(
                file_.read(),
                trim_blocks=True,
                lstrip_blocks=True,
                keep_trailing_newline=True,
            )
        output_config = template.render(model=model_config)
        with open(
            f"{model_repo_path}/{model_config.name}/config.pbtxt",
            "w+",
        ) as output_file_:
            output_file_.write(output_config)

    inference_client.load_model(model_config.name)
