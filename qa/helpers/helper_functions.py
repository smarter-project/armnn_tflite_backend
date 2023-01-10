# Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

from jinja2 import Template
import numpy as np
from tritonclient.utils import triton_to_np_dtype
from tritonclient.grpc import model_config_pb2
from google.protobuf import json_format, text_format
import json
from time import sleep
import sys


def get_random_triton_inputs(model_input_info, batch_size, client_type):
    inputs = []
    for i, input in enumerate(model_input_info):
        input_dims = [int(i) for i in input.dims]

        if batch_size:
            input_dims.insert(0, batch_size)

        dtype_str = input.datatype.split("_")[1]
        inputs.append(client_type.InferInput(input.name, input_dims, dtype_str))

        # Initialize input data using random values
        random_array = np.random.randn(*input_dims).astype(
            triton_to_np_dtype(dtype_str), copy=False
        )
        inputs[i].set_data_from_numpy(random_array)
    return inputs


def load_model(inference_client, model_config):
    json_string_config = None
    if model_config:
        with open("config-template.pbtxt") as file_:
            template = Template(
                file_.read(),
                trim_blocks=True,
                lstrip_blocks=True,
                keep_trailing_newline=True,
            )
        output_config = str(template.render(model=model_config))
        print(output_config)
        protobuf_message = text_format.Parse(
            output_config, model_config_pb2.ModelConfig()
        )
        model_config_dict = json_format.MessageToDict(protobuf_message)
        json_string_config = json.dumps(model_config_dict)
        print(json_string_config)

    retries = 10
    while not (inference_client.is_server_ready()):
        sleep(1)
        retries -= 1
        if retries == 0:
            sys.exit(1)

    inference_client.load_model(model_config.name, config=json_string_config)


def unload_model(inference_client, model_name):
    retries = 10
    while not (inference_client.is_server_ready()):
        sleep(1)
        retries -= 1
        if retries == 0:
            sys.exit(1)

    inference_client.unload_model(model_name)
