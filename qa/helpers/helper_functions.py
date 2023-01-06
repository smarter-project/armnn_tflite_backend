# Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

from jinja2 import Template
import numpy as np
from tritonclient.utils import triton_to_np_dtype


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


def load_model(inference_client, model_config, model_repo_path: str):
    if inference_client.is_model_ready(model_config.name):
        inference_client.unload_model(model_config.name)

    if model_config:

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
