# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import subprocess


def test_l0_batcher(triton_qa_model_repo_ver):
    test_proc = subprocess.Popen(
        ["bash", "-x", "./test.sh", triton_qa_model_repo_ver],
        env={
            "BACKENDS": "armnn_tflite",
            "DIFFERENT_SHAPE_TESTS": "test_multi_batch_not_preferred_different_shape test_multi_batch_preferred_different_shape test_multi_batch_different_shape",
        },
        cwd="/opt/tritonserver/qa/L0_batcher",
    )
    assert test_proc.wait() == 0


def test_l0_infer(triton_qa_model_repo_ver):
    test_proc = subprocess.Popen(
        ["bash", "-x", "./test.sh", triton_qa_model_repo_ver],
        env={
            "BACKENDS": "armnn_tflite",
            "TRITON_SERVER_CPU_ONLY": "1",
            "ENSEMBLES": "0",
            "EXPECTED_NUM_TESTS": "25",
        },
        cwd="/opt/tritonserver/qa/L0_infer",
    )
    assert test_proc.wait() == 0


# def test_l0_perf_no_model():
#     test_proc = subprocess.Popen(
#         ["bash", "-x", "./test.sh", "21.08"],
#         env={"BACKENDS": "armnn_tflite"},
#         cwd="/opt/tritonserver/qa/L0_perf_no_model",
#     )
#     assert test_proc.wait() == 0


# def test_l0_sequence_batcher():
#     test_proc = subprocess.Popen(
#         ["bash", "-x", "./test.sh", "21.08"],
#         env={"BACKENDS": "armnn_tflite"},
#         cwd="/opt/tritonserver/qa/L0_sequence_batcher",
#     )
#     assert test_proc.wait() == 0


# def test_l0_warmup():
#     test_proc = subprocess.Popen(
#         ["bash", "-x", "./test.sh", "21.08"],
#         env={"BACKENDS": "armnn_tflite"},
#         cwd="/opt/tritonserver/qa/L0_warmup",
#     )
#     assert test_proc.wait() == 0
