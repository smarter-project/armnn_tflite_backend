# TFLite Backend

The Triton backend for [TFLite](https://www.tensorflow.org/lite). 
You can learn more about Triton backends in the [backend
repo](https://github.com/triton-inference-server/backend). Ask
questions or report problems on the [issues
page](https://github.com/triton-inference-server/server/issues).
This backend is designed to run TFLite Serialized Models
models using the TFLite runtime.

This backend was developed using the existing [Triton PyTorch Backend](https://github.com/triton-inference-server/pytorch_backend) as reference.

This backend is only currently available for **linux arm64** platforms.

## Build the TFLite Backend
The TFLite backend can be built either integrated with the build process for the [triton server repo](https://github.com/triton-inference-server/server) or it may be built independently using only this repository.

### Build with Triton Build Convenience Script
The easiest way to get up and running with the triton tflite backend is to build a custom triton docker image using the `build.py` script available in the triton server repo. 

To build a triton server docker image with the tflite backend built in simply run the following command from the root of the server repo:
```bash
./build.py --cmake-dir=/workspace/build --build-dir=/tmp/citritonbuild --target-platform=ubuntu/arm64 --enable-logging --enable-stats --enable-tracing --enable-metrics --endpoint=http --endpoint=grpc --backend=tflite
```

### Build Independently with CMake
Use a recent cmake to build. First install the required dependencies. Make sure you are using a cmake version greater than 3.18.

```
$ apt-get install rapidjson-dev scons gcc-9 g++-9
```

```
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
$ make install
```

The following required Triton repositories will be pulled and used in
the build. By default the "main" branch/tag will be used for each repo
but the listed CMake argument can be used to override.

* triton-inference-server/backend: -DTRITON_BACKEND_REPO_TAG=[tag]
* triton-inference-server/core: -DTRITON_CORE_REPO_TAG=[tag]
* triton-inference-server/common: -DTRITON_COMMON_REPO_TAG=[tag]

You can update the version pins for TFLite, [ArmNN](https://github.com/ARM-software/armnn) and it's dependencies ([Arm Compute Library](https://github.com/ARM-software/ComputeLibrary) and [Flatbuffers](https://github.com/google/flatbuffers)) using the following CMake arguments:

* ArmNN tag: -DARMNN_TAG=[tag]
* ACL tag: -DACL_TAG=[tag]
* Flatbuffers tag: -DFLATBUFFERS_VERSION=[tag]

## Model Repository Structure
The layout for your model repoitory remains the exact same as for other standard triton backends. Your model name should be set to `model.tflite`. An example model repository layout for ssd_mobilenetv1_coco is shown below:
```
tflite-backend-model-test
├── ssd_mobilenetv1_coco_armnn
│   ├── 1
│   │   └── model.tflite
│   └── config.pbtxt
```

## Runtime Optimizations
The backend supports both the [ArmNN](https://arm-software.github.io/armnn/latest/delegate.xhtml) and [XNNPACK](https://github.com/google/XNNPACK) TFLite delegates to accelerate inference.

An example model configuration for ssd_mobilenetv1_coco with armnn cpu execution acceleration can be seen below:
```
name: "ssd_mobilenetv1_coco_armnn"
backend: "tflite"
max_batch_size: 0
input [
  {
    name: "normalized_input_image_tensor"
    data_type: TYPE_FP32
    dims: [ 1, 300, 300, 3 ]
  }
]
output [
  {
    name: "TFLite_Detection_PostProcess"
    data_type: TYPE_FP32
    dims: [ 1, 10, 4 ]
  },
  {
    name: "TFLite_Detection_PostProcess:1"
    data_type: TYPE_FP32
    dims: [ 1, 10 ]
  },
  {
    name: "TFLite_Detection_PostProcess:2"
    data_type: TYPE_FP32
    dims: [ 1, 10 ]
  },
  {
    name: "TFLite_Detection_PostProcess:3"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }
]
optimization { execution_accelerators {
  cpu_execution_accelerator : [ { name : "armnn" } ]
}}
```

To use xnnpack acceleration in the above example, you would simply replace `armnn` with `xnnpack`

For gpu acceleration on Mali platforms the ArmNN delegate can be used. To specify gpu acceleration with ArmNN in a model configuration use:
```
optimization { execution_accelerators {
  gpu_execution_accelerator : [ { name : "armnn" } ]
}}
```

To use both cpu and gpu acceleration when available we would have:
```
optimization { execution_accelerators {
  cpu_execution_accelerator : [ { name : "armnn" } ]
  gpu_execution_accelerator : [ { name : "armnn" } ]
}}
```

### ArmNN Delegate Optimization Options
Users also have the ability to specify ArmNN specific optimizations. 
The following options are available for CPU:
```
optimization { execution_accelerators {
  cpu_execution_accelerator : [ {
    name : "armnn"
    parameters { key: "num_threads" value: "<num threads>" }
    parameters { key: "reduce_fp32_to_fp16" value: "<on/off>" }
    parameters { key: "reduce_fp32_to_bf16" value: "<on/off>" }
    parameters { key: "fast_math_enabled" value: "<on/off>" }
  }]
}}
```
And the following options are available for MALI GPU acceleration:
```
optimization { execution_accelerators {
  gpu_execution_accelerator : [ {
    name : "armnn"
    parameters { key: "reduce_fp32_to_fp16" value: "<on/off>" }
    parameters { key: "reduce_fp32_to_bf16" value: "<on/off>" }
    parameters { key: "fast_math_enabled" value: "<on/off>" }
    parameters { key: "tuning_level" value: "<0-3>" }
  }]
}}
```
Note that for MALI GPU tuning level the value corresponds to the following: `(0=UseOnly(default) | 1=RapidTuning | 2=NormalTuning | 3=ExhaustiveTuning)`

### XNNPACK Delegate Optimization Options
Users also have the ability to specify XNNPACK specific optimizations. 
```
optimization { execution_accelerators {
  cpu_execution_accelerator : [ {
    name : "xnnpack"
    parameters { key: "num_threads" value: "<num threads>" }
  }]
}}
```