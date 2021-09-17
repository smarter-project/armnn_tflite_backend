# ArmNN TFLite Backend

The Triton backend for [TFLite](https://www.tensorflow.org/lite) with support for ArmNN acceleration. 
You can learn more about Triton backends in the [backend
repo](https://github.com/triton-inference-server/backend). Ask
questions or report problems on the [issues
page](https://github.com/triton-inference-server/server/issues).
This backend is designed to run TFLite Serialized Models
models using the TFLite runtime.

This backend was developed using the existing [Triton PyTorch Backend](https://github.com/triton-inference-server/pytorch_backend) as reference.

This backend is only currently available for **linux arm64** platforms.

## Build the ArmNN TFLite Backend
The ArmNN TFLite backend can be built either integrated with the build process for the [triton server repo](https://github.com/triton-inference-server/server) or it may be built independently using only this repository.

### Build with Triton Build Convenience Script
The easiest way to get up and running with the triton armnn tflite backend is to build a custom triton docker image using the `build.py` script available in the triton server repo. 

To build a triton server docker image with the armnn tflite backend built in simply run the following command from the root of the server repo:
```bash
./build.py --cmake-dir=/workspace/build --build-dir=/tmp/citritonbuild --target-platform=ubuntu/arm64 --enable-logging --enable-stats --enable-tracing --enable-metrics --endpoint=http --endpoint=grpc --backend=armnn_tflite
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

### Default Optimization Options
Optimization parameters for the default tflite interpreter can be passed using the `parameters` section of the model configuration.

By default the tflite interpreter will use the maximum number of threads available to the system. 
To set the number to threads available to the tflite interpreter you can add the following section to your model configuration:
```
parameters: {
key: "tflite_num_threads"
value: {
string_value:"<num_threads>"
}
}
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
  }]
}}
```

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

## Running ArmNN TFLite Backend on MALI GPU
The best way to run the ArmNN TFLite backend on a platform with a MALI GPU is via Docker. For example on a hikey 970, we can run the following after building our custom tritonserver image using the command from the build with convenience script above:
```
docker run --rm -it --device /dev/mali0 -v /usr/lib/aarch64-linux-gnu/libmali.so:/usr/lib/aarch64-linux-gnu/libmali.so -v <full path to your model repo on host>:/models -p 8000:8000 -p 8001:8001 -p 8002:8002 tritonserver:latest
```
Then from inside the container you can invoke the server by running:
```
tritonserver --model-repository /models
```

In addition you must ensure that your instance type is set to GPU like the following:
```
instance_group [
    {
      count: 1
      kind: KIND_GPU
      gpus: [ 0 ]
    }
]
```