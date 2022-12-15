# Testing

The CI in this repository runs on github actions. To build and test locally you can use [act](https://github.com/nektos/act), which runs github workflows locally against your machine. You can install act and docker on your local machine, then from the root of this repo simply run:
```bash
mkdir /tmp/artifacts
act -P self-hosted=catthehacker/ubuntu:act-22.04 --detect-event --artifact-server-path /tmp/artifacts -W .github/workflows/cmake.yml
```
This will build and test the backend.

If you wish to run tests without using the github workflow, you first build triton server in your local environment:
```bash
git clone https://github.com/triton-inference-server/server.git
cd server
git checkout {branch_name}
apt update
apt-get install -y --no-install-recommends \
    ca-certificates \
    autoconf \
    automake \
    build-essential \
    docker.io \
    git \
    libre2-dev \
    libssl-dev \
    libtool \
    libboost-dev \
    libcurl4-openssl-dev \
    libb64-dev \
    patchelf \
    python3-dev \
    python3-pip \
    python3-setuptools \
    rapidjson-dev \
    scons \
    software-properties-common \
    unzip \
    wget \
    zlib1g-dev \
    libarchive-dev \
    pkg-config \
    uuid-dev \
    libnuma-dev
./build.py -v --no-container-build --build-dir=`pwd`/build --backend=identity --endpoint=grpc --endpoint=http --enable-logging --enable-stats --enable-tracing --enable-metrics
```

Then change directory back to the base of this repo build it using cmake (instructions for this can be found in the github ci). With the backend built you can then run:
```
cd qa
wget -q https://images.getsmarter.io/ml-models/armnn_tflite_backend_triton_model_repo.tar.gz
tar -xzf armnn_tflite_backend_triton_model_repo.tar.gz
python3 -m pytest tests/ --model-repo-path $(pwd)/triton_qa_models/accuracy_test_model_repo --triton-path /workspaces/server/build/opt/tritonserver/bin/tritonserver --backend-directory $(pwd)/../install/backends -v
```
