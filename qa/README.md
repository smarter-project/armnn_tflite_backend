# Testing
To run tests you must first build triton server in your local environment. 
To do so you can run the following:
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

# Need boost > 1.78
wget -O /tmp/boost.tar.gz \
    https://boostorg.jfrog.io/artifactory/main/release/1.80.0/source/boost_1_80_0.tar.gz && \
    (cd /tmp && tar xzf boost.tar.gz) && \
    mv /tmp/boost_1_80_0/boost /usr/include/boost
    
./build.py -v --no-container-build --build-dir=`pwd`/build --backend=identity --endpoint=grpc --endpoint=http --enable-logging --enable-stats --enable-tracing --enable-metrics
```

Then change directory back to the base of this repo build it using cmake (instructions for this can be found in the github ci). With the backend built you can then run:
```
cd qa
wget -q https://images.getsmarter.io/ml-models/armnn_tflite_backend_triton_model_repo.tar.gz
tar -xzf armnn_tflite_backend_triton_model_repo.tar.gz
python3 -m pytest tests/ --model-repo-path $(pwd)/triton_qa_models/accuracy_test_model_repo --triton-path /workspaces/server/build/opt/tritonserver/bin/tritonserver --backend-directory $(pwd)/../install/backends -v
```
