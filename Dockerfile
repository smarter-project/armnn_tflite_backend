ARG UBUNTU_VERSION=20.04

FROM ubuntu:${UBUNTU_VERSION} as armnn_tflite_backend

ENV DEBIAN_FRONTEND=noninteractive

# Cmake Version options
ARG CMAKE_VERSION=3.21.1

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -yqq --no-install-recommends \
    git \
    wget \
    scons \
    ca-certificates \
    curl \
    autoconf \
    libtool \
    python3-dev \
    python3-pip \
    python3-numpy \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    default-jdk \
    zip \
    unzip \
    xxd \
    rapidjson-dev \
    software-properties-common && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
    gpg --dearmor - |  \
    tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake-data=${CMAKE_VERSION}-0kitware1ubuntu20.04.1 cmake=${CMAKE_VERSION}-0kitware1ubuntu20.04.1 && \
    pip3 install -U pip wheel && \
    rm -rf /var/lib/apt/lists/*

# Triton version pins, assumed same across backend, core, and common
# Note that this is set to the rX.XX branches, not the vX.X.X tags
ARG TRITON_REPO_TAG=main

# CMake build arguments defaults
ARG CMAKE_BUILD_TYPE=RELEASE
ARG TRITON_ENABLE_MALI_GPU=ON
ARG TFLITE_ENABLE_RUY=ON
ARG TFLITE_BAZEL_BUILD=OFF
ARG TFLITE_ENABLE_FLEX_OPS=OFF
ARG TFLITE_TAG=v2.10.0
ARG ARMNN_VERSION=23.02
ARG ARMNN_DELEGATE_ENABLE=ON
ARG ACL_VERSION=${ARMNN_VERSION}

# Install Bazel from source
RUN if [ "$TFLITE_BAZEL_BUILD" = "ON" ]; then wget -O bazel-3.1.0-dist.zip https://github.com/bazelbuild/bazel/releases/download/3.1.0/bazel-3.1.0-dist.zip && \
    unzip -d bazel bazel-3.1.0-dist.zip && \
    rm bazel-3.1.0-dist.zip && \
    cd bazel && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    env EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" bash ./compile.sh && \
    cp output/bazel /usr/bin/bazel; else echo "Not using bazel in build"; fi

# Configure ArmNN TFLite Backend first and build tflite lib, then build backend.
# This allows us to cache as much as we can at the expense of disk space
WORKDIR /opt/armnn_tflite_backend
COPY . .
RUN cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DTRITON_BACKEND_REPO_TAG=${TRITON_REPO_TAG} \
    -DTRITON_CORE_REPO_TAG=${TRITON_REPO_TAG} \
    -DTRITON_COMMON_REPO_TAG=${TRITON_REPO_TAG} \
    -DTRITON_ENABLE_GPU=OFF \
    -DTRITON_ENABLE_MALI_GPU=${TRITON_ENABLE_MALI_GPU} \
    -DTFLITE_ENABLE_RUY=${TFLITE_ENABLE_RUY} \
    -DTFLITE_BAZEL_BUILD=${TFLITE_BAZEL_BUILD} \
    -DTFLITE_ENABLE_FLEX_OPS=${TFLITE_ENABLE_FLEX_OPS} \
    -DTFLITE_TAG=${TFLITE_TAG} \
    -DARMNN_VERSION=${ARMNN_VERSION} \
    -DARMNN_DELEGATE_ENABLE=${ARMNN_DELEGATE_ENABLE} \
    -DACL_VERSION=${ACL_VERSION} \
    -DJOBS=$(nproc) && \
    cmake --build build -j $(nproc) -t install 
