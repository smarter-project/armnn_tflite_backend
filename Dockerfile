ARG UBUNTU_VERSION=20.04

FROM ubuntu:${UBUNTU_VERSION} as armnn_tflite_backend

# Triton version pins, assumed same across backend, core, and common
ARG TRITON_REPO_TAG=main

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
    libtool \
    zip \
    unzip \
    xxd \
    rapidjson-dev \
    software-properties-common \
    unzip && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
    gpg --dearmor - |  \
    tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake-data=${CMAKE_VERSION}-0kitware1ubuntu20.04.1 cmake=${CMAKE_VERSION}-0kitware1ubuntu20.04.1 && \
    pip3 install -U pip wheel && \
    rm -rf /var/lib/apt/lists/*

# Install Bazel from source
RUN wget -O bazel-3.1.0-dist.zip https://github.com/bazelbuild/bazel/releases/download/3.1.0/bazel-3.1.0-dist.zip && \
    unzip -d bazel bazel-3.1.0-dist.zip && \
    rm bazel-3.1.0-dist.zip && \
    cd bazel && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    env EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" bash ./compile.sh && \
    cp output/bazel /usr/bin/bazel

# Build ArmNN TFLite Backend
WORKDIR /opt/armnn_tflite_backend
COPY . .
RUN mkdir build && \
    cd build && \
    cmake .. \
    -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install \
    -DTRITON_BACKEND_REPO_TAG=${TRITON_REPO_TAG} \
    -DTRITON_CORE_REPO_TAG=${TRITON_REPO_TAG} \
    -DTRITON_COMMON_REPO_TAG=${TRITON_REPO_TAG} \
    -DTRITON_ENABLE_GPU=OFF \
    -DTRITON_ENABLE_MALI_GPU=ON \
    -DTFLITE_ENABLE_RUY=ON \
    -DJOBS=$(nproc) && \
    make -j$(nproc) install
