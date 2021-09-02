ARG UBUNTU_VERSION=20.04

FROM ubuntu:${UBUNTU_VERSION} as armnn_tflite_backend

# Triton version pins, assumed same across backend, core, and common
ARG TRITON_REPO_TAG=main

# Cmake Version options
ARG CMAKE_VERSION=3.19

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
    build-essential \
    libssl-dev \
    xxd \
    rapidjson-dev \
    unzip

# Install cmake from source
RUN build=1 && \
    mkdir /temp && \
    cd /temp && \
    wget https://cmake.org/files/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.$build.tar.gz && \
    tar -xzvf cmake-${CMAKE_VERSION}.$build.tar.gz && \
    cd cmake-${CMAKE_VERSION}.$build/ && \
    ./bootstrap --parallel=$(nproc) && \
    make -j$(nproc) && \
    make install

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
    -DJOBS=$(nproc) \
    && \
    make -j$(nproc) install
