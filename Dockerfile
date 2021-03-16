ARG UBUNTU_VERSION=20.04

FROM ubuntu:${UBUNTU_VERSION} as armnn_backend

# ArmNN build options
ARG ACL_DEBUG=1
ARG ARMNN_BRANCH=branches/armnn_21_02
ARG ACL_BRANCH=branches/arm_compute_21_02
ARG ARMNN_BUILD_TYPE=Debug
ARG ARMNN_BASEDIR=/opt/armnn-dev

# Triton version pins, assumed same across backend, core, and common
ARG TRITON_REPO_TAG=r21.02

# Cmake Version options
ARG CMAKE_VERSION=3.19

ENV DEBIAN_FRONTEND=noninteractive

# With ubuntu 20.04 a newer gcc/g++ is default, so we must install gcc/g++-7
# and make it the default compiler
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
        gcc-7 \
        g++-7 \
        xxd \
        rapidjson-dev \
        unzip && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 10 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 10

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

# Build ArmNN Backend
WORKDIR /opt/armnn_backend
COPY . .
RUN mkdir build && \
    cd build && \
    cmake .. \
        -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install \
        -DTRITON_BACKEND_REPO_TAG=${TRITON_REPO_TAG} \
        -DTRITON_CORE_REPO_TAG=${TRITON_REPO_TAG} \
        -DTRITON_COMMON_REPO_TAG=${TRITON_REPO_TAG} \
        -DTRITON_ENABLE_GPU=OFF \
        -DJOBS=$(nproc) \
    && \
    make -j$(nproc) install
