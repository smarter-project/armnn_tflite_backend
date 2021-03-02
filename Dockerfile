ARG UBUNTU_VERSION=18.04

FROM ubuntu:${UBUNTU_VERSION} as tritonserver_armnn

# ArmNN build options
ARG ACL_DEBUG=1
ARG ARMNN_BRANCH=branches/armnn_21_02
ARG ACL_BRANCH=branches/arm_compute_21_02
ARG ARMNN_BUILD_TYPE=Debug
ARG ARMNN_BASEDIR=/opt/armnn-dev

# Triton version pins, assumed same across backend, core, and common
ARG TRITON_REPO_TAG=r21.02

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
        g++ \
        xxd \
        unzip

# Install cmake 3.19
RUN version=3.19 && \
    build=1 && \
    mkdir /temp && \
    cd /temp && \
    wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz && \
    tar -xzvf cmake-$version.$build.tar.gz && \
    cd cmake-$version.$build/ && \
    ./bootstrap --parallel=$(nproc) && \
    make -j$(nproc) && \
    make install

# Build flatbuffers
WORKDIR $ARMNN_BASEDIR
RUN wget -O flatbuffers-1.12.0.zip https://github.com/google/flatbuffers/archive/v1.12.0.zip && \
    unzip -d . flatbuffers-1.12.0.zip && \
    cd flatbuffers-1.12.0 && \
    mkdir install && mkdir build && cd build && \
    # I'm using a different install directory but that is not required
    cmake .. -DCMAKE_INSTALL_PREFIX:PATH=$ARMNN_BASEDIR/flatbuffers-1.12.0/install && \
    make -j$(nproc) install

# Build ArmComputeLibrary
WORKDIR $ARMNN_BASEDIR
RUN git clone https://review.mlplatform.org/ml/ComputeLibrary && \
    cd ComputeLibrary/ && \
    git checkout ${ACL_BRANCH} && \
    # The machine used for this guide only has a Neon CPU which is why I only have "neon=1" but if 
    # your machine has an arm Gpu you can enable that by adding `opencl=1 embed_kernels=1 to the command below
    scons arch=arm64-v8a debug=${ACL_DEBUG} neon=1 opencl=1 embed_kernels=1 extra_cxx_flags="-fPIC" benchmark_tests=0 validation_tests=0 -j$(nproc) internal_only=0

# Build ArmNN
WORKDIR $ARMNN_BASEDIR
RUN git clone "https://review.mlplatform.org/ml/armnn" && \
    cd armnn && \
    git checkout ${ARMNN_BRANCH} && \
    mkdir build && cd build && \
    # if you've got an arm Gpu add `-DARMCOMPUTECL=1` to the command below
    cmake .. \
        -DARMCOMPUTE_ROOT=$ARMNN_BASEDIR/ComputeLibrary \
        -DARMCOMPUTENEON=1 \
        -DARMCOMPUTECL=1 \
        -DBUILD_UNIT_TESTS=0 \
        -DCMAKE_BUILD_TYPE=${ARMNN_BUILD_TYPE} \
        -DBUILD_ARMNN_SERIALIZER=1 \
        -DARMNNREF=1 \
        -DFLATBUFFERS_ROOT=${ARMNN_BASEDIR}/flatbuffers-1.12.0/install \
        -DFLATC_DIR=${ARMNN_BASEDIR}/flatbuffers-1.12.0/build \
        -DFLATC=${ARMNN_BASEDIR}/flatbuffers-1.12.0/install/bin/flatc \
        && \
    make -j$(nproc)

# Build ArmNN Backend
WORKDIR /opt/armnn_backend
RUN apt install -yqq rapidjson-dev
COPY . .
RUN mkdir build && \
    cd build && \
    cmake .. \
        -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install \
        -DTRITON_ARMNN_INCLUDE_PATHS=${ARMNN_BASEDIR}/armnn/include \
        -DTRITON_ARMNN_LIB_PATHS=${ARMNN_BASEDIR}/armnn/build \
        -DTRITON_BACKEND_REPO_TAG=${TRITON_REPO_TAG} \
        -DTRITON_CORE_REPO_TAG=${TRITON_REPO_TAG} \
        -DTRITON_COMMON_REPO_TAG=${TRITON_REPO_TAG} \
        -DTRITON_ENABLE_GPU=OFF \
    && \
    make -j$(nproc) install
