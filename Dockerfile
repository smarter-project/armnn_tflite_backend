ARG UBUNTU_VERSION=20.04

FROM ubuntu:${UBUNTU_VERSION}

ARG ACL_DEBUG=1
ARG ARMNN_BRANCH=branches/armnn_21_02
ARG ACL_BRANCH=branches/arm_compute_21_02
ARG ARMNN_BUILD_TYPE=Debug
ARG BASEDIR=/opt/armnn-dev

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -yqq --no-install-recommends \
        git \
        wget \
        scons \
        cmake \
        ca-certificates \
        curl \
        autoconf \
        libtool \
        build-essential \
        g++ \
        xxd \
        unzip

# Build flatbuffers
WORKDIR $BASEDIR
RUN wget -O flatbuffers-1.12.0.zip https://github.com/google/flatbuffers/archive/v1.12.0.zip && \
    unzip -d . flatbuffers-1.12.0.zip && \
    cd flatbuffers-1.12.0 && \
    mkdir install && mkdir build && cd build && \
    # I'm using a different install directory but that is not required
    cmake .. -DCMAKE_INSTALL_PREFIX:PATH=$BASEDIR/flatbuffers-1.12.0/install && \
    make -j$(nproc) install

# Build ArmComputeLibrary
WORKDIR $BASEDIR
RUN git clone https://review.mlplatform.org/ml/ComputeLibrary && \
    cd ComputeLibrary/ && \
    git checkout ${ACL_BRANCH} && \
    # The machine used for this guide only has a Neon CPU which is why I only have "neon=1" but if 
    # your machine has an arm Gpu you can enable that by adding `opencl=1 embed_kernels=1 to the command below
    scons arch=arm64-v8a debug=${ACL_DEBUG} neon=1 opencl=1 embed_kernels=1 extra_cxx_flags="-fPIC" benchmark_tests=0 validation_tests=0 -j$(nproc) internal_only=0

WORKDIR $BASEDIR
RUN git clone "https://review.mlplatform.org/ml/armnn" && \
    cd armnn && \
    git checkout ${ARMNN_BRANCH} && \
    mkdir build && cd build && \
    # if you've got an arm Gpu add `-DARMCOMPUTECL=1` to the command below
    cmake .. \
        -DARMCOMPUTE_ROOT=$BASEDIR/ComputeLibrary \
        -DARMCOMPUTENEON=1 \
        -DARMCOMPUTECL=1 \
        -DBUILD_UNIT_TESTS=0 \
        -DCMAKE_BUILD_TYPE=${ARMNN_BUILD_TYPE} \
        -DBUILD_ARMNN_SERIALIZER=1 \
        -DARMNNREF=1 \
        -DFLATBUFFERS_ROOT=${BASEDIR}/flatbuffers-1.12.0/install \
        -DFLATC_DIR=${BASEDIR}/flatbuffers-1.12.0/build \
        -DFLATC=${BASEDIR}/flatbuffers-1.12.0/install/bin/flatc \
        && \
    make -j$(nproc)