# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required (VERSION 3.18)

project(tritonarmnnbackend LANGUAGES C CXX)

#
# Options
#
# To build the ArmNN backend you must point to the already built ArmNN library using
#     TRITON_ARMNN_INCLUDE_PATHS and TRITON_ARMNN_LIB_PATHS
#
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")

set(ACL_DEBUG "1" CACHE STRING "Whether to compile ACL in debug mode")
set(ARMNN_BRNACH "branches/armnn_21_02" CACHE STRING "Tag for ArmNN repo")
set(ACL_BRANCH "branches/arm_compute_21_02" CACHE STRING "Tag for ACL repo")
set(ARMNN_BUILD_TYPE "Debug" CACHE STRING "Set ArmNN build type")
set(ACL_DEBUG "1" CACHE STRING "Whether to compile ACL in debug mode")
set(FLATBUFFERS_VERSION "1.12.0" CACHE STRING "Whether to compile ACL in debug mode")
set(JOBS "8" CACHE STRING "Number of parallel jobs for builds")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
# FetchContent's composibility isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

#
# Shared library implementing the Triton Backend API
#
configure_file(src/libtriton_armnn.ldscript libtriton_armnn.ldscript COPYONLY)

#
# Build ArmNN dependencies using custom command
#
include(ExternalProject)
set(FLATBUFFERS_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/external/flatbuffers)
set(ACL_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/external/acl)
set(ARMNN_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/external/armnn)
file(MAKE_DIRECTORY ${ARMNN_LOCATION})

# Flatbuffers
ExternalProject_Add(
  flatbuffers
  URL https://github.com/google/flatbuffers/archive/v${FLATBUFFERS_VERSION}.zip
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${FLATBUFFERS_LOCATION}/flatbuffers-${FLATBUFFERS_VERSION}/install
)

# ACL
ExternalProject_Add(
  acl
  PREFIX ${ACL_LOCATION}
  GIT_REPOSITORY https://review.mlplatform.org/ml/ComputeLibrary
  GIT_TAG ${ACL_BRANCH}
  GIT_SHALLOW ON
  CONFIGURE_COMMAND ""
  BUILD_COMMAND scons -j${JOBS} arch=arm64-v8a debug=${ACL_DEBUG} neon=1 opencl=1 embed_kernels=1 extra_cxx_flags=-fPIC benchmark_tests=0 build=native validation_tests=0 internal_only=0
  BUILD_IN_SOURCE ON
  INSTALL_COMMAND ""
)

# ArmNN
ExternalProject_Add(
  armnn
  PREFIX ${ARMNN_LOCATION}
  GIT_REPOSITORY https://review.mlplatform.org/ml/armnn
  GIT_TAG ${ARMNN_BRANCH}
  GIT_SHALLOW ON
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${ARMNN_LOCATION} -DARMCOMPUTE_ROOT=${ACL_LOCATION}/src/acl -DARMCOMPUTENEON=1 -DARMCOMPUTECL=1 -DBUILD_UNIT_TESTS=0 -DCMAKE_BUILD_TYPE=${ARMNN_BUILD_TYPE} -DBUILD_ARMNN_SERIALIZER=1 -DARMNNREF=1 -DFLATBUFFERS_ROOT=${FLATBUFFERS_LOCATION}/flatbuffers-${FLATBUFFERS_VERSION}/install -DFLATC_DIR=${FLATBUFFERS_LOCATION}/flatbuffers-${FLATBUFFERS_VERSION}/build -DFLATC=${FLATBUFFERS_LOCATION}/flatbuffers-${FLATBUFFERS_VERSION}/install/bin/flatc
  DEPENDS flatbuffers acl
)

#
# Handle libs for ArmNN Backend
#

add_library(
  triton-armnn-backend SHARED
  src/armnn.cc
  src/armnn_utils.cc
  src/armnn_utils.h
)

add_dependencies(triton-armnn-backend armnn)

add_library(
  TritonArmNNBackend::triton-armnn-backend ALIAS triton-armnn-backend
)

target_include_directories(
  triton-armnn-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(
  triton-armnn-backend
  PRIVATE 
    ${ARMNN_LOCATION}/include
)

target_compile_features(triton-armnn-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-armnn-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
)

set_target_properties(
  triton-armnn-backend
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_armnn
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH "$\{ORIGIN\}"
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_armnn.ldscript
    LINK_FLAGS "-Wl,--no-as-needed,--version-script libtriton_armnn.ldscript"
)

FOREACH(p ${ARMNN_LOCATION}/lib)
  set(TRITON_ARMNN_LDFLAGS ${TRITON_ARMNN_LDFLAGS} "-L${p}")
ENDFOREACH(p)

message(STATUS "Load flags are ${TRITON_ARMNN_LDFLAGS}")

target_link_libraries(
  triton-armnn-backend
  PRIVATE
    triton-core-serverapi  # from repo-core
    triton-core-backendapi # from repo-core
    triton-core-serverstub # from repo-core
    triton-backend-utils   # from repo-backend
    ${TRITON_ARMNN_LDFLAGS}
)

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonArmNNBackend)

install(
  TARGETS
    triton-armnn-backend
  EXPORT
    triton-armnn-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/armnn
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/armnn
)

install(
  EXPORT
    triton-armnn-backend-targets
  FILE
    TritonArmNNBackendTargets.cmake
  NAMESPACE
    TritonArmNNBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonArmNNBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonArmNNBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonArmNNBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-armnn-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonArmNNBackendTargets.cmake
  NAMESPACE TritonArmNNBackend::
)

export(PACKAGE TritonArmNNBackend)
