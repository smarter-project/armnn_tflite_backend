cmake_minimum_required (VERSION 3.18)

project(tritontflitebackend LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)
option(TRITON_ENABLE_MALI_GPU "Enable Arm MALI GPU support in backend utilities" ON)

set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")

set(TFLITE_TAG "v2.4.1" CACHE STRING "Version of TFLite to build")

set(ACL_DEBUG "0" CACHE STRING "Whether to compile ACL in debug mode")
set(ARMNN_TAG "v21.05" CACHE STRING "Tag for ArmNN repo")
set(ARMNN_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Set ArmNN build type")
set(ACL_TAG "v21.05" CACHE STRING "Tag for ACL repo")
set(ACL_GCC_MAJOR_VERSION "9" CACHE STRING "GCC major version")
set(FLATBUFFERS_VERSION "1.12.0" CACHE STRING "Version of Flatbuffers to use")
set(JOBS "8" CACHE STRING "Number of parallel jobs for builds")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(ACL_DEBUG "1")
endif()

# ACL before 21.05 compiled with only gcc7
if (NOT(ACL_TAG STREQUAL "v21.05"))
  set(ACL_GCC_MAJOR_VERSION "7")
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
  GIT_REPOSITORY https://github.com/jishminor/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  tensorflow-lite
  GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
  GIT_TAG ${TFLITE_TAG}
  GIT_SHALLOW ON
  PATCH_COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/patches/xnnpack_commit.patch || true # patch updates the commit hash for xnnpack for tensorflow v2.4.1
  SOURCE_SUBDIR tensorflow/lite
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend tensorflow-lite)

#
# Shared library implementing the Triton Backend API
#
configure_file(src/libtriton_tflite.ldscript libtriton_tflite.ldscript COPYONLY)

include(ExternalProject)

#
# Build ArmNN dependencies using custom command
#
set(FLATBUFFERS_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/external/flatbuffers)
set(ACL_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/external/acl)
set(ARMNN_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/external/armnn)
file(MAKE_DIRECTORY ${ARMNN_LOCATION})

# Flatbuffers (note that tflite brings in it's own flatbuffers, so we rename it here)
ExternalProject_Add(
  flatbuffers-armnn
  URL https://github.com/google/flatbuffers/archive/v${FLATBUFFERS_VERSION}.zip
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${FLATBUFFERS_LOCATION}/flatbuffers-${FLATBUFFERS_VERSION}/install
)

# ACL (note requires GCC 7 if using v21.02)
ExternalProject_Add(
  acl
  PREFIX ${ACL_LOCATION}
  GIT_REPOSITORY https://review.mlplatform.org/ml/ComputeLibrary
  GIT_TAG ${ACL_TAG}
  GIT_SHALLOW ON
  CONFIGURE_COMMAND ""
  BUILD_COMMAND CC=gcc-${ACL_GCC_MAJOR_VERSION} CXX=g++-${ACL_GCC_MAJOR_VERSION} scons -j${JOBS} arch=arm64-v8a debug=${ACL_DEBUG} neon=1 opencl=1 embed_kernels=1 extra_cxx_flags=-fPIC benchmark_tests=0 build=native validation_tests=0 internal_only=0 examples=0
  BUILD_IN_SOURCE ON
  INSTALL_COMMAND ""
)

# ArmNN and it's tflite delegate
ExternalProject_Add(
  armnn
  PREFIX ${ARMNN_LOCATION}
  GIT_REPOSITORY https://review.mlplatform.org/ml/armnn
  GIT_TAG ${ARMNN_TAG}
  GIT_SHALLOW ON
  PATCH_COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/patches/tflite_armnn_delegate.patch || true # Never fail as patch is always applied after repo clone
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${ARMNN_LOCATION} -DBUILD_ARMNN_TFLITE_DELEGATE=1 -DTENSORFLOW_ROOT=${CMAKE_CURRENT_BINARY_DIR}/_deps/tensorflow-lite-src -DTFLITE_LIB_ROOT=${CMAKE_CURRENT_BINARY_DIR}/_deps/tensorflow-lite-build -DARMCOMPUTE_ROOT=${ACL_LOCATION}/src/acl -DARMCOMPUTENEON=1 -DARMCOMPUTECL=1 -DBUILD_UNIT_TESTS=0 -DCMAKE_BUILD_TYPE=${ARMNN_BUILD_TYPE} -DBUILD_ARMNN_SERIALIZER=0 -DARMNNREF=1 -DFLATBUFFERS_ROOT=${FLATBUFFERS_LOCATION}/flatbuffers-${FLATBUFFERS_VERSION}/install
  DEPENDS flatbuffers-armnn acl tensorflow-lite
)

#
# Handle libs for TFLite Backend
#

add_library(
  triton-tflite-backend SHARED
  src/tflite.cc
  src/tflite_utils.cc
  src/tflite_utils.h
)

add_dependencies(triton-tflite-backend armnn)

add_library(
  TritonTFLiteBackend::triton-tflite-backend ALIAS triton-tflite-backend
)

target_include_directories(
  triton-tflite-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}/_deps/tensorflow-lite-src
    ${ARMNN_LOCATION}/include # for armnn headers
    ${ARMNN_LOCATION}/src/armnn/delegate/include # for delegate headers
)

target_compile_features(triton-tflite-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-tflite-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
)

set_target_properties(
  triton-tflite-backend
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_tflite
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH "$\{ORIGIN\}"
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_tflite.ldscript
    LINK_FLAGS "-Wl,--no-as-needed,--version-script libtriton_tflite.ldscript"
)

target_link_libraries(
  triton-tflite-backend
  PRIVATE
    triton-core-serverapi  # from repo-core
    triton-core-backendapi # from repo-core
    triton-core-serverstub # from repo-core
    triton-backend-utils   # from repo-backend
    tensorflow-lite
    "-L${ARMNN_LOCATION}/lib" # from armnn
    -larmnn
    -larmnnDelegate
    ${CMAKE_DL_LIBS}
)

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonTFLiteBackend)

install(
  TARGETS
    triton-tflite-backend
  EXPORT
    triton-tflite-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/tflite
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/tflite
)

# Install ArmNN libraries
install(
  DIRECTORY
    ${ARMNN_LOCATION}/lib/
  DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/tflite
  FILES_MATCHING PATTERN "*.so*"
)

# Install ArmNN and Tensorflow licenses
install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/_deps/tensorflow-lite-src/LICENSE
  RENAME tensorflow.LICENSE
  DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/tflite
)

install(
  FILES
    ${ARMNN_LOCATION}/src/armnn/LICENSE
  RENAME armnn.LICENSE
  DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/tflite
)

install(
  EXPORT
    triton-tflite-backend-targets
  FILE
    TritonTFLiteBackendTargets.cmake
  NAMESPACE
    TritonTFLiteBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonTFLiteBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonTFLiteBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonTFLiteBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-tflite-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonTFLiteBackendTargets.cmake
  NAMESPACE TritonTFLiteBackend::
)

export(PACKAGE TritonTFLiteBackend)
