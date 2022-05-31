cmake_minimum_required(VERSION 3.18)

project(tritonarmnntflitebackend LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(TARGET_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})

# Triton Options
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)
option(TRITON_ENABLE_MALI_GPU
       "Enable Arm MALI GPU support in backend utilities" OFF)

set(TRITON_BACKEND_REPO_TAG
    "main"
    CACHE STRING "Tag for triton-inference-server/backend repo")
set(TRITON_CORE_REPO_TAG
    "main"
    CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_COMMON_REPO_TAG
    "main"
    CACHE STRING "Tag for triton-inference-server/common repo")

# TFLite Options
set(TFLITE_TAG
    "v2.5.0"
    CACHE STRING "Version of TFLite to build")
set(BAZEL_BINARY
    "/usr/bin/bazel"
    CACHE STRING "Location of bazel binary on host")
option(TFLITE_BAZEL_BUILD "Build tflite bazel" OFF)
option(TFLITE_ENABLE_XNNPACK "Build backend with xnnpack delgate support" ON)
option(TFLITE_ENABLE_FLEX_OPS
       "Build tflite with flex ops support (must build with bazel)" OFF)

# ArmNN Options
set(ARMNN_TAG
    "v22.05"
    CACHE STRING "Tag for ArmNN repo")
set(ARMNN_BUILD_TYPE
    ${CMAKE_BUILD_TYPE}
    CACHE STRING "Set ArmNN build type")
option(ARMNN_DELEGATE_ENABLE "Build ArmNN Delegate" ON)

# ACL Options
set(ACL_TAG
    ${ARMNN_TAG}
    CACHE STRING "Tag for ACL repo")
set(ACL_DEBUG
    "0"
    CACHE STRING "Whether to compile ACL in debug mode")
set(ACL_CC_COMPILER
    ${CMAKE_C_COMPILER}
    CACHE STRING "CC compiler to use for scons acl build")
set(ACL_CXX_COMPILER
    ${CMAKE_CXX_COMPILER}
    CACHE STRING "CXX compiler to use for scons acl build")
set(ACL_ENABLE_NEON
    "1"
    CACHE STRING "Enable Neon Backend in build")
set(ACL_ENABLE_CL
    "0"
    CACHE STRING "Enable OpenCL Backend in build")

set(FLATBUFFERS_VERSION
    "1.12.0"
    CACHE STRING "Version of Flatbuffers to use")
set(JOBS
    "8"
    CACHE STRING "Number of parallel jobs for builds")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(ACL_DEBUG "1")
endif()

if(TRITON_ENABLE_MALI_GPU)
  set(ACL_ENABLE_CL "1")
endif()

# Ensure bazel build set if enabling flex ops delegate
if(TFLITE_ENABLE_FLEX_OPS AND (NOT (TFLITE_BAZEL_BUILD)))
  message(
    FATAL_ERROR
      "Please set TFLITE_BAZEL_BUILD if you wish to build flex ops delegate")
endif()

if(NOT DEFINED ACL_ARCH)
  if(${TARGET_ARCH} MATCHES "armv7")
    set(ACL_ARCH "armv7a")
  elseif(${TARGET_ARCH} MATCHES "armhf")
    set(ACL_ARCH "armv7a-hf")
  elseif(${TARGET_ARCH} MATCHES "aarch64")
    set(ACL_ARCH "arm64-v8a")
  elseif(${TARGET_ARCH} MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(ARMNN_DELEGATE_ENABLE OFF)
  else()
    message(FATAL_ERROR "Unsupported target architecture: ${TARGET_ARCH}")
  endif()
endif()

# ACL before 21.05 compiled with only gcc7
if(NOT (ACL_TAG MATCHES "(v21.(05|08|11))|(v22.(02|05))"))
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7)
    set(ACL_CC_COMPILER "gcc-7")
    set(ACL_CXX_COMPILER "g++-7")
  endif()
endif()

set(CMAKE_CXX_FLAGS "-fPIC")
set(CMAKE_C_FLAGS "-fPIC")

set(ACL_BUILD_FLAGS "-j${JOBS}" "arch=${ACL_ARCH}" "debug=${ACL_DEBUG}" "neon=${ACL_ENABLE_NEON}" "opencl=${ACL_ENABLE_CL}" "embed_kernels=1" "extra_cxx_flags=-fPIC" "benchmark_tests=0" "build=native" "validation_tests=0" "examples=0")

# ACL before 21.05 compiled with only gcc7
if(NOT (ACL_TAG MATCHES "v22.(02|05)"))
  list(APPEND ACL_BUILD_FLAGS "internal_only=0")
endif()

#
# Dependencies
#
# FetchContent's composibility isn't very good. We must include the transitive
# closure of all repos so that we can override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON)

set(MAKE_AVAILABLE_LIST repo-common repo-core repo-backend)

FetchContent_MakeAvailable(${MAKE_AVAILABLE_LIST})

#
# Shared library implementing the Triton Backend API
#
configure_file(src/libtriton_armnn_tflite.ldscript
               libtriton_armnn_tflite.ldscript COPYONLY)

include(ExternalProject)

set(TFLITE_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/external/tensorflow_lite)

if(TFLITE_BAZEL_BUILD)
  set(BAZEL_BUILD_FLAGS "--config=monolithic" "-c" "opt")

  if(TFLITE_ENABLE_RUY)
    list(APPEND BAZEL_BUILD_FLAGS "--define=tflite_with_ruy=true")
  else()
    list(APPEND BAZEL_BUILD_FLAGS "--define=tflite_with_ruy=false")
  endif()

  if(TFLITE_ENABLE_XNNPACK)
    set(BAZEL_BUILD_PATCH
        "${CMAKE_CURRENT_SOURCE_DIR}/patches/tflite_bazel_build_with_xnnpack.patch"
    )
  else()
    set(BAZEL_BUILD_PATCH
        "${CMAKE_CURRENT_SOURCE_DIR}/patches/tflite_bazel_build.patch")
  endif()

  if(TFLITE_ENABLE_FLEX_OPS)
    set(TFLITE_PATCH_COMMAND "git" "apply" ${BAZEL_BUILD_PATCH} "||" "true")
  else()
    set(TFLITE_PATCH_COMMAND "")
  endif()

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND BAZEL_BUILD_FLAGS "-c" "dbg" "--copt=-g")
  else()
    list(APPEND BAZEL_BUILD_FLAGS "--strip=always")
  endif()

  # Build tensorflow lite shared lib using bazel (required for flex ops
  # delegate)
  ExternalProject_Add(
    tensorflow-lite
    PREFIX ${TFLITE_LOCATION}
    GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
    GIT_TAG ${TFLITE_TAG}
    GIT_SHALLOW ON
    PATCH_COMMAND ${TFLITE_PATCH_COMMAND}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${BAZEL_BINARY} build ${BAZEL_BUILD_FLAGS}
                  //tensorflow/lite:libtensorflowlite.so
    BUILD_IN_SOURCE ON
    INSTALL_COMMAND "")
else()
  ExternalProject_Add(
    tensorflow-lite
    PREFIX ${TFLITE_LOCATION}
    GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
    GIT_TAG ${TFLITE_TAG}
    GIT_SHALLOW ON
    PATCH_COMMAND
      git apply ${CMAKE_CURRENT_SOURCE_DIR}/patches/xnnpack_commit.patch ||
      true # patch updates the commit hash for xnnpack for tensorflow v2.4.1
    CMAKE_ARGS
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DTFLITE_ENABLE_XNNPACK=${TFLITE_ENABLE_XNNPACK}
      -DTFLITE_ENABLE_RUY=${TFLITE_ENABLE_RUY}
      -DCMAKE_C_FLAGS="-funsafe-math-optimizations"
      -DCMAKE_CXX_FLAGS="-funsafe-math-optimizations"
    SOURCE_SUBDIR tensorflow/lite)
endif()

if(ARMNN_DELEGATE_ENABLE)
  #
  # Build ArmNN dependencies using custom command
  #
  set(FLATBUFFERS_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/external/flatbuffers)
  set(ACL_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/external/acl)
  set(ARMNN_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/external/armnn)
  file(MAKE_DIRECTORY ${ARMNN_LOCATION})

  if(ACL_TAG STREQUAL "v21.08")
    set(ACL_PATCH_COMMAND "git" "apply" "${CMAKE_CURRENT_SOURCE_DIR}/patches/acl_21_08.patch" "||" "true")
  elseif(ACL_TAG STREQUAL "v21.11")
    set(ACL_PATCH_COMMAND "git" "apply" "${CMAKE_CURRENT_SOURCE_DIR}/patches/acl_21_11.patch" "||" "true")
  else()
    set(ACL_PATCH_COMMAND "")
  endif()

  # Flatbuffers (note that tflite brings in it's own flatbuffers, so we rename
  # it here)
  ExternalProject_Add(
    flatbuffers-armnn
    URL https://github.com/google/flatbuffers/archive/v${FLATBUFFERS_VERSION}.zip
    CMAKE_ARGS
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_INSTALL_PREFIX:PATH=${FLATBUFFERS_LOCATION}/flatbuffers-${FLATBUFFERS_VERSION}/install
      -DFLATBUFFERS_BUILD_TESTS=OFF
  )

  # ACL (note requires GCC 7 if using v21.02)
  ExternalProject_Add(
    acl
    PREFIX ${ACL_LOCATION}
    GIT_REPOSITORY https://review.mlplatform.org/ml/ComputeLibrary
    GIT_TAG ${ACL_TAG}
    GIT_SHALLOW ON
    PATCH_COMMAND ${ACL_PATCH_COMMAND}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND
      CC=${ACL_CC_COMPILER} CXX=${ACL_CXX_COMPILER} scons ${ACL_BUILD_FLAGS}
    BUILD_IN_SOURCE ON
    INSTALL_COMMAND "")

  set(ARMNN_PATCH_COMMAND "")
  set(TENSORFLOW_ROOT ${TFLITE_LOCATION}/src/tensorflow-lite)
  set(TFLITE_LIB_ROOT ${TFLITE_LOCATION}/src/tensorflow-lite-build)

  if(TFLITE_BAZEL_BUILD)
    set(TFLITE_LIB_ROOT ${TFLITE_LOCATION}/src/tensorflow-lite/bazel-bin)
  endif()

  if(NOT (ACL_TAG MATCHES "v22.(02|05)"))
    # Never fail as patch is always applied after repo clone
    set(ARMNN_PATCH_COMMAND
        "git" "apply"
        "${CMAKE_CURRENT_SOURCE_DIR}/patches/tflite_armnn_delegate.patch" "||"
        "true")
  endif()

  # ArmNN and its tflite delegate
  ExternalProject_Add(
    armnn
    PREFIX ${ARMNN_LOCATION}
    GIT_REPOSITORY https://review.mlplatform.org/ml/armnn
    GIT_TAG ${ARMNN_TAG}
    GIT_SHALLOW ON
    PATCH_COMMAND ${ARMNN_PATCH_COMMAND}
    CMAKE_ARGS
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_INSTALL_PREFIX:PATH=${ARMNN_LOCATION}
      -DBUILD_ARMNN_TFLITE_DELEGATE=1
      -DTENSORFLOW_ROOT=${TENSORFLOW_ROOT}
      -DTFLITE_LIB_ROOT=${TFLITE_LIB_ROOT}
      -DARMCOMPUTE_ROOT=${ACL_LOCATION}/src/acl
      -DARMCOMPUTENEON=${ACL_ENABLE_NEON}
      -DARMCOMPUTECL=${ACL_ENABLE_CL}
      -DBUILD_UNIT_TESTS=0
      -DCMAKE_BUILD_TYPE=${ARMNN_BUILD_TYPE}
      -DBUILD_ARMNN_SERIALIZER=0
      -DARMNNREF=1
      -DFLATBUFFERS_ROOT=${FLATBUFFERS_LOCATION}/flatbuffers-${FLATBUFFERS_VERSION}/install
    DEPENDS flatbuffers-armnn acl tensorflow-lite)
endif()

#
# Handle libs for TFLite Backend
#

add_library(triton-armnn-tflite-backend SHARED
            src/tflite.cc src/tflite_utils.cc src/tflite_utils.h)

if(ARMNN_DELEGATE_ENABLE)
  add_dependencies(triton-armnn-tflite-backend armnn)
endif()

add_library(TritonArmNNTFLiteBackend::triton-armnn-tflite-backend ALIAS
            triton-armnn-tflite-backend)

set(BACKEND_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${TENSORFLOW_ROOT} # for tflite headers
    ${ARMNN_LOCATION}/include # for armnn headers
    ${ARMNN_LOCATION}/src/armnn/delegate/include # for delegate headers
)

# As per https://review.mlplatform.org/c/ml/armnn/+/7327
if(ACL_TAG MATCHES "v22.05")
  list(APPEND BACKEND_INCLUDE_DIRS ${ARMNN_LOCATION}/src/armnn/profiling)
endif()

if(TFLITE_BAZEL_BUILD)
  list(APPEND BACKEND_INCLUDE_DIRS
       ${TENSORFLOW_ROOT}/bazel-tensorflow-lite/external/flatbuffers/include)
else()
  list(APPEND BACKEND_INCLUDE_DIRS
       ${TFLITE_LIB_ROOT}/flatbuffers/include)
endif()

target_include_directories(triton-armnn-tflite-backend
                           PRIVATE ${BACKEND_INCLUDE_DIRS})

target_compile_features(triton-armnn-tflite-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-armnn-tflite-backend
  PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wno-type-limits
    -Werror>)

# ARMNN_DELEGATE_ENABLE exposed in header so set PUBLIC
if(${ARMNN_DELEGATE_ENABLE})
  target_compile_definitions(triton-armnn-tflite-backend
                             PUBLIC ARMNN_DELEGATE_ENABLE=1)
endif() # ARMNN_DELEGATE_ENABLE

set_target_properties(
  triton-armnn-tflite-backend
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_armnn_tflite
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH "$\{ORIGIN\}"
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_armnn_tflite.ldscript
    LINK_FLAGS
      "-Wl,--no-as-needed,--version-script libtriton_armnn_tflite.ldscript")

set(BACKEND_LINK_LIBS
    triton-core-serverapi triton-core-backendapi triton-core-serverstub
    triton-backend-utils ${CMAKE_DL_LIBS})

if(TFLITE_BAZEL_BUILD)
  # Link the tensorflow lite library from bazel tfile build
  target_link_libraries(
    triton-armnn-tflite-backend
    PRIVATE "-L${TFLITE_LOCATION}/src/tensorflow-lite/bazel-bin/tensorflow/lite"
            -ltensorflowlite)
else()
  # Link the tensorflow lite library from cmake tflite build
  set(TFLITE_STATIC_LIB_DIRS "-L${TFLITE_LIB_ROOT}" "-L${TFLITE_LIB_ROOT}/_deps/abseil-cpp-build/absl/strings" "-L${TFLITE_LIB_ROOT}/_deps/farmhash-build" "-L${TFLITE_LIB_ROOT}/_deps/fft2d-build" "-L${TFLITE_LIB_ROOT}/_deps/flatbuffers-build")
  set(TFLITE_STATIC_LIBS "-ltensorflow-lite" "-labsl_strings" "-lfarmhash" "-lfft2d_fftsg" "-lfft2d_fftsg2d" "-lflatbuffers")
  if(TFLITE_ENABLE_RUY)
    list(APPEND TFLITE_STATIC_LIB_DIRS "-L${TFLITE_LIB_ROOT}/_deps/ruy-build")
    list(APPEND TFLITE_STATIC_LIBS "-lruy")
  endif()
  if(TFLITE_ENABLE_XNNPACK)
    list(APPEND TFLITE_STATIC_LIB_DIRS "-L${TFLITE_LIB_ROOT}/_deps/xnnpack-build")
    list(APPEND TFLITE_STATIC_LIBS "-lXNNPACK")
  endif()
  target_link_libraries(
    triton-armnn-tflite-backend
    PRIVATE ${TFLITE_STATIC_LIB_DIRS} ${TFLITE_STATIC_LIBS}
  )
endif()

target_link_libraries(triton-armnn-tflite-backend PRIVATE ${BACKEND_LINK_LIBS})

if(ARMNN_DELEGATE_ENABLE)
  # Link the armnn lib
  target_link_libraries(
    triton-armnn-tflite-backend PRIVATE "-L${ARMNN_LOCATION}/lib" -larmnn
                                        -larmnnDelegate)
endif()

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonArmNNTFLiteBackend)

install(
  TARGETS triton-armnn-tflite-backend
  EXPORT triton-armnn-tflite-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/armnn_tflite
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/armnn_tflite)

if(ARMNN_DELEGATE_ENABLE)
  # Install ArmNN libraries and license
  install(
    DIRECTORY ${ARMNN_LOCATION}/lib/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/armnn_tflite
    FILES_MATCHING
    PATTERN "*.so*")

  install(
    FILES ${ARMNN_LOCATION}/src/armnn/LICENSE
    RENAME armnn.LICENSE
    DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/armnn_tflite)
endif()

if(TFLITE_BAZEL_BUILD)
  # Install tflite library
  install(
    FILES
      ${TFLITE_LOCATION}/src/tensorflow-lite/bazel-bin/tensorflow/lite/libtensorflowlite.so
    DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/armnn_tflite)
endif()

# Install Tensorflow license
install(
  FILES ${TENSORFLOW_ROOT}/LICENSE
  RENAME tensorflow.LICENSE
  DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/armnn_tflite)

install(
  EXPORT triton-armnn-tflite-backend-targets
  FILE TritonArmNNTFLiteBackendTargets.cmake
  NAMESPACE TritonArmNNTFLiteBackend::
  DESTINATION ${INSTALL_CONFIGDIR})

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonArmNNTFLiteBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonArmNNTFLiteBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR})

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/TritonArmNNTFLiteBackendConfig.cmake
        DESTINATION ${INSTALL_CONFIGDIR})

#
# Export from build tree
#
export(
  EXPORT triton-armnn-tflite-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonArmNNTFLiteBackendTargets.cmake
  NAMESPACE TritonArmNNTFLiteBackend::)

export(PACKAGE TritonArmNNTFLiteBackend)
