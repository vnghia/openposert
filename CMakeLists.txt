cmake_minimum_required(VERSION 3.22)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  add_compile_options(-fdiagnostics-color=always)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  add_compile_options(-fcolor-diagnostics)
endif()

# Project
project(
  openposert
  VERSION 0.0.1
  LANGUAGES CXX
)
enable_language(CUDA)
set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND CMAKE_MODULE_PATH "${PROJECT_ROOT_DIR}/cmake")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Prevent building in the source directory
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
    "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()

# Ccache
option(ENABLE_CCACHE "Automatically use ccache if available" ON)

if("${ENABLE_CCACHE}")
  find_program(CCACHE_PROGRAM ccache NAMES /usr/bin/ccache)

  if(CCACHE_PROGRAM)
    message(STATUS "ccache found: ${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CC_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  endif()
endif()

# Output directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_ROOT_DIR}/lib/${CMAKE_BUILD_TYPE}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_ROOT_DIR}/lib/${CMAKE_BUILD_TYPE}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_ROOT_DIR}/bin/${CMAKE_BUILD_TYPE}")

# CUDA
find_package(CUDAToolkit REQUIRED)

# Spdlog
include(openposert/external/spdlog)

# Cxxopts
include(openposert/external/cxxopts)

# OpenCV
find_package(OpenCV REQUIRED)

# MinRT
include(openposert/external/minrt)

# Half
include(openposert/external/half)

# openposert
add_library(${PROJECT_NAME}
  src/net/input_preprocessing.cu
  src/net/nms.cu
  src/net/output_postprocessing.cu
  src/net/reoder_and_normalize.cu
  src/net/resize_and_merge.cu

  src/pose/pose_parameters.cpp

  src/openposert.cu
)

target_compile_features(${PROJECT_NAME} PUBLIC cuda_std_17)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

target_include_directories(${PROJECT_NAME} PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  ${half_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  spdlog::spdlog
  CUDA::cudart
  ${OpenCV_LIBS}
  minrt
)

target_link_directories(${PROJECT_NAME} PUBLIC ${OpenCV_LIB_DIR})

# cli
add_executable(cli
  sample/cli.cpp
)

target_link_libraries(cli PUBLIC
  ${PROJECT_NAME}
  cxxopts::cxxopts
)
