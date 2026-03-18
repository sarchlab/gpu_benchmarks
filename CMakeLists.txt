cmake_minimum_required(VERSION 3.18)

project(gpu_benchmarks LANGUAGES CXX CUDA)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Enable verbose NVCC output (optional, set via cmake -DCUDA_VERBOSE=ON)
option(CUDA_VERBOSE "Enable verbose CUDA compiler output" OFF)
if(CUDA_VERBOSE)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
endif()

# Subdirectories
add_subdirectory(tier1)
