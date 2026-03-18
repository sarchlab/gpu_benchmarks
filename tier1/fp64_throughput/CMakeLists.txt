# tier1/fp64_throughput/CMakeLists.txt
#
# Builds the fp64_throughput microbenchmark executable.

cmake_minimum_required(VERSION 3.18)

# The target inherits the project's CUDA language support from the root
# CMakeLists.txt, so we don't call project() here.

add_executable(fp64_throughput fp64_throughput.cu)

# Link against the CUDA runtime (provided by the CUDAToolkit found at root)
target_link_libraries(fp64_throughput PRIVATE CUDA::cudart)

# Require C++17 / CUDA 17 consistent with the root configuration
set_target_properties(fp64_throughput PROPERTIES
    CXX_STANDARD  17
    CUDA_STANDARD 17
)

# Target sm_70 (Volta) as the minimum architecture.
# Additional architectures can be added by passing -DCMAKE_CUDA_ARCHITECTURES
# on the cmake command line (e.g. "70;80;86;89;90").
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(fp64_throughput PROPERTIES
        CUDA_ARCHITECTURES "70;80;86"
    )
endif()

install(TARGETS fp64_throughput RUNTIME DESTINATION bin/tier1)
