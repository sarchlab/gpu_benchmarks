# tier1/memory_bandwidth/CMakeLists.txt
#
# Builds the memory_bandwidth microbenchmark executable.

cmake_minimum_required(VERSION 3.18)

# The target inherits the project's CUDA language support from the root
# CMakeLists.txt, so we don't call project() here.

add_executable(memory_bandwidth memory_bandwidth.cu)

# Link against the CUDA runtime (provided by the CUDAToolkit found at root)
target_link_libraries(memory_bandwidth PRIVATE CUDA::cudart)

# Require C++17 / CUDA 17 consistent with the root configuration
set_target_properties(memory_bandwidth PROPERTIES
    CXX_STANDARD  17
    CUDA_STANDARD 17
)

# Optional: enable separable compilation for larger projects
# set_target_properties(memory_bandwidth PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS memory_bandwidth RUNTIME DESTINATION bin/tier1)
