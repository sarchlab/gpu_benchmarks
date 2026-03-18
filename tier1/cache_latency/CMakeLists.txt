# tier1/cache_latency/CMakeLists.txt
#
# Builds the cache_latency microbenchmark executable.
#
# The benchmark uses a single-thread pointer-chasing kernel to measure
# round-trip latency for the L1 cache, L2 cache, and global (DRAM) memory.

cmake_minimum_required(VERSION 3.18)

# The target inherits the project's CUDA language support from the root
# CMakeLists.txt, so we don't call project() here.

add_executable(cache_latency cache_latency.cu)

# Link against the CUDA runtime (provided by the CUDAToolkit found at root).
target_link_libraries(cache_latency PRIVATE CUDA::cudart)

# Require C++17 / CUDA 17 consistent with the root configuration.
set_target_properties(cache_latency PROPERTIES
    CXX_STANDARD  17
    CUDA_STANDARD 17
)

install(TARGETS cache_latency RUNTIME DESTINATION bin/tier1)
