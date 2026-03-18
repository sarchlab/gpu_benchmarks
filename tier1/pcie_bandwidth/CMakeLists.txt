# tier1/pcie_bandwidth/CMakeLists.txt
#
# Builds the pcie_bandwidth microbenchmark executable.

cmake_minimum_required(VERSION 3.18)

# The target inherits CUDA language support from the root CMakeLists.txt.

add_executable(pcie_bandwidth pcie_bandwidth.cu)

# Link against the CUDA runtime (CUDAToolkit found at root level)
target_link_libraries(pcie_bandwidth PRIVATE CUDA::cudart)

# Match C++17 / CUDA 17 standard used across the project
set_target_properties(pcie_bandwidth PROPERTIES
    CXX_STANDARD  17
    CUDA_STANDARD 17
)

install(TARGETS pcie_bandwidth RUNTIME DESTINATION bin/tier1)
