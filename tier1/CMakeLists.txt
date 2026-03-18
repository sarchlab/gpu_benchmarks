# Tier 1 — Microbenchmarks
# Each subdirectory contains a self-contained CUDA benchmark.

add_subdirectory(memory_bandwidth)
add_subdirectory(fp32_throughput)
add_subdirectory(fp64_throughput)
add_subdirectory(fp16_throughput)
add_subdirectory(cache_latency)
add_subdirectory(pcie_bandwidth)
