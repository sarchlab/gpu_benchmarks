# GPU Benchmarks

A comprehensive GPGPU benchmark suite covering microbenchmarks, application workloads, library kernels, alternative language implementations, and ML workloads.

---

## Overview

This suite is organized into five tiers of increasing complexity and scope, allowing researchers and engineers to evaluate GPU hardware and software stacks at every level — from raw memory bandwidth to end-to-end ML training throughput.

---

## 5-Tier Structure

| Tier | Name | Description |
|------|------|-------------|
| **Tier 1** | Microbenchmarks | Low-level hardware characterization: memory bandwidth, compute throughput (FP32/FP64/FP16), cache latency, PCIe transfer rates |
| **Tier 2** | Application Benchmarks | Classic GPU benchmark suites: SHOC, PolyBench-GPU, Rodinia, Parboil |
| **Tier 3** | Library Benchmarks | Vendor and third-party library kernels: cuBLAS, CUTLASS, cuFFT, cuSPARSE, cuDNN |
| **Tier 4** | Alternative Languages | Non-CUDA implementations: Triton, OpenCL, HIP, SYCL |
| **Tier 5** | ML Workloads | End-to-end machine learning training/inference: MLPerf, torchbenchmark, model-specific workloads |

---

## Directory Layout

```
gpu_benchmarks/
├── CMakeLists.txt          # Root build file
├── README.md               # This file
├── tier1/                  # Microbenchmarks (CUDA)
│   ├── CMakeLists.txt
│   ├── memory_bandwidth/
│   ├── fp32_throughput/
│   ├── fp64_throughput/
│   ├── fp16_throughput/
│   ├── cache_latency/
│   └── pcie_bandwidth/
├── tier2/                  # Application benchmarks
├── tier3/                  # Library benchmarks
├── tier4/                  # Alternative language benchmarks
├── tier5/                  # ML workloads
└── scripts/                # Helper scripts (run-all, report generation, etc.)
```

---

## Tier 1 Microbenchmarks

| Benchmark | Metric | Description |
|-----------|--------|-------------|
| `memory_bandwidth` | GB/s | Global memory bandwidth (H2D, D2H, device-to-device) |
| `fp32_throughput` | GFLOPS | Single-precision floating-point throughput |
| `fp64_throughput` | GFLOPS | Double-precision floating-point throughput |
| `fp16_throughput` | GFLOPS | Half-precision (FP16/BF16) floating-point throughput |
| `cache_latency` | ns / cycles | L1/L2 cache latency via pointer-chasing |
| `pcie_bandwidth` | GB/s | PCIe transfer bandwidth between host and device |

---

## How to Build

### Prerequisites

- CMake ≥ 3.18
- CUDA Toolkit ≥ 11.0 (with `nvcc`)
- A C++17-capable host compiler (GCC ≥ 9, Clang ≥ 10, or MSVC ≥ 2019)

### Build Steps

```bash
# Clone the repository
git clone https://github.com/sarchlab/gpu_benchmarks.git
cd gpu_benchmarks

# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build all benchmarks
cmake --build build --parallel

# (Optional) Build only Tier 1
cmake --build build --target tier1 --parallel
```

### Run a Benchmark

Each benchmark produces a standalone executable in the corresponding build subdirectory:

```bash
./build/tier1/memory_bandwidth/memory_bandwidth
./build/tier1/fp32_throughput/fp32_throughput
# etc.
```

---

## Reference Documents

- [`spec.md`](spec.md) — Project specification
- [`roadmap.md`](roadmap.md) — Development roadmap and milestones
- [`benchmark_report.md`](benchmark_report.md) — Research report: benchmark selection and rationale

---

## License

See [LICENSE](LICENSE) for details.
