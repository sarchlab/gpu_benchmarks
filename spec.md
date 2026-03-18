# Project Specification

## What do you want to build?

We want to create an extensive GPGPU benchmark suite. We need to include 4 tiers of benchmarks. 
* Tier 1, microbenchmarks
* Tier 2, benchmarks (shoc, polybench, rodinia, parboil style)
* Tier 3, libraries (cuBLAS, cutlass, etc.)
* Tier 4, other languages like Triton, 
* Tier 5, ML workloads. 

We want to be as extensive as possible. We should include popular benchmark suites.  As the first stage, please research which benchmarks to include. Generate a report for me, and I will give you more instructions for the next step.

## How do you consider the project is success?

(Not yet defined by human.)

## Constraints

The following constraints apply to all benchmarks across all tiers:

1. **Build system: Make only.** Each benchmark must use a `Makefile`. No CMake or other build systems. Each benchmark directory contains its own `Makefile`.

2. **Multi-GPU platform support:** Every benchmark must support:
   - **NVIDIA** — compiled with `nvcc` (CUDA)
   - **AMD** — compiled with `hipcc` (ROCm/HIP)
   - **Apple** — compiled with Metal toolchain (macOS)
   The `Makefile` must auto-detect the platform (or accept `PLATFORM=cuda|rocm|metal`) and invoke the correct compiler.

3. **Self-contained benchmarks.** Each benchmark lives in its own directory and has **no shared or common dependencies** with other benchmarks. All required code is within the benchmark's own directory. The only exception is Tier 5 (ML workloads), which may depend on ML frameworks (PyTorch, JAX, etc.).

## Notes

- The `benchmark_report.md` at the repo root is the reference document for which benchmarks to include in each tier.
- Tier 1 microbenchmarks have already been implemented (M2) with CMake/CUDA. They must be refactored to meet the constraints above.
