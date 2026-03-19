# Project Specification

## What do you want to build?

We want to create an extensive GPGPU benchmark suite. We need to include 5 tiers of benchmarks: 
* Tier 1, microbenchmarks
* Tier 2, benchmarks (shoc, polybench, rodinia, parboil style)
* Tier 3, libraries (cuBLAS, cutlass, etc.)
* Tier 4, other languages like Triton
* Tier 5, ML workloads

We want to be as extensive as possible. We should include popular benchmark suites.

Every benchmark must be **parameterized** with well-defined scaling and non-scaling parameters. A CLI tool must be able to compile, run, and collect results into an SQLite database with platform info, parameter values, and kernel-level timing.

## How do you consider the project is success?

A comprehensive, well-organized benchmark suite covering all 5 tiers with extensive coverage. Must include modern ML/LLM workloads and broad Tier 2 application benchmark coverage (8+ distinct suites beyond the initial set). Every benchmark must have:
1. Full parameterization (scaling + non-scaling parameters with documented values)
2. Kernel-level execution timing
3. A CLI that can compile, run, and collect results into SQLite

## Constraints

The following constraints apply to all benchmarks across all tiers:

1. **Build system: Make only.** Each benchmark must use a `Makefile`. No CMake or other build systems. Each benchmark directory contains its own `Makefile`.

2. **Multi-GPU platform support:** Every benchmark must support:
   - **NVIDIA** — compiled with `nvcc` (CUDA)
   - **AMD** — compiled with `hipcc` (ROCm/HIP)
   - **Apple** — compiled with Metal toolchain (macOS)
   The `Makefile` must auto-detect the platform (or accept `PLATFORM=cuda|rocm|metal`) and invoke the correct compiler.

3. **Self-contained benchmarks.** Each benchmark lives in its own directory and has **no shared or common dependencies** with other benchmarks. All required code is within the benchmark's own directory. The only exception is Tier 5 (ML workloads), which may depend on ML frameworks (PyTorch, JAX, etc.).

4. **Complete benchmarks.** If we skip some workloads, we need to have a clear reason. (from human, Issue #142)

## Parameterization Requirements (from human, Issue #143)

For every benchmark:
1. **Identify a single scaling parameter.** The overall performance should be predictable with linear regression (or polynomial/exponential) if other factors don't change.
2. **Identify all non-scaling parameters.**
3. **Identify parameter values.** For non-scaling factors, select 3–5 values per parameter. For scaling factors, select 20–30 values.
4. **Document the parameter selection.**

## CLI Requirements (from human, Issue #144)

Build a CLI tool that:
1. **Compiles everything** — or a specific suite/benchmark.
2. **Runs benchmarks with repetitions** — default 7 repetitions per configuration.
3. **Supports filtering** — run one suite or one benchmark.
4. **Generates SQLite output** containing:
   - Platform information (as detailed as possible: GPU model, driver version, OS, etc.)
   - Execution results: tier, suite, benchmark, scaling parameter value, non-scaling parameter values, execution time
   - Kernel-level execution times (one kernel per entry)

## Tier 5 — Modern ML Workloads (from human, Issue #100)

The suite needs comprehensive modern ML workloads including:

1. **LLM Inference Frameworks**: vLLM, ollama, llama.cpp inference workloads
2. **Popular Open-Source LLM Models**: LLaMA, OpenAI OSS models, Qwen, GLM (ChatGLM), DeepSeek
3. **Both large and small LLM workloads**
4. **PyTorch training workloads** with popular models
5. **Non-LLM models**: Diffusion models (Stable Diffusion, SDXL)
6. **Advanced patterns**: Chain of thought (CoT), Mixture of Experts (MoE), RAG (Retrieval-Augmented Generation), multi-agent execution
7. It is OK if benchmarks cannot execute on the local development environment (e.g., require specific GPU hardware or large models)

## Tier 2 — More Benchmark Suites (from human, Issue #101)

Add at least 8 more distinct benchmark suites beyond the existing ones (SHOC, Rodinia, PolyBench, Parboil, CUDA Samples). Each suite should include representative benchmarks.

## Notes

- The `benchmark_report.md` at the repo root is the reference document for which benchmarks to include in each tier.
- Tier 1 microbenchmarks have already been implemented (M2) with CMake/CUDA. They must be refactored to meet the constraints above.
- Tier 5 Python benchmarks may depend on ML frameworks (PyTorch, transformers, vllm, diffusers, etc.)
- Benchmarks that require large models or specific hardware should still be included with correct code — they don't need to run on the development machine.
