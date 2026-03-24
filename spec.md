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
4. **CI must pass (green)** before any PR is merged or the project is declared complete (from human, Issue #276)

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

## CLI Requirements (from human, Issues #144 and #217)

Build a CLI tool **implemented in Go** (from human, Issue #217) that:
1. **Compiles everything** — or a specific suite/benchmark.
2. **Runs benchmarks with repetitions** — default 7 repetitions per configuration.
3. **Supports filtering** — run one suite or one benchmark.
4. **Generates SQLite output** containing:
   - Platform information (as detailed as possible: GPU model, driver version, OS, etc.)
   - Execution results: tier, suite, benchmark, scaling parameter value, non-scaling parameter values, execution time
   - Kernel-level execution times (one kernel per entry)

The CLI must be a Go program that compiles to a single binary. It must maintain full feature parity with the previous Python implementation: same commands (list, compile, run, info, validate), same flags, same SQLite schema, same JSON-lines output protocol parsing, same exit codes.

## Compilation & Error Reporting (from human, Issue #221) ✅ RESOLVED

1. **Compilation errors must always be visible.** When `gpubench compile` encounters a failure, it must show the full compiler error output (stderr from `make`) by default — not just a ✗ symbol. Users should not need `--verbose` to see why compilation failed.
2. **All Tier 2 benchmarks must compile successfully** on all supported platforms (CUDA, ROCm, Metal).

## Native CUDA Source Files (from human, Issue #224)

**Write CUDA programs, not only HIP.** Every C++ benchmark must have a native `.cu` source file using the CUDA API (`cudaMalloc`, `cudaMemcpy`, `#include <cuda_runtime.h>`, etc.) for NVIDIA compilation. HIP source (`.hip`) files continue to exist for AMD/ROCm compilation. Benchmarks must NOT rely on HIP-to-CUDA compatibility shims for NVIDIA builds — they must be proper CUDA programs.

File convention per C++ benchmark:
- `name.cu` — Native CUDA source (compiled with `nvcc` for NVIDIA)
- `name.hip` — Native HIP source (compiled with `hipcc` for AMD)
- `name_metal.mm` + `name.metal` — Metal source (for Apple)

## Complete Benchmark Suites (from human, Issue #226)

If we include a benchmark suite, we must include **all** its benchmarks — not just a selection. For any benchmarks we skip, we need a documented, clear reason.

## MLPerf Benchmarks (from human, Issue #228)

Investigate MLPerf benchmarks and include all relevant ones. What ML benchmarks do they have? Can we include all of them?

## Result Persistence (from human, Issue #229)

Design a mechanism for users to persist and share benchmark results with the community. Options include:
- Contributing results to a GitHub repository
- Using a shared database
- Standardized result format for submission

## Fix Compilation Errors (from human, Issues #268, #273, #10)

8 benchmarks failed to compile on CUDA (Issue #273):
- **Tier 2:** cuda_nbody, graph_cc, graph_pr, heteromark_pagerank, lonestar_bh, lonestar_sssp
- **Tier 3:** cudnn_batchnorm, cudnn_conv

Initial root cause for Tier 2 was missing `#include <cstdint>` in `.cu` files using `uint32_t`.

**Current regression (human issue #10):** compile summary still reports two failures:
- `cudnn_batchnorm`
- `cudnn_conv`

These two benchmarks must have robust CUDA/cuDNN detection and deterministic compile behavior with clear diagnostics (no opaque linker/header failures). The project is not considered complete until issue #10 is resolved.

All benchmarks must compile successfully on all supported platforms, or fail with explicit dependency diagnostics that the CLI surfaces clearly.

## CI Must Pass — Use Self-Hosted Runners (from human, Issues #269, #272, #276)

The CI pipeline must pass. GitHub Actions CI fails due to billing issues. **Use self-hosted CI runners** (Issue #272) instead of `ubuntu-latest`. The `sarchlab` org has self-hosted runners available. Ensure Docker containers have all required libraries (including cuDNN for Tier 3 DNN benchmarks).

**CI must be green (all checks passing) before any PR can be merged or the project can be declared complete.** (from human, Issue #276)

**Known CI infrastructure issues (discovered in M34):**
- Self-hosted runners are ARM64 (`linux/arm64/v8`), runner name: `marin-5-linux`
- Docker containers inside the runner have no internet access (container network bridge is isolated — host can pull images but containers can't run `apt-get`)
- ROCm Docker images are x86_64 only — they crash immediately on ARM64 runners
- Fix: Use `--network host` for Docker containers; handle ROCm gracefully on ARM64

## Docker-Based CI (from human, Issue #223)

We need Docker-based CI for NVIDIA/AMD/Apple compilation verification. At minimum:
- Docker containers with NVIDIA CUDA toolkit
- Docker containers with AMD ROCm
- Apple compilation may require specific CI runners (Docker doesn't support macOS natively)
The CI should verify that all benchmarks compile successfully on all platforms.

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

## Workspace Files Must Not Be in Repo (from human, Issue #176)

The `workspace/` directory (containing agent skill files, project state DB, workspace notes, etc.) has been accidentally committed to the git repository. It must be removed from git tracking and added to `.gitignore`. The workspace directory should only exist at the filesystem level under `.thebotcompany`, not in the project repo.

## Benchmark Failure Diagnostics (from human, Issue #177)

Users need to be able to understand WHY a benchmark fails in their environment. Currently, when `gpubench run` encounters a failure, the error reporting is minimal — stderr is only shown with `--verbose`. The CLI must provide clear, actionable diagnostic information by default when benchmarks fail. This includes:
1. **Always show stderr** on benchmark failure (not just with `--verbose`)
2. **Common failure diagnosis** — identify and report common causes like: missing compilation (binary not found), missing GPU/driver, missing libraries, unsupported platform, out of memory, etc.
3. **Actionable suggestions** — tell the user what to do next (e.g., "Run `gpubench compile --benchmark X` first", "Install CUDA toolkit", etc.)

## Persistent Compilation Failures (from human, tbc-db Issue #1)

8 benchmarks still fail to compile in the human's CUDA environment (different from CI):
- **Tier 2:** cuda_nbody, graph_cc, graph_pr, heteromark_pagerank, lonestar_bh, lonestar_sssp
- **Tier 3:** cudnn_batchnorm, cudnn_conv

Human's compile summary: Passed: 80, Failed: 8. Note: CI (nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04 on ARM64) passes all 88. The failures occur in a different CUDA environment. The code must compile across all reasonable CUDA environments, not just the CI container.

## Remove Broken Submodule Entries from Repo (from human, tbc-db Issue #2)

The human asks: "What is repo and repo-chat?" These are broken git submodule entries (`repo` and `repo-chat`) committed to the repository without a `.gitmodules` file. They appear to be self-referential clones/worktrees. They must be removed from git tracking and added to `.gitignore`. Additionally, other accidentally tracked files should be cleaned up: `config.yaml`, `results.db*`, `test_tier1.db*`.

## Notes

- The `benchmark_report.md` at the repo root is the reference document for which benchmarks to include in each tier.
- Tier 1 microbenchmarks have already been implemented (M2) with CMake/CUDA. They must be refactored to meet the constraints above.
- Tier 5 Python benchmarks may depend on ML frameworks (PyTorch, transformers, vllm, diffusers, etc.)
- Benchmarks that require large models or specific hardware should still be included with correct code — they don't need to run on the development machine.
