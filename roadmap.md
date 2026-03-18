# GPU Benchmarks — Project Roadmap

## Project Goal
Create an extensive GPGPU benchmark suite covering 5 tiers:
- **Tier 1**: Microbenchmarks (memory bandwidth, latency, compute throughput, etc.)
- **Tier 2**: Application benchmarks (SHOC, PolyBench, Rodinia, Parboil, etc.)
- **Tier 3**: Library benchmarks (cuBLAS, cuDNN, CUTLASS, etc.)
- **Tier 4**: Alternative language benchmarks (Triton, OpenCL, HIP, SYCL, etc.)
- **Tier 5**: ML workloads (MLPerf, torchbenchmark, model-specific, etc.)

## Project-Wide Constraints (from human)
1. **Make only** — no CMake or other build systems
2. **Multi-GPU**: NVIDIA (nvcc/CUDA), AMD (hipcc/ROCm), Apple (Metal)
3. **Self-contained**: each benchmark is one directory, no shared dependencies (except Tier 5 may use ML frameworks)

## Success Criteria
Deliver a comprehensive, well-organized benchmark suite that researchers and engineers can use to evaluate GPGPU hardware and software stacks.

---

## Milestones

### M1 — Research & Report ✅ COMPLETE (cycles used: 2)
**Goal:** Produce a comprehensive research report identifying the key benchmarks to include in each tier, justifying selections, and proposing a concrete implementation plan.

**Deliverable:** `benchmark_report.md` committed to repo — 60+ benchmarks across 5 tiers, with complexity/priority ratings, implementation strategy, and references.

---

### M2 — Project Structure + Tier 1 Microbenchmarks ✅ COMPLETE (cycles used: 2)
**Goal:** Establish the project's directory layout, build system, and implement the highest-priority Tier 1 microbenchmarks.

**Note:** Completed with CMake/CUDA. M3 will refactor to meet the updated project constraints (Make + multi-GPU).

**Deliverables (completed):**
- Directory structure: `tier1/`, `tier2/`, `tier3/`, `tier4/`, `tier5/`, `scripts/`
- Root `CMakeLists.txt` (to be replaced in M3)
- 6 Tier 1 microbenchmarks: memory BW, FP32/FP64/FP16 throughput, L1/L2 cache latency, PCIe BW
- Each benchmark: CUDA source, `README.md`

---

### M3 — Refactor Tier 1: Make + Multi-GPU Support (current)
**Goal:** Rework all 6 Tier 1 benchmarks to comply with the project-wide constraints. Replace CMake with per-benchmark Makefiles, add AMD ROCm and Apple Metal support.

**Deliverables:**
- Remove all `CMakeLists.txt` files from `tier1/` and the root (replace root CMakeLists.txt with a top-level Makefile or remove it)
- Each of the 6 Tier 1 benchmarks gets a **self-contained `Makefile`** supporting:
  - `PLATFORM=cuda` (nvcc) — default on Linux with CUDA
  - `PLATFORM=rocm` (hipcc) — AMD ROCm
  - `PLATFORM=metal` (xcrun metal) — Apple GPU
- Source code refactored to use HIP portability layer for NVIDIA+AMD (single `.hip`/`.cu` source), and Metal compute shader (`.metal` + `.mm` host) for Apple
- Updated `README.md` in each benchmark documenting multi-GPU build/run instructions
- Top-level `Makefile` at the repo root with `make all`, `make clean`, `make tier1`

**Benchmarks to refactor:**
1. `tier1/memory_bandwidth` (Ben)
2. `tier1/fp32_throughput` (Cara)
3. `tier1/fp64_throughput` (Dan)
4. `tier1/fp16_throughput` (Eva)
5. `tier1/cache_latency` (Finn)
6. `tier1/pcie_bandwidth` (Gina)
7. Root structure + top-level Makefile (Felix)

**Cycles budget:** 5

---

### M4 — Tier 2 Application Benchmarks (planned)
Implement/integrate selected Tier 2 benchmarks using Make + multi-GPU support. Priority: SHOC core kernels, top-5 Rodinia, top-5 PolyBench.

### M5 — Tier 3 Library Benchmarks (planned)
Implement cuBLAS GEMM, CUTLASS GEMM, cuFFT, cuSPARSE benchmark wrappers with Make + multi-GPU support.

### M6 — Tier 4 & 5: Triton + ML Workloads (planned)
Implement Triton-based benchmarks and representative ML training/inference workloads.

### M7 — Integration & Harness (planned)
Unified benchmark runner: make-based run-all, result collection, reporting infrastructure.

---

## Lessons Learned
- Research phase produced excellent coverage (60+ benchmarks, all tiers). One cycle was sufficient.
- The benchmark_report.md is the definitive guide for implementation order and scope.
- Human constraints (Make, multi-GPU, self-contained) established after M2. Apply retroactively to M3, and proactively to all future milestones.
- Each benchmark should be treated as fully independent from the start — don't build shared infrastructure that needs to be undone later.

## Cycle History
| Cycle | Phase | Agent | Action |
|-------|-------|-------|--------|
| 1 | Planning | Athena | Created roadmap, hired Iris for research |
| 2 | Implementation | Iris | Produced benchmark_report.md (60+ benchmarks) |
| 3 | Planning | Athena | Evaluated M1 output, defined M2 |
| 4–5 | Implementation | Ares+workers | Built project structure + 6 Tier 1 benchmarks |
| 6 | Verification | Apollo | Verified M2, merged PR #1 |
| 7 | Planning | Athena | Updated spec with human constraints (issue #10), defined M3 |
