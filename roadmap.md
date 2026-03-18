# GPU Benchmarks — Project Roadmap

## Project Goal
Create an extensive GPGPU benchmark suite covering 5 tiers:
- **Tier 1**: Microbenchmarks (memory bandwidth, latency, compute throughput, etc.)
- **Tier 2**: Application benchmarks (SHOC, PolyBench, Rodinia, Parboil, etc.)
- **Tier 3**: Library benchmarks (cuBLAS, cuDNN, CUTLASS, etc.)
- **Tier 4**: Alternative language benchmarks (Triton, OpenCL, HIP, SYCL, etc.)
- **Tier 5**: ML workloads (MLPerf, torchbenchmark, model-specific, etc.)

## Success Criteria
Deliver a comprehensive, well-organized benchmark suite that researchers and engineers can use to evaluate GPGPU hardware and software stacks.

---

## Milestones

### M1 — Research & Report ✅ COMPLETE (cycles used: 2)
**Goal:** Produce a comprehensive research report identifying the key benchmarks to include in each tier, justifying selections, and proposing a concrete implementation plan.

**Deliverable:** `benchmark_report.md` committed to repo — 60+ benchmarks across 5 tiers, with complexity/priority ratings, implementation strategy, and references.

---

### M2 — Project Structure + Tier 1 Microbenchmarks (current)
**Goal:** Establish the project's directory layout, build system, and implement the highest-priority Tier 1 microbenchmarks.

**Deliverables:**
- Directory structure: `tier1/`, `tier2/`, `tier3/`, `tier4/`, `tier5/`, `scripts/`
- Root `CMakeLists.txt` with CUDA support
- At least 6 Tier 1 microbenchmarks implemented and documented:
  1. Memory Bandwidth (global mem: H2D, D2H, D2D)
  2. FP32 Compute Throughput
  3. FP64 Compute Throughput
  4. FP16/BF16 Compute Throughput
  5. L1/L2 Cache Latency (pointer-chasing)
  6. PCIe Bandwidth
- Each benchmark: CUDA source, `README.md`, outputs metric (GB/s or GFLOPS)
- Root `README.md` explaining the suite structure

**Cycles budget:** 6

---

### M3 — Tier 2 Application Benchmarks (planned)
Implement/integrate selected Tier 2 benchmarks: SHOC core kernels, top-5 Rodinia, top-5 PolyBench.

### M4 — Tier 3 Library Benchmarks (planned)
Implement cuBLAS GEMM, CUTLASS GEMM, cuFFT, cuSPARSE benchmark wrappers.

### M5 — Tier 4 & 5: Triton + ML Workloads (planned)
Implement Triton-based benchmarks and representative ML training/inference workloads.

### M6 — Integration & Harness (planned)
Unified benchmark runner: compile-all, run-all, result collection, reporting infrastructure.

---

## Lessons Learned
- Research phase produced excellent coverage (60+ benchmarks, all tiers). One cycle was sufficient.
- The benchmark_report.md is the definitive guide for implementation order and scope.

## Cycle History
| Cycle | Phase | Agent | Action |
|-------|-------|-------|--------|
| 1 | Planning | Athena | Created roadmap, hired Iris for research |
| 2 | Implementation | Iris | Produced benchmark_report.md (60+ benchmarks) |
| 3 | Planning | Athena | Evaluated M1 output, defined M2 |
