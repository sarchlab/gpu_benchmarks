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
**Goal:** Produce a comprehensive research report identifying the key benchmarks to include in each tier.
**Deliverable:** `benchmark_report.md` — 60+ benchmarks across 5 tiers.

---

### M2 — Project Structure + Tier 1 Microbenchmarks ✅ COMPLETE (cycles used: 2)
**Goal:** Establish project directory layout and implement Tier 1 microbenchmarks (originally CMake/CUDA).

---

### M3 — Refactor Tier 1: Make + Multi-GPU Support ✅ COMPLETE (cycles used: 2)
**Goal:** Rework all 6 Tier 1 benchmarks to use Make + HIP + Metal (no CMake).
**Deliverables (completed):**
- Per-benchmark Makefiles (PLATFORM=cuda|rocm|metal auto-detect)
- HIP source (.hip) for NVIDIA+AMD, Metal source (.metal + .mm) for Apple
- Top-level Makefile, tier1/Makefile
- Updated README files

---

### M4 — Tier 2 Application Benchmarks: SHOC Core + Rodinia HotSpot (current)
**Goal:** Implement 6 high-priority Tier 2 application benchmarks from SHOC and Rodinia, following the same project conventions as Tier 1.

**Reference implementations:** `/Users/yifan/.thebotcompany/dev/src/github.com/sarchlab/mgpusim-dev/repo/workloads/` has CUDA and HIP variants for most benchmarks. Use them as starting points.

**Deliverables:**
- `tier2/shoc_triad/` — Stream Triad (STREAM-style memory bandwidth)
- `tier2/shoc_reduction/` — Parallel sum reduction
- `tier2/shoc_scan/` — Prefix scan (exclusive)
- `tier2/shoc_gemm/` — Dense matrix multiply (naive + tiled)
- `tier2/shoc_devicememory/` — Device memory global bandwidth (read/write/readwrite)
- `tier2/rodinia_hotspot/` — 2D thermal simulation stencil
- `tier2/Makefile` — iterates over all tier2 subdirectories
- Root `Makefile` updated to include `tier2` target

**Per-benchmark requirements (same as Tier 1):**
- `<name>.hip` — HIP source (NVIDIA + AMD via hipcc; single source, hip/hip_runtime.h)
- `<name>.metal` + `<name>_metal.mm` — Apple Metal (compute shader + ObjC++ host)
- `Makefile` — platform auto-detect (Darwin→metal, /opt/rocm→rocm, else→cuda)
- `README.md` — documents build/run for all platforms
- **Self-contained**: copy any needed common utilities inline; NO imports from shared directories

**Branch:** `ares/m4-tier2`
**Cycles budget:** 6

---

### M5 — Tier 2: Top Rodinia + PolyBench (planned)
Implement top Rodinia benchmarks (BFS, NW, SRAD) and PolyBench GPU (GEMM, 2DConv, atax).

### M6 — Tier 3 Library Benchmarks (planned)
cuBLAS GEMM, CUTLASS GEMM, cuFFT, cuSPARSE benchmark wrappers.

### M7 — Tier 4 & 5: Triton + ML Workloads (planned)
Triton-based benchmarks and representative ML training/inference workloads.

### M8 — Integration & Harness (planned)
Unified benchmark runner: make-based run-all, result collection, reporting infrastructure.

---

## Lessons Learned
- Research phase produced excellent coverage (60+ benchmarks). One cycle sufficient.
- Human constraints (Make, multi-GPU, self-contained) established after M2 — apply to all future work proactively.
- Each benchmark is fully independent from the start — don't build shared infrastructure.
- mgpusim-dev repo has mature CUDA+HIP implementations for Tier 2; use as reference, adapt to our format (inline common headers, add Metal, unify Makefile).
- Parallel worker assignment (1 benchmark per worker) works well — 6 benchmarks finished in ~1 cycle.

## Cycle History
| Cycle | Phase | Agent | Action |
|-------|-------|-------|--------|
| 1 | Planning | Athena | Created roadmap, hired Iris for research |
| 2 | Implementation | Iris | Produced benchmark_report.md (60+ benchmarks) |
| 3 | Planning | Athena | Evaluated M1 output, defined M2 |
| 4–5 | Implementation | Ares+workers | Built project structure + 6 Tier 1 benchmarks |
| 6 | Verification | Apollo | Verified M2, merged PR #1 |
| 7 | Planning | Athena | Updated spec with human constraints (issue #10), defined M3 |
| 8–9 | Implementation | Ares+workers | Refactored Tier 1: Makefile + HIP + Metal for all 6 benchmarks |
| 10 | Verification | Apollo | Verified M3, merged PR #2 |
| 11 | Planning | Athena | Evaluated state, defined M4 (Tier 2: SHOC + Rodinia) |
