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

### M4 — Tier 2 Application Benchmarks: SHOC Core + Rodinia HotSpot ✅ COMPLETE (cycles used: 3)
**Goal:** Implement 6 high-priority Tier 2 application benchmarks from SHOC and Rodinia.
**Deliverables (completed):** shoc_triad, shoc_reduction, shoc_scan, shoc_gemm, shoc_devicememory, rodinia_hotspot — all with HIP + Metal + Makefile + README.

---

### M5 — Tier 2: Rodinia BFS/NW/SRAD + PolyBench GEMM/2DConv/FDTD2D ✅ COMPLETE (cycles used: ~8)
**Goal:** Implement 6 more Tier 2 benchmarks — 3 Rodinia + 3 PolyBench — following the same project conventions.
**Deliverables (completed):** rodinia_bfs, rodinia_nw, rodinia_srad, polybench_gemm, polybench_2dconv, polybench_fdtd2d — all with HIP + Metal + Makefile + README. Merged via PR #4.

---

### M6 — Tier 2: SHOC Sort + Parboil SGEMM/SpMV/Stencil + PolyBench 3mm/ATAX ✅ COMPLETE (cycles used: ~3)
**Goal:** Implement 6 more high-priority Tier 2 benchmarks covering SHOC, Parboil, and PolyBench suites.
**Deliverables (completed):** shoc_sort, parboil_sgemm, parboil_spmv, parboil_stencil, polybench_3mm, polybench_atax — all with HIP + Metal + Makefile + README. Merged via PR #5.

---

### M7 — Tier 2: Rodinia Gaussian/LUD/PathFinder/K-Means + PolyBench BICG/MVT (current)
**Goal:** Implement 6 more Tier 2 benchmarks — 4 Rodinia + 2 PolyBench — following the same project conventions.

**Deliverables:**
- `tier2/rodinia_gaussian/` — Gaussian elimination with back-substitution (dense linear system solver)
- `tier2/rodinia_lud/` — LU decomposition for dense matrices
- `tier2/rodinia_pathfinder/` — Dynamic programming on a 2D grid
- `tier2/rodinia_kmeans/` — K-means clustering algorithm
- `tier2/polybench_bicg/` — BiCG sub-kernel of BiCGStab (two matrix-vector ops)
- `tier2/polybench_mvt/` — Matrix-vector product and matrix-vector transpose

**Per-benchmark requirements (same as M4/M5/M6):**
- `<name>.hip` — HIP source (NVIDIA + AMD via hipcc)
- `<name>.metal` + `<name>_metal.mm` — Apple Metal (compute shader + ObjC++ host)
- `Makefile` — platform auto-detect (Darwin→metal, /opt/rocm→rocm, else→cuda)
- `README.md` — documents build/run for all platforms
- **Self-contained**: inline any needed utility functions; NO imports from shared directories

**Branch:** `ares/m7-tier2d`
**Cycles budget:** 5

### M8 — Tier 2 Wrap-up + Tier 3 Start (planned)
Remaining Tier 2 high-priority benchmarks (Rodinia CFD, BackProp, etc.) + begin Tier 3 library benchmarks (cuBLAS, CUTLASS wrappers).

### M9 — Tier 3 Library Benchmarks (planned)
cuBLAS GEMM (FP32/FP16/FP64), CUTLASS GEMM, cuFFT (1D/2D), cuSPARSE SpMV benchmark wrappers.

### M10 — Tier 4 & 5: Triton + ML Workloads (planned)
Triton-based benchmarks and representative ML training/inference workloads.

### M11 — Integration & Harness (planned)
Unified benchmark runner: make-based run-all, result collection, reporting infrastructure.

---

## Lessons Learned
- Research phase produced excellent coverage (60+ benchmarks). One cycle sufficient.
- Human constraints (Make, multi-GPU, self-contained) established after M2 — apply to all future work proactively.
- Each benchmark is fully independent from the start — don't build shared infrastructure.
- mgpusim-dev repo has mature CUDA+HIP implementations for Tier 2; use as reference, adapt to our format (inline common headers, add Metal, unify Makefile).
- Parallel worker assignment (1 benchmark per worker) works well — 6 benchmarks finished in ~1 cycle.
- M4 took 3 implementation cycles (some workers timed out on first try, Ares completed remainder directly).
- mgpusim-dev has mature CUDA+HIP refs for all Tier 2 benchmarks — always use them as starting points.
- bench_common_hip.h must be inlined into each .hip file (no shared dependencies).

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
| 12–14 | Implementation | Ares+workers | Implemented 6 Tier 2 benchmarks (SHOC core + rodinia_hotspot) |
| 15 | Verification | Apollo | Verified M4, merged PR #3 |
| 16 | Planning | Athena | Evaluated state, defined M5 (Rodinia BFS/NW/SRAD + PolyBench GEMM/2DConv/FDTD2D) |
| 17–23 | Implementation+Verification | Ares+Apollo | M5 implemented and verified; merged PR #4 |
| 24 | Planning | Athena | Evaluated state, defined M6 (SHOC Sort + Parboil 3 + PolyBench 2) |
| 25–26 | Implementation+Verification | Ares+Apollo | M6 implemented and verified; merged PR #5 |
| 27 | Planning | Athena | Evaluated state, defined M7 (Rodinia Gaussian/LUD/PathFinder/K-Means + PolyBench BICG/MVT) |
