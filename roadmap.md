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

### M7 — Tier 2: Rodinia Gaussian/LUD/PathFinder/K-Means + PolyBench BICG/MVT ✅ COMPLETE (cycles used: ~3)
**Goal:** Implement 6 more Tier 2 benchmarks — 4 Rodinia + 2 PolyBench — following the same project conventions.
**Deliverables (completed):** rodinia_gaussian, rodinia_lud, rodinia_pathfinder, rodinia_kmeans, polybench_bicg, polybench_mvt — all with HIP + Metal + Makefile + README. Merged via PR #6.

---

### M8 — Tier 2 Continued: SHOC SpMV/Stencil2D + PolyBench GRAMSCHMIDT/JACOBI2D + Rodinia BackProp + Parboil Histogram ✅ COMPLETE (cycles used: ~4)
**Goal:** Implement 6 more Tier 2 benchmarks, covering remaining high-priority SHOC, PolyBench, Rodinia, and Parboil kernels.
**Deliverables (completed):** shoc_spmv, shoc_stencil2d, polybench_gramschmidt, polybench_jacobi2d, rodinia_backprop, parboil_histogram — all with HIP + Metal + Makefile + README. Merged via PR #7.

---

### M9 — Tier 3: GPU Library Benchmarks (cuBLAS/rocBLAS SGEMM/DGEMM/GEMV + cuFFT/hipFFT 1D/2D + cuSPARSE/rocSPARSE SpMV) (current)
**Goal:** Implement 6 Tier 3 library benchmarks that wrap vendor GPU math libraries. Each benchmark tests a different library category (BLAS, FFT, Sparse).

**Deliverables:**
- `tier3/cublas_sgemm/` — FP32 GEMM: cuBLAS (NVIDIA) / rocBLAS (AMD) / MPSMatrixMultiplication (Apple)
- `tier3/cublas_dgemm/` — FP64 GEMM: cuBLAS (NVIDIA) / rocBLAS (AMD) / cblas_dgemm Accelerate (Apple CPU — Metal has no FP64)
- `tier3/cublas_gemv/` — FP32 GEMV: cuBLAS (NVIDIA) / rocBLAS (AMD) / MPSMatrixVectorMultiplication (Apple)
- `tier3/cufft_1d/` — 1D complex FFT: cuFFT (NVIDIA) / hipFFT (AMD) / vDSP Accelerate (Apple)
- `tier3/cufft_2d/` — 2D complex FFT: cuFFT (NVIDIA) / hipFFT (AMD) / vDSP Accelerate (Apple)
- `tier3/cusparse_spmv/` — CSR SpMV: cuSPARSE (NVIDIA) / rocSPARSE (AMD) / custom Metal shader (Apple)

**Per-benchmark file structure:**
- `<name>.hip` — HIP source with `#ifdef USE_CUBLAS / USE_ROCBLAS / USE_HIPFFT / USE_CUFFT / USE_CUSPARSE / USE_ROCSPARSE` for platform-specific library calls; compiles with `nvcc -x cu -DUSE_CUBLAS` (CUDA) or `hipcc -DUSE_ROCBLAS` (ROCm)
- `<name>_metal.mm` — Apple ObjC++ host using MPS/Accelerate/custom Metal
- `Makefile` — platform auto-detect (Darwin→metal, /opt/rocm→rocm, else→cuda); add correct `-D` flags and `-l` link flags per platform
- `README.md` — build/run for all 3 platforms, output format, metric formulas

**Key technical details per benchmark:**
- SGEMM/DGEMM/GEMV: matrix size N=4096 (MxN=4096×4096); verify first 64 elements vs CPU; report GFLOPS + GB/s
- FFT: N=1<<22 points (4M) for 1D; N=2048×2048 for 2D; verify output vs reference; report GB/s + GFLOPS
- SpMV: reuse shoc_spmv parameters (NUM_ROWS=16384, NNZ_PER_ROW=32); verify vs CPU; report GB/s + GFLOPS
- Run 5 warmup + 5 timed iterations; report average; CSV output: `benchmark_name,size,time_ms,GBs,GFLOPS`

**Self-contained**: each benchmark has all code inline, no shared dependencies.

**Branch:** `ares/m9-tier3`
**Cycles budget:** 5

### M10 — Tier 4: Triton + BabelStream (planned)
Triton-based benchmarks (GEMM, Softmax, LayerNorm, Flash Attention) and BabelStream cross-platform bandwidth.

### M11 — Tier 5: ML Workloads (planned)
Representative ML training/inference workloads (ResNet-50, nanoGPT, BERT fine-tuning).

### M12 — Integration & Harness (planned)
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
| 28–30 | Implementation+Verification | Ares+Apollo | M7 implemented and verified; merged PR #6 |
| 31 | Planning | Athena | Evaluated state, defined M8 (SHOC SpMV/Stencil2D + PolyBench GRAMSCHMIDT/JACOBI2D + Rodinia BackProp + Parboil Histogram) |
| 32–35 | Implementation+Verification | Ares+Apollo | M8 implemented and verified; merged PR #7 |
| 36 | Planning | Athena | Evaluated state, defined M9 (Tier 3: cuBLAS/rocBLAS SGEMM/DGEMM/GEMV + cuFFT/hipFFT 1D/2D + cuSPARSE/rocSPARSE SpMV) |
