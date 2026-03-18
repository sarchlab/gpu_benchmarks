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

### M9 — Tier 3: GPU Library Benchmarks (cuBLAS/rocBLAS SGEMM/DGEMM/GEMV + cuFFT/hipFFT 1D/2D + cuSPARSE/rocSPARSE SpMV) ✅ COMPLETE
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

### M10 — Tier 4: Triton Python Kernels + BabelStream (current)
**Goal:** Implement 6 Tier 4 alternative-language benchmarks. Triton benchmarks use Python DSL targeting NVIDIA/AMD GPUs; Apple uses equivalent PyTorch MPS operations. BabelStream is a STREAM-style C++ bandwidth benchmark with CUDA/HIP/Metal backends.

**Deliverables:**
- `tier4/triton_vector_add/` — Simplest Triton kernel: element-wise vector add. NVIDIA/AMD: Triton JIT. Apple: PyTorch MPS.
- `tier4/triton_gemm/` — Matrix multiplication in Triton with auto-tuning tile sizes. Apple: PyTorch MPS matmul.
- `tier4/triton_softmax/` — Fused online softmax in Triton. Apple: PyTorch MPS softmax.
- `tier4/triton_layer_norm/` — Fused layer normalization in Triton. Apple: PyTorch MPS layer_norm.
- `tier4/triton_dropout/` — Element-wise dropout kernel in Triton (uses `tl.rand`). Apple: PyTorch MPS dropout.
- `tier4/babelstream/` — STREAM benchmark (copy/scale/add/triad/dot) in CUDA/HIP/Metal. C++ implementation.

**Per-benchmark structure (Triton benchmarks):**
- `bench.py` — Python Triton kernel (NVIDIA/AMD). Imports: triton, torch. Output: CSV `benchmark_name,size,time_ms,GBs` or `benchmark_name,size,time_ms,TFLOPS`
- `bench_mps.py` — Apple-specific: PyTorch MPS equivalent (same computation). Same CSV output format.
- `Makefile` — Platform auto-detect; CUDA/ROCm → `python3 bench.py`; metal → `python3 bench_mps.py`. Includes `setup` target for `pip3 install`.
- `README.md` — Usage, requirements, output format.

**BabelStream structure:**
- `babelstream.hip` — C++ with `#ifdef USE_CUDA` (nvcc) / `#elif defined(USE_HIP)` (hipcc). Implements 5 STREAM kernels on arrays of N=32M floats. 5 warmup + 5 timed iterations. Reports GB/s per kernel.
- `babelstream_metal.mm` — Apple Metal compute shader for STREAM kernels. Same output.
- `Makefile` — Same platform auto-detect pattern as Tier 3.
- `README.md`

**Triton bench.py structure:**
```python
import torch, triton, triton.language as tl, time, sys

@triton.jit
def kernel(...):
    ...

def benchmark(n=...):
    # allocate tensors on cuda/hip device
    # warmup 5 iterations
    # timed 5 iterations using torch.cuda.Event
    # print: "benchmark_name,size,time_ms,metric"

if __name__ == "__main__":
    benchmark()
```

**Triton bench_mps.py structure:**
```python
import torch, time

def benchmark(n=...):
    device = torch.device("mps")
    # allocate tensors on MPS device
    # warmup 5 iterations  
    # timed 5 iterations using time.perf_counter (synchronize with .item())
    # print same CSV format

if __name__ == "__main__":
    benchmark()
```

**Tier 4 Makefile template (Triton benchmarks):**
```makefile
ifndef PLATFORM
  ifeq ($(shell uname -s),Darwin)
    PLATFORM := metal
  else ifneq ($(wildcard /opt/rocm),)
    PLATFORM := rocm
  else
    PLATFORM := cuda
  endif
endif
.PHONY: all run setup clean
ifeq ($(PLATFORM),metal)
all: run
run:
	python3 bench_mps.py
setup:
	pip3 install --quiet torch
else
all: run
run:
	python3 bench.py
setup:
	pip3 install --quiet torch triton
endif
clean:
	rm -rf __pycache__ *.pyc
```

**Key technical details:**
- Triton vector_add: N=2^26 elements (64M floats = 256MB). Measure GB/s.
- Triton gemm: M=N=K=4096. Measure TFLOPS. Use tile size BLOCK_M=128, BLOCK_N=256, BLOCK_K=64.
- Triton softmax: N_ROWS=4096, N_COLS=4096. Measure GB/s.
- Triton layer_norm: N_ROWS=4096, N_COLS=4096, eps=1e-5. Measure GB/s.
- Triton dropout: N=2^26 elements, p=0.5. Measure GB/s.
- BabelStream: N=32*1024*1024 floats per array. Measure GB/s for copy, scale, add, triad, dot.

**Branch:** `ares/m10-tier4`
**Cycles budget:** 6

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
- Tier 4 introduces Python-based benchmarks (Triton): Makefile wraps `python3 bench.py`. Apple uses PyTorch MPS fallback.
- Triton does not support Apple Metal — use equivalent PyTorch MPS ops in `bench_mps.py`.
- BabelStream follows the same C++ pattern as Tier 3 (`.hip` + `_metal.mm`).
- For Python benchmark timing, use `torch.cuda.Event` for CUDA/ROCm, `time.perf_counter` for MPS.

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
| 37–41 | Implementation+Verification | Ares+Apollo | M9 implemented (6 Tier 3 benchmarks) and verified; merged PR #8 |
| 42 | Planning | Athena | Evaluated state, defined M10 (Tier 4: Triton Python kernels + BabelStream) |
