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

### M10 — Tier 4: Triton Python Kernels + BabelStream ✅ COMPLETE
**Goal:** Implement 6 Tier 4 alternative-language benchmarks. Triton benchmarks use Python DSL targeting NVIDIA/AMD GPUs; Apple uses equivalent PyTorch MPS operations. BabelStream is a STREAM-style C++ bandwidth benchmark with CUDA/HIP/Metal backends.
**Deliverables (completed):** triton_vector_add, triton_gemm, triton_softmax, triton_layer_norm, triton_dropout (all Triton/PyTorch MPS) + babelstream (CUDA/HIP/Metal C++). Merged via PR #9.

---

### M11 — Tier 5: ML Workloads (current)
**Goal:** Implement 6 Tier 5 ML workload benchmarks using Python + PyTorch. All benchmarks use synthetic data (no dataset downloads). Platform: `cuda` for NVIDIA/AMD, `mps` for Apple Silicon.

**Deliverables:**
- `tier5/resnet50_train/` — ResNet-50 training step (forward+backward), synthetic ImageNet-like input, torchvision. Reports images/sec.
- `tier5/resnet50_infer/` — ResNet-50 inference only (model.eval(), torch.no_grad()), synthetic input, torchvision. Reports images/sec.
- `tier5/bert_inference/` — BERT-base-uncased inference (HuggingFace transformers), synthetic token IDs, seq_len=128, batch=16. Reports sequences/sec.
- `tier5/nanogpt_train/` — GPT-2 style transformer training step (pure PyTorch, model defined inline — no external model deps). Reports tokens/sec.
- `tier5/vit_inference/` — ViT-B/16 inference (torchvision.models.vit_b_16), synthetic ImageNet input. Reports images/sec.
- `tier5/sdpa_bench/` — Scaled Dot Product Attention benchmark (pure PyTorch F.scaled_dot_product_attention). Reports TFLOPS.

**Per-benchmark file structure:**
- `bench.py` — CUDA/ROCm: uses `torch.device('cuda')`, 5 warmup + 10 timed iterations, timing via `torch.cuda.Event`, CSV output
- `bench_mps.py` — Apple: uses `torch.device('mps')`, same logic, timing via `time.perf_counter` + `.item()` for sync, CSV output
- `Makefile` — Same auto-detect pattern (Darwin→metal→`python3 bench_mps.py`, else→`python3 bench.py`); `setup` target installs deps
- `README.md`

Plus:
- `tier5/Makefile` — wraps all 6 sub-benchmarks (same pattern as tier4/Makefile)
- Root `Makefile` update to include `tier5`

**Technical specs per benchmark:**
- **resnet50_train**: batch=32, input shape [32,3,224,224], random labels (1000 classes), SGD optimizer, 5 warmup + 10 timed forward+backward passes. Metric: `images/sec = batch_size * 10 / total_time_s`
- **resnet50_infer**: batch=64, model.eval(), no_grad, [64,3,224,224] random input, 5 warmup + 10 timed passes. Metric: `images/sec`
- **bert_inference**: batch=16, seq_len=128, random input_ids in [0,30522), model.eval(), no_grad, 5 warmup + 10 timed passes. Metric: `sequences/sec`
- **nanogpt_train**: GPT-2 config: n_layer=6, n_head=6, n_embd=384, block_size=256, vocab_size=50257; batch=4, random token ids; AdamW optimizer; 5 warmup + 10 timed forward+backward. Metric: `tokens/sec = batch * block_size * 10 / total_time_s`
- **vit_inference**: `torchvision.models.vit_b_16(weights=None)`, batch=32, [32,3,224,224] random input, model.eval(), no_grad, 5 warmup + 10 timed. Metric: `images/sec`
- **sdpa_bench**: batch=8, n_heads=16, seq_len=1024, head_dim=64; query/key/value = random float16 tensors; 5 warmup + 50 timed calls; `F.scaled_dot_product_attention(q,k,v)`. Metric: `TFLOPS = 2*B*H*S*S*D / time_per_iter_s / 1e12`

**CSV output format:** `benchmark_name,size,time_ms,metric_value`
Examples:
- `resnet50_train,32,15.234,2099.45` (images/sec)
- `bert_inference,16,8.123,1971.23` (sequences/sec)
- `sdpa_bench,8*16*1024*64,0.312,45.67` (TFLOPS)

**Makefile template (same as tier4):**
```makefile
ifndef PLATFORM
  ifeq ($(shell uname -s),Darwin)
    PLATFORM := metal
  else
    PLATFORM := cuda
  endif
endif
.PHONY: all run setup clean
ifeq ($(PLATFORM),metal)
run:
	python3 bench_mps.py
setup:
	pip3 install torch torchvision transformers
else
run:
	python3 bench.py
setup:
	pip3 install torch torchvision transformers
endif
all: run
clean:
	find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
```

**Branch:** `ares/m11-tier5`
**Cycles budget:** 6

### M12 — Integration & Harness (current)
**Goal:** Build a functional benchmark runner so the 54-benchmark suite can be driven from a single command and results collected automatically.

**Deliverables:**
1. `.gitignore` — ignore compiled binaries, `__pycache__`, `results/`, `*.pyc`, `.DS_Store`
2. `scripts/run_all.sh` — iterates all `tier*/*/Makefile` directories, runs `make run` with 5-minute per-benchmark timeout, captures stdout (CSV) to `results/<timestamp>/<tier>_<benchmark>.csv`; prints progress and errors to terminal
3. `scripts/report.py` — reads all `.csv` files from a `results/<timestamp>/` directory, generates `results/<timestamp>/report.md` with a summary table: tier, benchmark, time_ms, metric columns; handles both 4-field and 7-field CSV formats gracefully
4. Root `Makefile` — add `run-all` target that calls `scripts/run_all.sh`; add `report` target that calls `scripts/report.py results/<latest>/`
5. Fix Tier 1 CSV output — all 6 Tier 1 benchmarks currently output only human-readable text; add a single CSV line to stdout: `benchmark_name,size,time_ms,metric` (matching the 4-field format used by Tier 3/4/5). Keep existing human-readable output, add CSV on a separate line.

**Acceptance criteria:**
- `make run-all` runs without error on a CUDA machine (individual benchmarks may fail due to missing hardware, but the script should not crash)
- `python3 scripts/report.py results/<timestamp>/` generates a readable markdown report
- All 6 Tier 1 benchmarks emit at least one CSV line to stdout when run
- `.gitignore` is committed and suppresses compiled binaries + Python cache

**Branch:** `ares/m12-harness`
**Cycles budget:** 5

### M13 — Critical Benchmark Gaps (planned)
Add the highest-priority missing benchmarks based on Iris's evaluation:
- Tensor Core Throughput (Tier 1)
- Mixbench or Shared Memory Bandwidth (Tier 1)
- HGEMM FP16 / CUTLASS GEMM (Tier 3)
- Triton Flash Attention (Tier 4)
**Cycles budget:** 6

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
- Tier 5 ML benchmarks: use synthetic data (random tensors) to avoid dataset downloads. PyTorch `cuda` device works for both NVIDIA and AMD/ROCm.
- Tier 5 deps: `pip install torch torchvision transformers` in `setup` target.

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
| 43–48 | Implementation+Verification | Ares+Apollo | M10 implemented (5 Triton + BabelStream); merged PR #9 |
| 49 | Planning | Athena | Evaluated state, defined M11 (Tier 5: ML workloads) |
| 50–55 | Implementation+Verification | Ares+Apollo | M11 implemented (6 Tier 5 ML workloads); merged PR #10 |
| 56 | Planning | Athena | Iris evaluated state (54 benchmarks, 3 CSV formats, no harness); defined M12 (Integration & Harness) |
