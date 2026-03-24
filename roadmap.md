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
4. **Complete benchmarks** — if we skip some workloads, we need a clear reason (Issue #142)
5. **Parameterized** — every benchmark has a scaling parameter (20-30 values) and non-scaling parameters (3-5 values each) (Issue #143)
6. **CLI tool** — compile, run with repetitions, SQLite results with platform info + kernel-level timing (Issue #144)

## Success Criteria
Deliver a comprehensive, well-organized benchmark suite that researchers and engineers can use to evaluate GPGPU hardware and software stacks. Every benchmark must be parameterized, and a CLI tool must support automated compilation, execution with configurable repetitions, and result storage in SQLite with kernel-level timing.

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

### M12 — Integration & Harness ✅ COMPLETE (cycles used: 3)
**Goal:** Build a functional benchmark runner so the 54-benchmark suite can be driven from a single command and results collected automatically.

**Deliverables (completed):**
1. `.gitignore` — ignores compiled binaries, `__pycache__`, `results/`, `*.pyc`, `.DS_Store`
2. `scripts/run_all.sh` — iterates all `tier*/*/Makefile` directories, runs `make run` with 5-minute timeout, captures CSV to `results/<timestamp>/<tier>_<benchmark>.csv`
3. `scripts/report.py` — reads CSV files, handles 4-field and 7-field formats, generates `report.md` with summary table
4. Root `Makefile` — `run-all` and `report` targets added
5. All 6 Tier 1 benchmarks — CSV output added to .hip and _metal.mm files

**Branch:** `ares/m12-harness` (merged to main)

### M13 — Critical Benchmark Gaps ✅ COMPLETE (cycles used: ~3)
**Goal:** Fill the 6 highest-priority gaps identified by Iris's evaluation. These are all HIGH-priority items from `benchmark_report.md` that are missing from the suite.

**Deliverables:**

**Tier 1 (3 new microbenchmarks):**
- `tier1/tensor_core_throughput/` — Tensor Core TOPS measurement using WMMA API (HIP). FP16 and TF32 data types. Metal: simdgroup_matrix_multiply_accumulate. Reports TOPS.
  - `.hip` file with `#ifdef` for CUDA WMMA vs HIP WMMA
  - `_metal.mm` + `.metal` using simdgroup matrix ops
  - CSV: `tensor_core_throughput,<dtype>,<time_ms>,<TOPS>`
- `tier1/mixbench/` — Parametric roofline benchmark varying compute-to-memory ratio. Multiple data points from pure memory-bound to pure compute-bound. Reference: ekondis/mixbench on GitHub.
  - `.hip` kernel with configurable arithmetic intensity (FLOPs per byte)
  - `_metal.mm` + `.metal` equivalent
  - CSV: `mixbench,<flops_per_byte>,<time_ms>,<achieved_gflops>` (one line per intensity level)
- `tier1/shared_mem_bandwidth/` — Shared memory throughput with bank-conflict-free and conflicted access patterns.
  - `.hip` kernel with stride-1 (no conflict) and stride-32 (max conflict) patterns
  - `_metal.mm` + `.metal` using threadgroup memory
  - CSV: `shared_mem_bw,<pattern>,<time_ms>,<GBs>`

**Tier 3 (2 new library benchmarks):**
- `tier3/cublas_hgemm/` — FP16 GEMM: cublasHgemm (NVIDIA) / rocblas_hgemm (AMD) / MPSMatrixMultiplication float16 (Apple). M=N=K=4096. Leverages Tensor Cores when available.
  - `.hip` file with `#ifdef USE_CUBLAS / USE_ROCBLAS`
  - `_metal.mm` using MPS float16 matrices
  - CSV: `cublas_hgemm,<M>,<time_ms>,<TFLOPS>`
- `tier3/cublas_batched_gemm/` — Batched FP32 GEMM: cublasGemmBatched / rocblas_gemm_batched / loop-of-MPS. batch_count=64, M=N=K=512.
  - `.hip` file with `#ifdef USE_CUBLAS / USE_ROCBLAS`
  - `_metal.mm` using MPS with loop over batch
  - CSV: `cublas_batched_gemm,<batch>x<M>,<time_ms>,<GFLOPS>`

**Tier 4 (1 new alternative-language benchmark):**
- `tier4/triton_flash_attention/` — Flash Attention kernel in Triton (CUDA/ROCm). Apple fallback: PyTorch F.scaled_dot_product_attention on MPS.
  - `bench.py` with Triton flash attention kernel (reference: Triton tutorials)
  - `bench_mps.py` with PyTorch SDPA on MPS
  - CSV: `triton_flash_attention,<B>x<H>x<S>x<D>,<time_ms>,<TFLOPS>`

**Per-benchmark conventions (same as existing):**
- Makefile with PLATFORM auto-detect (Darwin→metal, /opt/rocm→rocm, else→cuda)
- README.md with algorithm, build/run for all 3 platforms, output format
- 5 warmup + 5 timed iterations, report average
- CSV to stdout, human-readable to stderr
- Self-contained, no shared dependencies
- Update tier1/Makefile, tier3/Makefile, tier4/Makefile to include new subdirectories

**Branch:** `ares/m13-gaps` (merged to main via PR #11)
**Cycles budget:** 5

### M14 — DNN Library Benchmarks + More Coverage ✅ COMPLETE (cycles used: ~2)
**Goal:** Add 6 more benchmarks covering critical DNN library primitives (cuDNN/MIOpen/MPS convolution & batch normalization), classic GPU kernel patterns (FFT, transpose), and modern ML inference workloads (LLM generation, torch.compile).

**Deliverables:**

**Tier 2 (2 new application benchmarks):**
- `tier2/shoc_fft/` — Custom Cooley-Tukey radix-2 FFT kernel. Tests butterfly operations and bit-reversal permutation. N=1M complex elements. HIP + Metal.
  - `.hip` file: radix-2 FFT with shared memory butterfly stages
  - `_metal.mm` + `.metal`: equivalent Metal compute shader
  - CSV: `shoc_fft,<N>,<time_ms>,<GFLOPS>`
  - Report GFLOPS = 5*N*log2(N) / time_s / 1e9

- `tier2/cuda_transpose/` — Optimized matrix transpose with shared memory tiling and bank-conflict avoidance. N=4096×4096 float matrix.
  - `.hip` file: naive transpose + shared-memory tiled transpose (TILE=32, +1 padding for bank conflicts)
  - `_metal.mm` + `.metal`: equivalent Metal compute shader
  - CSV: `cuda_transpose,<N>,<time_ms>,<GBs>` (one line for naive, one for optimized)
  - Report effective bandwidth = 2*N*N*sizeof(float) / time_s / 1e9

**Tier 3 (2 new library benchmarks):**
- `tier3/cudnn_conv/` — Forward 2D convolution using DNN libraries: cuDNN (NVIDIA) / MIOpen (AMD) / MPSCNNConvolution (Apple).
  - `.hip` file with `#ifdef USE_CUDNN / USE_MIOPEN` for platform-specific library calls
  - `_metal.mm` using MPSCNNConvolution
  - Input: N=32, C=3, H=W=224; Filter: K=64, C=3, R=S=7, stride=2, pad=3
  - Algorithm: auto-tune (cuDNN: CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM or auto; MIOpen: find best)
  - CSV: `cudnn_conv,<NxCxHxW>,<time_ms>,<GFLOPS>`
  - GFLOPS = 2*N*K*OH*OW*C*R*S / time_s / 1e9
  - Link: `-lcudnn` (CUDA), `-lMIOpen` (ROCm), Metal+MPS frameworks (Apple)
  - Makefile: `PLATFORM=cuda → nvcc -x cu -DUSE_CUDNN -lcudnn`, `PLATFORM=rocm → hipcc -DUSE_MIOPEN -lMIOpen`

- `tier3/cudnn_batchnorm/` — Batch normalization forward + backward: cuDNN / MIOpen / MPSCNNBatchNormalization (Apple).
  - `.hip` file with `#ifdef USE_CUDNN / USE_MIOPEN`
  - `_metal.mm` using MPSCNNBatchNormalization
  - Input: N=64, C=256, H=W=56 (typical ResNet intermediate)
  - CSV: `cudnn_batchnorm,<NxCxHxW>,<time_ms>,<GBs>`
  - Forward + backward timing
  - Link: same as cudnn_conv

**Tier 5 (2 new ML benchmarks):**
- `tier5/gpt2_generate/` — GPT-2 autoregressive text generation (inference). Uses same inline GPT-2 model as nanogpt_train (n_layer=6, n_head=6, n_embd=384, vocab_size=50257). Generates 128 tokens autoregressively from a random prompt of 32 tokens.
  - `bench.py` (CUDA/ROCm): torch.cuda.Event timing, KV-cache optional
  - `bench_mps.py` (Apple): time.perf_counter timing
  - CSV: `gpt2_generate,<seq_len>,<time_ms>,<tokens_per_sec>`
  - 5 warmup + 10 timed generation runs

- `tier5/torch_compile_bench/` — Compare PyTorch eager mode vs torch.compile on ResNet-50 inference. Tests torch.compile (Inductor backend) speedup.
  - `bench.py` (CUDA/ROCm): runs ResNet-50 inference in eager mode and compiled mode, reports both timings and speedup
  - `bench_mps.py` (Apple): same but uses MPS backend (torch.compile may fallback to eager on MPS)
  - CSV: `torch_compile_bench,eager,<time_ms>,<images_per_sec>` and `torch_compile_bench,compiled,<time_ms>,<images_per_sec>`
  - batch=32, input [32,3,224,224], 5 warmup + 10 timed, report speedup ratio

**Per-benchmark conventions (same as existing):**
- Makefile with PLATFORM auto-detect
- README.md with build/run instructions for all 3 platforms
- 5 warmup + 5 timed iterations, report average
- CSV to stdout, human-readable to stderr
- Self-contained, no shared dependencies
- Update tier2/Makefile, tier3/Makefile, tier5/Makefile to include new subdirectories

**Branch:** `ares/m14-benchmarks` (merged to main via PR #12)
**Cycles budget:** 5

### M15 — Final Polish: README + Coverage Gaps ✅ COMPLETE (cycles used: ~3)
**Goal:** Update README.md to comprehensively list all 66 benchmarks, and add 4 remaining high-value microbenchmarks/library benchmarks to reach ~70 total benchmarks.

**Deliverables:**

**Documentation (1 update):**
- **README.md** — Complete rewrite to list all 66+ benchmarks organized by tier with descriptions, metrics, and build/run instructions. Update directory layout to reflect current state. Add benchmark harness usage (`make run-all`, `make report`).

**Tier 1 (3 new microbenchmarks):**
- `tier1/atomic_operations/` — Atomic throughput: `atomicAdd`, `atomicCAS`, `atomicMin` on global and shared memory. Measures throughput (ops/sec) and contention effects across varying thread counts. HIP + Metal.
  - `.hip` file: test atomicAdd/CAS/Min on global/shared with 1024 to 1M threads
  - `_metal.mm` + `.metal`: Metal atomic_fetch_add_explicit / atomic_compare_exchange
  - CSV: `atomic_ops,<operation>_<memory>,<time_ms>,<Gops>`
  
- `tier1/warp_shuffle/` — Warp shuffle (butterfly) throughput: `__shfl_xor_sync`, `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`. Measures warp-level communication bandwidth.
  - `.hip` file: repeated shuffle operations within warp
  - `_metal.mm` + `.metal`: simd_shuffle / simd_shuffle_xor
  - CSV: `warp_shuffle,<variant>,<time_ms>,<GBs>`

- `tier1/int32_throughput/` — Integer compute throughput: INT32 multiply-add kernels maximizing integer ALU utilization.
  - `.hip` file: MAD-heavy INT32 kernel
  - `_metal.mm` + `.metal`: equivalent
  - CSV: `int32_throughput,<N>,<time_ms>,<GIOPS>`

**Tier 3 (1 new library benchmark):**
- `tier3/cutlass_gemm/` — CUTLASS-style tiled GEMM implementation. Custom tiled FP32 GEMM with shared-memory staging, register tiling, and double-buffering. NOT using CUTLASS library — implements the technique inline. M=N=K=4096. Reports GFLOPS.
  - `.hip` file: custom tiled GEMM kernel (BLOCK_M=128, BLOCK_N=128, BLOCK_K=8, thread tile 8x8)
  - `_metal.mm` + `.metal`: equivalent Metal compute shader with threadgroup memory tiling
  - CSV: `cutlass_gemm,<M>,<time_ms>,<GFLOPS>`

**Per-benchmark conventions (same as existing):**
- Makefile with PLATFORM auto-detect
- README.md with build/run instructions for all 3 platforms
- 5 warmup + 5 timed iterations, report average
- CSV to stdout, human-readable to stderr
- Self-contained, no shared dependencies
- Update tier1/Makefile, tier3/Makefile for new subdirectories

**Branch:** `ares/m15-polish`
**Cycles budget:** 4

### M16 — Tier 5: LLM Inference Frameworks + Popular Model Inference ✅ COMPLETE (cycles used: ~3)
**Goal:** Add 6 LLM inference benchmarks covering the major inference frameworks and popular open-source models.
**Deliverables (completed):** vllm_offline, llamacpp_bench, ollama_bench, llama_inference, deepseek_inference, qwen_inference — all with bench.py + bench_mps.py + Makefile + README. Merged via PR #14.

### M17 — Tier 5: More LLM Models + Training Workloads + Diffusion ✅ COMPLETE (cycles used: ~3)
**Goal:** Add 6 more ML benchmarks: GLM (ChatGLM), Phi/Mistral model inference, BERT fine-tuning, Whisper inference, Stable Diffusion inference. Covers remaining popular models from Issue #100 and adds diffusion.
**Deliverables (completed):** glm_inference, phi_inference, mistral_inference, bert_finetune, whisper_inference, stable_diffusion_bench — all with bench.py + bench_mps.py + Makefile + README. Merged via PR #15.

### M18 — Tier 5: Advanced ML Patterns (MoE, RAG, CoT, Multi-Agent, SDXL, GPT-NeoX) ✅ COMPLETE (cycles used: ~3)
**Goal:** Add 6 benchmarks covering advanced ML patterns: Mixture-of-Experts inference, RAG pipeline, Chain-of-Thought reasoning, multi-agent execution, SDXL diffusion, GPT-NeoX training.
**Deliverables (completed):** moe_inference, rag_pipeline, cot_reasoning, multi_agent_bench, sdxl_inference, gptneox_train — all with bench.py + bench_mps.py + Makefile + README. Merged via PR #16.
**Note:** Issue #100 (modern ML workloads) now fully covered with 26 Tier 5 benchmarks. Issue closed.

### M19 — Tier 2: New Benchmark Suites Batch 1 (4 new suites, 10 benchmarks) ✅ COMPLETE (cycles used: ~4)
**Goal:** Add 10 benchmarks from 4 new Tier 2 suites. This addresses Issue #101 (need 8+ more suites beyond SHOC/Rodinia/PolyBench/Parboil/CUDA Samples).

**New suites in this batch:**
1. **HeteroMark** (3): heteromark_aes, heteromark_fir, heteromark_pagerank
2. **CUDA SDK Extended** (3): cuda_nbody, cuda_convolution_separable, cuda_scan_large
3. **Tango** (2): tango_blackscholes, tango_binomial_options
4. **NPB-GPU** (2): npb_ep, npb_is

**Deliverables (completed):** All 10 benchmarks with HIP + Metal + Makefile + README. Merged via PR #17.

### M20 — Tier 2: New Benchmark Suites Batch 2 (4 new suites, 9 benchmarks) ✅ COMPLETE (cycles used: ~4)
**Goal:** Add benchmarks from 4 more Tier 2 suites to reach 8+ new suites total.

**New suites in this batch:**
5. **Altis** (3): altis_cfd, altis_particlefilter, altis_raytracing
6. **Lonestar GPU** (2): lonestar_sssp, lonestar_bh
7. **Graph Analytics** (2): graph_pr, graph_tc
8. **Chai** (2): chai_hsto, chai_cedd

**Deliverables (completed):** All 9 benchmarks with HIP + Metal + Makefile + README. Merged via PR #18.
**Note:** Issue #101 (8+ more Tier 2 suites) now fully covered. Issue closed.

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
| 57–59 | Implementation+Verification | Ares+Apollo | M12 implemented (harness scripts, .gitignore, Tier 1 CSV); merged to main |
| 60 | Planning | Athena | Evaluated state (54 benchmarks, all tiers covered, harness complete); defined M13 (6 critical benchmark gaps) |
| 61–63 | Implementation+Verification | Ares+Apollo | M13 implemented (6 gap-fill benchmarks); merged PR #11 |
| 64 | Planning | Athena | Evaluated state (60 benchmarks); defined M14 (DNN library + FFT/Transpose + ML inference) |
| 65 | Implementation+Verification | Ares+Apollo | M14 implemented (6 benchmarks: shoc_fft, cuda_transpose, cudnn_conv, cudnn_batchnorm, gpt2_generate, torch_compile_bench); merged PR #12 |
| 66 | Planning | Athena | Evaluated state (66 benchmarks); defined M15 (README update + 4 more benchmarks) |
| 67–69 | Implementation+Verification | Ares+Apollo | M15 implemented (README rewrite + 4 new benchmarks: atomic_operations, warp_shuffle, int32_throughput, cutlass_gemm); merged PR #13 |
| 70 | Planning | Athena | All milestones complete. 70 benchmarks across 5 tiers. Project complete. |
| 71 | Planning | Athena | New human requests: Issue #100 (modern ML workloads) + Issue #101 (8+ more Tier 2 suites). Reopen project. |
| 72–74 | Implementation+Verification | Ares+Apollo | M16 implemented (6 LLM benchmarks: vllm_offline, llamacpp_bench, ollama_bench, llama_inference, deepseek_inference, qwen_inference); merged PR #14 |
| 75 | Planning | Athena | Evaluated state (76 benchmarks). M16 complete. Define M17. |
| 76–78 | Implementation+Verification | Ares+Apollo | M17 implemented (6 benchmarks: glm_inference, phi_inference, mistral_inference, bert_finetune, whisper_inference, stable_diffusion_bench); merged PR #15 |
| 79 | Planning | Athena | Evaluated state (82 benchmarks). M17 complete. Define M18. |
| 80–82 | Implementation+Verification | Ares+Apollo | M18 implemented (6 benchmarks: moe_inference, rag_pipeline, cot_reasoning, multi_agent_bench, sdxl_inference, gptneox_train); merged PR #16 |
| 83 | Planning | Athena | Evaluated state (88 benchmarks). M18 complete. Issue #100 closed. Define M19. |
| 84–87 | Implementation+Verification | Ares+Apollo | M19 implemented (10 Tier 2 benchmarks on branch ares/m19-tier2-suites1). Epoch ended before merge. |
| 88 | Planning | Athena | M19 branch exists but not merged. Define combined M19+M20 milestone. |
| 89–91 | Implementation+Verification | Ares+Apollo | M19 merged (PR #17) + M20 implemented and merged (PR #18). 19 new Tier 2 benchmarks from 8 new suites. |
| 92 | Planning | Athena | All milestones complete. 107 benchmarks across 5 tiers. All human requests fulfilled. Project complete. |
| 93 | Planning | Athena | New human requests: Issue #142 (complete benchmarks), #143 (parameterize), #144 (CLI tool). Reopen project. Define M21. |
| 94–103 | Implementation+Verification | Ares+Apollo | M21 implemented: 107 params.json files + 4 design docs. Multiple fix rounds. Merged PR #19. |
| 104 | Planning | Athena | M21 complete. Define M22 (CLI tool + Tier 1 refactoring). |
| 105–117 | Implementation+Verification | Ares+Apollo | M22 implemented: gpubench CLI + 12 Tier 1 benchmarks refactored. Fix rounds for enum mismatches. Merged PR #20. |
| 118 | Planning | Athena | M22 complete. Define M23 (Tier 2 refactoring, 51 benchmarks). |
| 119–125 | Implementation+Verification | Ares+Apollo | M23 implemented (51 Tier 2 benchmarks refactored, 102 files). Fix round for rodinia_lud + dead CSV. Merged PR #21. |
| 126 | Planning | Athena | M23 complete. Define M24 (Tier 3 + Tier 4 refactoring, 18 benchmarks). |
| 127–130 | Implementation+Verification | Ares+Apollo | M24 implemented (18 benchmarks, 36 files refactored). Fix round for triton_vector_add. Merged PR #22. |
| 131 | Planning | Athena | M24 complete. Define M25 (Tier 5 refactoring, 26 benchmarks, 52 files). |
| 132–136 | Implementation+Verification | Ares+Apollo | M25 implemented (26 Tier 5 benchmarks, 52 Python files refactored). Merged PR #23. |
| 137 | Planning | Athena | M25 complete. Define M26 (Integration, Testing & Final Polish). |
| 138–140 | Implementation+Verification | Ares+Apollo | M26 implemented (README rewrite, old harness cleanup, Makefile update, smoke test). Merged PR #24. |
| 141 | Planning | Athena | M26 complete. All milestones done. All human issues closed. **PROJECT COMPLETE.** |
| 142 | Planning | Athena | New human request: Issue #217 (Go CLI). Updated spec, defined M27. |
| 143–145 | Implementation+Verification | Ares+Apollo | M27 implemented: Go CLI at cmd/gpubench/ with 7 source files. PR #25 merged. |
| 146 | Planning | Athena | M27 verified complete. All issues closed. **PROJECT COMPLETE.** |
| 147 | Planning | Athena | New human issue #221 (compilation failures). Defined M28. |
| 148–150 | Implementation+Verification | Ares+Apollo | M28 implemented (CLI error logging + CUDA compat block). Merged PR #26. |
| 151 | Planning | Athena | M28 complete. New human issues #223-#229. Updated spec/roadmap. Defining M29. |
| 152–157 | Implementation+Verification | Ares+Apollo | M29 implemented (75 native .cu files + Makefile updates). Merged PR #27. |
| 158 | Planning | Athena | M29 complete. Issue #224 closed. Defining M30 (complete suites). |
| 159–168 | Implementation+Verification | Ares+Apollo | M30 implemented (13 Tier 2 benchmarks) + fix rounds. Merged PR #28. |
| 169 | Planning | Athena | M30 complete. 120 benchmarks. Defining M31 (MLPerf). |
| 170–177 | Implementation+Verification | Ares+Apollo | M31 implemented (5 MLPerf-inspired Tier 5 benchmarks). Fix rounds for DLRM offsets + UNet3D params. Merged PR #29. |
| 178 | Planning | Athena | M31 complete. 125 benchmarks. Issue #228 closed. Defining M32 (Docker CI + Result Persistence). |
| 179–183 | Implementation+Verification | Ares+Apollo | M32 implemented (Docker CI + Result Persistence + README). PR #31 merged. |
| 184 | Planning | Athena | M32 complete. All 32 milestones done. All 14 human issues closed. **PROJECT COMPLETE.** |
| ... | Various | Various | Phases 8-10: CI fixes (M33-M35.1), repo cleanup (M36), compilation fixes (M37). See detailed milestone entries. |
| 92 prev | Planning | Athena | M37 complete (PR #37 merged, CI green). Human issue #2 (subrepository). Defining M38 (repo cleanup). |
| current | Planning | Athena | M38 complete (PR #38 merged, CI green). All human issues closed. PROJECT COMPLETE. |

---

## Phase 2: Parameterization & CLI (Issues #142, #143, #144)

New human requirements: parameterize all benchmarks, build a CLI tool for compilation/execution, record results in SQLite with kernel-level timing.

### M21 — Parameter Definitions for All 107 Benchmarks + Output Protocol Design ✅ COMPLETE (cycles used: ~10)
**Goal:** Define scaling and non-scaling parameters for every benchmark; design the standardized output protocol for kernel-level timing; design the SQLite schema; design the CLI architecture.

**Deliverables (completed):**
1. `params.json` in each of the 107 benchmark directories
2. `docs/output_protocol.md` — JSON-lines output format
3. `docs/cli_design.md` — CLI specification
4. `docs/schema.sql` — SQLite schema
5. `docs/params_format.md` — params.json format spec

**Branch:** `ares/m21-milestone` (merged to main via PR #19)

### M22 — CLI Tool Implementation + Tier 1 Benchmark Refactoring (Prototype) ✅ COMPLETE (cycles used: ~6)
**Goal:** Build the `gpubench` CLI tool (scripts/gpubench) and refactor all 12 Tier 1 benchmarks to:
- Read parameters from `BENCH_PARAM_*` environment variables
- Output JSON-lines protocol (kernel events + summary) to stdout
- Send human-readable messages to stderr only
- Support the full parameter sweep driven by the CLI

The CLI implements: `list`, `compile`, `run`, `info`, `validate` commands per `docs/cli_design.md`.
End-to-end: `gpubench run --tier 1 --reps 1` works.

**Branch:** `ares/m22-cli-benchmarks` (merged to main via PR #20)

### M23 — Tier 2 Benchmark Refactoring (51 benchmarks) ✅ COMPLETE (cycles used: ~8)
**Goal:** Refactor all 51 Tier 2 benchmarks for parameterized execution and kernel-level timing.

Each benchmark has 2 source files (.hip + _metal.mm) = 102 files total. For each file:
1. Read all parameters from `BENCH_PARAM_*` environment variables (matching params.json)
2. Output JSON-lines protocol to stdout (kernel events with params + summary with metrics)
3. Send human-readable messages to stderr only
4. Add `setvbuf(stdout, NULL, _IONBF, 0)` at the start of main

After refactoring: `gpubench run --tier 2 --reps 1` works end-to-end.

**Branch:** `ares/m23-tier2-refactor` (merged to main via PR #21)

### M24 — Tier 3 + Tier 4 Benchmark Refactoring (18 benchmarks) ✅ COMPLETE (cycles used: ~4)
**Goal:** Refactor all Tier 3 (11 C++ benchmarks) and Tier 4 (7 benchmarks: 1 C++ + 6 Python) for parameterized execution and kernel-level timing.
**Deliverables (completed):** All 36 source files refactored (24 C++ + 12 Python). JSON-lines protocol, BENCH_PARAM_* env vars, setvbuf for C++. Fix round for triton_vector_add bench_mps.py. Merged via PR #22.
**Branch:** `ares/m24-tier3-tier4-refactor` (merged to main via PR #22)

### M25 — Tier 5 Benchmark Refactoring (26 benchmarks) ✅ COMPLETE (cycles used: ~5)
**Goal:** Refactor all 26 Tier 5 Python benchmarks for parameterized execution and kernel-level timing.
**Deliverables (completed):** All 52 Python files refactored (26 bench.py + 26 bench_mps.py). JSON-lines protocol, BENCH_PARAM_* env vars, import json/os/sys. All pass py_compile + AST validation. Merged via PR #23.
**Branch:** `ares/m25-tier5-refactor` (merged to main via PR #23)

### M26 — Integration, Testing & Final Polish ✅ COMPLETE (cycles used: ~3)
**Goal:** Complete the project: update README with CLI documentation, run end-to-end tests, clean up old harness code, close human issues.

**Deliverables (completed):**
1. **README.md rewrite** — Comprehensive gpubench CLI documentation (list, compile, run, validate, info). JSON-lines output protocol and SQLite schema documented. Old CSV/harness references removed.
2. **End-to-end smoke test** — `gpubench run --benchmark sdpa_bench --scaling-values 32 --reps 1` passed. All 5 SQLite tables verified.
3. **Old harness cleanup** — `scripts/run_all.sh` and `scripts/report.py` removed.
4. **Root Makefile update** — Old targets removed, gpubench convenience targets added.
5. **Human issues** — #142, #143, #144 all closed with verification comments.

**Branch:** `ares/m26-final-polish` (merged to main via PR #24)

---

## Previous Phase Summary

**Phase 1 complete (M1–M20).** 107 benchmarks across 5 tiers:
- **Tier 1:** 12 microbenchmarks (memory bandwidth, FP32/FP64/FP16/INT32 throughput, cache latency, PCIe, tensor cores, mixbench, shared mem, atomics, warp shuffle)
- **Tier 2:** 51 application benchmarks from 13 suites (SHOC, PolyBench, Rodinia, Parboil, CUDA SDK, HeteroMark, CUDA SDK Extended, Tango, NPB-GPU, Altis, Lonestar GPU, Graph Analytics, Chai)
- **Tier 3:** 11 library benchmarks (cuBLAS SGEMM/DGEMM/GEMV/HGEMM/Batched, cuFFT 1D/2D, cuSPARSE SpMV, cuDNN Conv/BatchNorm, CUTLASS GEMM)
- **Tier 4:** 7 alternative-language benchmarks (6 Triton + BabelStream)
- **Tier 5:** 26 ML workloads (ResNet, BERT, GPT-2, ViT, NanoGPT, SDPA, torch.compile, vLLM, llama.cpp, Ollama, LLaMA, DeepSeek, Qwen, GLM, Phi, Mistral, BERT fine-tune, Whisper, Stable Diffusion, MoE, RAG, CoT, Multi-Agent, SDXL, GPT-NeoX)

All benchmarks support 3 platforms (NVIDIA/CUDA, AMD/ROCm, Apple/Metal), use Make-only build system, and are self-contained.

**Phase 2 complete (M21–M26).** All 107 benchmarks parameterized with params.json. gpubench CLI tool built (list, compile, run, validate, info). SQLite output with platform info, session tracking, per-kernel timing, computed metrics. All human issues (#142, #143, #144) closed.

---

## Phase 3 — CLI Rewrite in Go (from human, Issue #217)

### M27 — Rewrite gpubench CLI in Go ✅ COMPLETE (cycles used: ~4)
**Goal:** Reimplement the `scripts/gpubench` Python CLI as a Go program with full feature parity.

**Requirements:**
- Go module at `cmd/gpubench/` with `go.mod` and `go.sum`
- Single binary output: `gpubench`
- Same 5 commands: `list`, `compile`, `run`, `info`, `validate`
- Same flags per command (exact same CLI interface)
- Same SQLite schema (from `docs/schema.sql`)
- Same JSON-lines output protocol parsing
- Same exit codes (0, 1, 2, 3, 4, 130)
- Same platform detection logic (Darwin→metal, /opt/rocm→rocm, else→cuda)
- Same benchmark discovery (walk `tier*/*/params.json`)
- Same parameter sweeping (scaling × non-scaling defaults, or full Cartesian with `--sweep-non-scaling`)
- Same BENCH_PARAM_* environment variable passing
- Graceful SIGINT handling (finish current run, save partial results)
- Update Makefile targets (`make list`, `make run`, etc.) to use Go binary
- Update README.md CLI section to reflect Go implementation
- Remove old Python `scripts/gpubench`
- Update `docs/cli_design.md` to reflect Go implementation

**Acceptance criteria:**
1. `go build ./cmd/gpubench` produces a working binary
2. `gpubench list` output matches Python version (same format, same benchmarks)
3. `gpubench validate` passes all 107 benchmarks
4. `gpubench info` shows correct platform info
5. `gpubench compile --tier 1 --platform metal` compiles tier 1 (on macOS)
6. `gpubench run --dry-run` shows correct execution plan
7. SQLite database schema matches `docs/schema.sql`
8. All exit codes match specification

**Budget:** 6 cycles

---

## Lessons Learned

- **Phase 1 (M1–M20):** 107 benchmarks across 5 tiers completed in ~60 cycles. Scaling workers helps — parallel benchmark implementation was effective.
- **Phase 2 (M21–M26):** Parameterization and CLI took ~30 cycles. Clear output protocols (JSON-lines) made integration smooth.
- **Phase 3 (M27):** CLI rewrite in Go completed in ~4 cycles. Well-documented existing Python implementation made the Go rewrite straightforward. PR #25 merged.
- **Phase 4 (M28):** Compilation fixes completed in ~3 cycles. CUDA compatibility shim approach (HIP→CUDA macros) was a quick fix.
- **Phase 5 (M29):** Native CUDA .cu files for all 75 C++ benchmarks completed in ~5 cycles. Workers split across tiers worked efficiently. Multiple workers (Max, Kai, Leo) each handled a subset.

---

## Phase 4 — Compilation Fixes (from human, Issue #221)

### M28 — Fix Compilation Failures & CLI Error Logging ✅ COMPLETE (cycles used: ~3)
**Goal:** Fix two issues reported by human:
1. CLI `gpubench compile` suppresses error output — users can't see why compilation failed
2. Tier 2 (and likely all tiers) compilation fails on some platforms

**Deliverables (completed):**
- CLI: Always print stderr on compilation failure (Kai, commit 1d101b8)
- Compilation: Added CUDA compatibility block to all 60 unguarded .hip files (Max, commit 8c9114a)
- Merged via PR #26

---

## Phase 5 — CUDA Native Source + Suite Completeness + CI (from human, Issues #224, #226, #228, #223, #229)

New batch of human requirements:
1. **#224**: Write CUDA programs, not only HIP — provide native .cu source files
2. **#226**: Complete benchmark suites — include ALL benchmarks from each suite
3. **#228**: Learn from MLPerf — include MLPerf benchmarks
4. **#223**: Docker-based CI for compilation testing
5. **#229**: Result persistence — community result sharing mechanism

### M29 — Native CUDA Source Files for All C++ Benchmarks ✅ COMPLETE (cycles used: ~5)
**Goal:** Create native `.cu` source files with CUDA API for every C++ benchmark.
**Deliverables (completed):** 75 native .cu files created for all C++ benchmarks (12 Tier 1, 51 Tier 2, 11 Tier 3, 1 Tier 4). All Makefiles updated to use .cu for CUDA platform. Merged via PR #27.
**Note:** Issue #224 closed.

### M30 — Complete Benchmark Suites: Highest-Priority Gaps (Issue #226) ✅ COMPLETE (cycles used: ~6)
**Goal:** Add the 9 highest-priority missing benchmarks from existing suites, plus 4 medium-priority ones. Addresses Issue #226 (include all benchmarks from each suite).
**Deliverables (completed):** 13 new Tier 2 benchmarks: npb_cg, npb_mg, parboil_cutcp, parboil_lbm, rodinia_hotspot3d, rodinia_lavamd, altis_gups, graph_cc, lonestar_dmr, polybench_2mm, polybench_3dconv, polybench_correlation, polybench_syr2k. All with .cu + .hip + Metal + Makefile + README + params.json + JSON-lines protocol.
**Branch:** `ares/m30-complete-suites` (merged to main via PR #28)
**Note:** Total benchmark count now 120. Issue #226 partially addressed — highest-priority gaps filled; remaining gaps in each suite are Low priority or covered by other tiers/suites.

### M31 — MLPerf Benchmarks for Tier 5 (Issue #228) ✅ COMPLETE (cycles used: ~3)
**Goal:** Add 5 MLPerf-inspired benchmarks to fill gaps in Tier 5 ML coverage.
**Deliverables (completed):** dlrm_inference, unet3d_inference, retinanet_inference, gnn_inference, gptj_inference — all with bench.py + bench_mps.py + Makefile + README + params.json. All verified on MPS.
**Branch:** `ares/m31-mlperf-tier5` (merged to main via PR #29)
**Note:** Issue #228 closed. Total benchmark count now 125.

### M32 — Docker CI + Result Persistence + README Update (Issues #223, #229) ✅ COMPLETE (cycles used: ~3)
**Goal:** Set up Docker-based CI for compilation verification, design a result persistence/sharing mechanism, and update README to reflect current 125 benchmarks.

**Deliverables (completed):**
1. **Docker CI** (Issue #223): `.github/workflows/compile.yml` — CUDA job (nvidia/cuda:12.6.3-devel-ubuntu22.04) + ROCm job (rocm/dev-ubuntu-22.04:6.0). Each installs Go, builds gpubench, compiles tiers 1-4.
2. **Result persistence** (Issue #229): `gpubench export` command (cmd/gpubench/export.go), `docs/result_sharing.md` spec, `community_results/` directory for community submissions.
3. **README update**: Updated to reflect 125 benchmarks, all tier tables complete, directory layout current.

**Branch:** `ares/m32-ci-persistence-readme` (merged to main via PR #31)

---

---

## Phase 6 — Fix Compilation & CI (from human, Issues #268, #269)

### M33 — Fix 11 Compilation Failures + CI Pipeline (Issues #268, #269) ✅ COMPLETE (cycles used: ~3)
**Goal:** Fix all 11 benchmarks that fail to compile on CUDA, and ensure the CI pipeline works correctly.

**Failing benchmarks (from Issue #268):**
1. `tier1/fp16_throughput` — Likely needs `-arch=sm_53` or higher in CUDA CFLAGS for half-precision intrinsics
2. `tier1/tensor_core_throughput` — Already has `-arch=sm_70`; investigate compile error
3. `tier2/cuda_nbody` — Standard CUDA; investigate compile error
4. `tier2/graph_cc` — Standard CUDA; investigate compile error
5. `tier2/graph_pr` — Standard CUDA; investigate compile error
6. `tier2/heteromark_pagerank` — Standard CUDA; investigate compile error
7. `tier2/lonestar_bh` — Standard CUDA; investigate compile error
8. `tier2/lonestar_sssp` — Standard CUDA; investigate compile error
9. `tier2/rodinia_backprop` — Has `#ifdef 0` (invalid preprocessor; should be `#if 0`)
10. `tier3/cudnn_batchnorm` — Needs cuDNN library; CI Docker container needs cuDNN
11. `tier3/cudnn_conv` — Needs cuDNN library; CI Docker container needs cuDNN

**Deliverables:**
1. Fix all 11 .cu files so they compile with `nvcc` on CUDA 12.6
2. Fix all 11 .hip files if corresponding issues exist
3. Update CI Docker container to use `nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04` (includes cuDNN) or add cuDNN installation step
4. Verify `gpubench compile --tier 1 --platform cuda` passes
5. Verify `gpubench compile --tier 2 --platform cuda` passes
6. Verify `gpubench compile --tier 3 --platform cuda` passes

**Branch:** `ares/m33-compile-fixes` (merged to main via PR #32)

**Deliverables (completed):**
1. Tier 1: fp16_throughput Makefile `-arch=sm_53`, tensor_core_throughput Makefile `-arch=sm_80`
2. Tier 2: 6 benchmarks added `--expt-relaxed-constexpr --expt-extended-lambda` to CUDA CFLAGS
3. Tier 2: cuda_nbody.cu fixed brace-init → `make_float3()`
4. Tier 2: rodinia_backprop.cu fixed `#ifdef 0` → `#if 0`
5. CI: CUDA container updated to `nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04` (includes cuDNN)
6. CI: ROCm job added `miopen-hip` package

**Note:** CI still fails due to GitHub Actions billing limits (external issue, not code). All code fixes verified correct by Apollo.

---

## Phase 7 — Fix Remaining Compilation Failures + Self-Hosted CI (from human, Issues #272, #273)

### M34 — Fix 6 CUDA Compilation Failures + Self-Hosted CI Runners
**Goal:** Fix 6 .cu files that fail to compile due to missing `#include <cstdint>`, and switch CI to self-hosted runners.

**Root cause (verified via Docker):** All 6 failing Tier 2 .cu files use `uint32_t` without including `<cstdint>`. nvcc 12.6 does not provide `uint32_t` transitively from other headers. Adding `#include <cstdint>` fixes all 6 files (verified in nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04 Docker container).

**Note:** cudnn_batchnorm and cudnn_conv from Issue #273 actually compile fine in the cuDNN container — they were false positives (likely the human tested without cuDNN installed).

**Files to fix (add `#include <cstdint>`):**
1. `tier2/cuda_nbody/cuda_nbody.cu`
2. `tier2/graph_cc/graph_cc.cu`
3. `tier2/graph_pr/graph_pr.cu`
4. `tier2/heteromark_pagerank/heteromark_pagerank.cu`
5. `tier2/lonestar_bh/lonestar_bh.cu`
6. `tier2/lonestar_sssp/lonestar_sssp.cu`

**Preventive fixes (also missing `<cstdint>` but currently compile by luck):**
7. `tier1/cache_latency/cache_latency.cu`
8. `tier1/int32_throughput/int32_throughput.cu`
9. `tier3/cusparse_spmv/cusparse_spmv.cu`

**CI fix:** Update `.github/workflows/compile.yml` to use `runs-on: self-hosted` instead of `runs-on: ubuntu-latest` for both CUDA and ROCm jobs. The sarchlab org has self-hosted runners available.

**Acceptance criteria:**
1. All 9 .cu files include `<cstdint>`
2. `gpubench compile --tier 1 --platform cuda` passes (0 failures)
3. `gpubench compile --tier 2 --platform cuda` passes (0 failures)
4. `gpubench compile --tier 3 --platform cuda` passes (0 failures)
5. CI workflow uses self-hosted runners
6. CI passes (or at least runs — runner availability may be an issue)

**Cycles budget:** 3

---

## Phase 8 — Fix CI Pipeline (from human, Issue #276)

### M35 — Fix CI to Pass on Self-Hosted Runners (Issue #276) ❌ DEADLINE MISSED (3/3 cycles used)
**Goal:** Make the CI pipeline green.
**What happened:** Ares's team tried 3 approaches over 3 cycles:
1. **Cycle 1**: Added `--network host` + arch-aware Go + commented ROCm. Failed: VCS error in `go build`.
2. **Cycle 2**: Removed `--network host`. CI passed apt-get/Go/checkout (networking WORKED). Failed: same VCS error.
3. **Cycle 3**: Re-added `--network host` + `-buildvcs=false`. Failed: Docker "conflicting network modes" error.

**Key discovery:** Run 2 (no `--network host`) proved default Actions Docker network has internet. Only fix needed was `-buildvcs=false`.

### M35.1 — Fix CI: Remove --network host, keep -buildvcs=false ✅ COMPLETE (cycles used: 1)
**Goal:** Make CI green. One change to PR #34: revert container config from object form (with `--network host`) back to inline string.
**Deliverables (completed):** PR #34 merged, main branch CI green (run 23445239783, success).

---

## Phase 9 — Repo Cleanup + Failure Diagnostics (from human, Issues #176, #177)

### M36 — Remove Workspace from Repo + Add Failure Diagnostics ✅ COMPLETE
**Goal:** Two tasks:
1. Remove `workspace/` directory from git tracking and add to `.gitignore` (Issue #176)
2. Improve `gpubench run` failure diagnostics: show stderr by default on failures, diagnose common issues, provide actionable suggestions (Issue #177)

**Deliverables (completed):** workspace removed from git, .gitignore updated, failure diagnostics improved. Merged via PRs #35 and #36.

---

## Historical Status Snapshot (before Issue #15)

M38 complete (PR #38 merged, CI green). All human issues resolved. Project fully complete.

**Final State:**
- 125 benchmarks across 5 tiers, all parameterized
- Go CLI tool (gpubench) with list, compile, run, validate, info, export
- Docker-based CI green on self-hosted runners
- Native .cu + .hip + Metal sources for all C++ benchmarks
- Result persistence mechanism (community_results/)
- No open issues. No open PRs. **CI GREEN.**

---

## Phase 10 — Fix Persistent Compilation Failures (from human, tbc-db Issue #1)

### M37 — Fix 8 Compilation Failures Across CUDA Environments ✅ COMPLETE (cycles used: 1)
**Goal:** Fix 8 benchmarks that fail to compile in the human's CUDA environment.
**Root cause:** 6 Tier 2 Makefiles had unnecessary `--expt-relaxed-constexpr --expt-extended-lambda` flags. 2 Tier 3 benchmarks needed cuDNN detection.
**Fix:** Removed unnecessary flags from 6 Tier 2 Makefiles, added cuDNN availability checks to 2 Tier 3 Makefiles.
**Branch:** `ares/fix-compilation-flags` (merged to main via PR #37). CI green.

---

## Phase 11 — Repo Cleanup (from human, tbc-db Issue #2)

### M38 — Remove Broken Submodule Entries + Clean Tracked Files ✅ COMPLETE (cycles used: 1)
**Goal:** Fix broken `repo` and `repo-chat` submodule entries and remove accidentally tracked files.

**Deliverables (completed):**
1. Removed `repo` and `repo-chat` from git tracking
2. Removed `config.yaml`, `results.db*`, `test_tier1.db*` from git tracking  
3. Added all above to `.gitignore`
4. CI remains green

**Branch:** `ares/m38-clean-submodules` (merged to main via PR #38)
**Note:** Human tbc-db issue #2 closed. All issues resolved.

---

## Previous Status — All Milestones Through M37 COMPLETE ✅

**125 benchmarks** across 5 tiers:
- **Tier 1:** 12 microbenchmarks
- **Tier 2:** 64 application benchmarks from 12+ suites (SHOC, PolyBench, Rodinia, Parboil, CUDA SDK, HeteroMark, Tango, NPB-GPU, Altis, Lonestar, Graph Analytics, Chai)
- **Tier 3:** 11 library benchmarks (cuBLAS, cuFFT, cuSPARSE, cuDNN, CUTLASS)
- **Tier 4:** 7 alternative-language benchmarks (6 Triton + BabelStream)
- **Tier 5:** 31 ML workloads (ResNet, BERT, GPT-2, ViT, SDPA, torch.compile, vLLM, llama.cpp, Ollama, LLaMA, DeepSeek, Qwen, GLM, Phi, Mistral, BERT fine-tune, Whisper, Stable Diffusion, MoE, RAG, CoT, Multi-Agent, SDXL, GPT-NeoX, DLRM, UNet3D, RetinaNet, GNN, GPT-J)

**All requirements fulfilled:**
- ✅ Make-only build system
- ✅ Multi-GPU platform support (CUDA, ROCm, Metal)
- ✅ Self-contained benchmarks
- ✅ Complete benchmark suites
- ✅ Full parameterization (125/125 params.json validated)
- ✅ Kernel-level execution timing (JSON-lines protocol)
- ✅ Go CLI tool (`gpubench` — list, compile, run, validate, info, export)
- ✅ SQLite results database with platform info
- ✅ Native .cu source files (88 C++ benchmarks)
- ✅ Docker-based CI (green on self-hosted runners)
- ✅ Result persistence mechanism (community_results/)
- ✅ MLPerf benchmarks included
- ✅ 12+ Tier 2 benchmark suites (beyond 8 required)
- ✅ All human issues closed (#100, #101, #142, #143, #144, #217, #221, #223, #224, #226, #228, #229, #268, #269, #272, #273, #276)

**No open issues. No open PRs. CI green.**

## Lessons Learned (Phase 8)
- Default GitHub Actions Docker networking works on self-hosted runners. `--network host` conflicts with Actions' managed network.
- Always check what WORKED in a failed CI run, not just the error.

---

## Phase 12 — cuDNN Compile Robustness Follow-up (from human, tbc-db Issue #10)

### M39 — Resolve Remaining cuDNN Compile Failures ✅ COMPLETE (cycles used: 3)
**Goal:** Ensure `cudnn_conv` and `cudnn_batchnorm` compile behavior is robust when cuDNN is missing and diagnostics are explicit.

**Deliverables (completed):**
1. Deterministic cuDNN header/library detection expanded for x86_64 + aarch64
2. Missing-cuDNN path compiles to explicit diagnostic fallback binary instead of opaque linker/header failure
3. CI regression check added for fallback behavior
4. Merged via PR #39; main branch CI green

---

## Phase 13 — Parameter Selection Re-evaluation (from human, tbc-db Issue #15)

Human feedback from real runs indicates many non-ML benchmarks still use oversized and power-of-two-only scaling ranges. Independent audit on main found:
- 94 non-ML benchmarks (tiers 1–4)
- 35 benchmarks with **all** scaling points as powers of two
- 46 benchmarks with max scaling value **> 65536**
- Tier 1 throughput-style params (`ops_per_thread`, `fmas_per_thread`, `mads_per_thread`) currently go up to 2,097,152

### M40 — Re-evaluate parameter selection for non-ML benchmarks ❌ DEADLINE MISSED (3/3 cycles used)
**What happened:** Scope was too broad for one milestone without tighter decomposition and explicit static acceptance checks.

### M40.1 — Tier 1 throughput + Tier 2 scaling range normalization (next)
**Goal:** Deliver a first verifiable slice of Issue #15 with strict, checkable constraints.

**Planned deliverables:**
1. Update Tier 1 throughput-like benchmarks (`atomic_operations`, `warp_shuffle`, `fp16_throughput`, `fp32_throughput`, `fp64_throughput`, `int32_throughput`) so scaling values include non-power-of-two points and upper bound is reduced to runtime-realistic levels (cap at 65536 for per-thread-op style params).
2. Update all Tier 2 `params.json` scaling arrays so each benchmark includes non-power-of-two points and removes unnecessary extreme upper-bound values chosen only for stress testing.
3. Add a lightweight static checker script for parameter policy (non-power-of-two coverage + upper-bound checks) and wire it into CI.
4. Document the new parameter selection rationale in `docs/params_format.md` (runtime realism for non-ML runs, target around 15s).

### M40.2 — Tier 3 + Tier 4 normalization (planned)
Apply the same policy and checker enforcement to all Tier 3 and Tier 4 non-ML benchmarks.

## Current Status
Project remains functionally complete, but **human issue #15 is open** and parameter ranges require recalibration. Next implementation focus is M40.1.

## Lessons Learned (Current)
- After a deadline miss, split by tier and add machine-checkable acceptance criteria before reassigning.
- Parameter quality must be treated as first-class correctness, not a post-hoc tuning task.
