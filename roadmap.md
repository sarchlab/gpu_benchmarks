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

### M19 — Tier 2: New Benchmark Suites Batch 1 (4 new suites, 10 benchmarks)
**Goal:** Add 10 benchmarks from 4 new Tier 2 suites. This addresses Issue #101 (need 8+ more suites beyond SHOC/Rodinia/PolyBench/Parboil/CUDA Samples).

**New suites in this batch:**
1. **HeteroMark** (3): heteromark_aes, heteromark_fir, heteromark_pagerank
2. **CUDA SDK Extended** (3): cuda_nbody, cuda_convolution_separable, cuda_scan_large
3. **Tango** (2): tango_blackscholes, tango_binomial_options
4. **NPB-GPU** (2): npb_ep, npb_is

**Status:** PLANNED
**Cycles budget:** 5

### M20 — Tier 2: New Benchmark Suites Batch 2 (4+ new suites, ~10 benchmarks)
**Goal:** Add benchmarks from 4+ more Tier 2 suites to reach 8+ new suites total.

**New suites in this batch:**
5. **Altis** (3): altis_cfd, altis_particlefilter, altis_raytracing
6. **Lonestar GPU** (2): lonestar_sssp, lonestar_bh
7. **Graph Analytics** (2): graph_pr, graph_tc
8. **Chai** (2): chai_hsto, chai_cedd

**Status:** PLANNED
**Cycles budget:** 5

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
