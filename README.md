# GPU Benchmarks

A comprehensive GPGPU benchmark suite covering microbenchmarks, application workloads, library kernels, alternative language implementations, and ML workloads.

---

## Purpose

This suite provides a **GPU performance reference** for two primary use cases:

1. **GPU hardware comparison** — Characterize and compare GPU performance across vendors, architectures, and generations using a standardized set of workloads spanning raw hardware capabilities to end-to-end applications.

2. **GPU simulator validation** — Provide reference performance data and NVBit instruction traces that GPU architecture simulators (e.g., [BBSim/bb_triosim](https://github.com/sarchlab/bb_triosim)) can use to validate simulation accuracy against real hardware.

The suite is designed to cover the widest possible diversity of GPU compute and memory access patterns, from low-level microbenchmarks through library calls to full ML inference pipelines. Each benchmark supports parameterized scaling sweeps, and all results are stored in a queryable SQLite database for systematic analysis.

---

## Overview

This suite contains **125 benchmarks across five tiers** of increasing complexity and scope, allowing researchers and engineers to evaluate GPU hardware and software stacks at every level — from raw memory bandwidth to end-to-end ML training throughput.

---

## Measurement Methodology

The default Tier 1 and Tier 2 protocol targets **cold-cache kernel execution at a raised GPU clock**. An untimed vector-add warmer conditions the device before the measured operation. Its intended effects are to raise clocks from idle and evict benchmark data from the last-level cache; the achieved clock/cache state must be qualified on each target. A fixed scratch size and duration do not by themselves prove a reproducible hardware state. See [measurement interpretation](docs/measurement_interpretation.md) for the distinction between the protocol's intent and qualified measurement claims.

The rules split cleanly by implementation language: everything that matters when writing the **C/C++/HIP/CUDA benchmark code** is a *benchmark rule*; everything that matters when writing the **Go runner (`gpubench`)** is a *harness rule*. A benchmark binary is a complete, correct measurement on its own; the harness only decides which binaries to run, with which parameters, how many times, and where the output goes.

### Benchmark rules (C/C++/HIP/CUDA)

**Sourcing and code structure:**

1. **Preserve the upstream algorithm; source-level identity is not required.** A Tier 2 benchmark may be an independent HIP/CUDA reimplementation rather than a copy of the upstream source. It is faithful when it preserves the upstream workload's problem definition, mathematical operations or recurrence, algorithmic stages and data dependencies, boundary and termination rules, and asymptotic work and parameter scaling. Exact source text, host scaffolding, memory layout, tiling, thread/block mapping, and kernel decomposition may differ, and portability or correctness fixes are allowed, provided they do not change those algorithmic semantics. Material implementation differences must be documented. Omitting a defining stage, substituting a different algorithm or numerical problem, changing update or convergence semantics, moving defining GPU work out of the measured workload, or changing how work scales is not faithful; such a variant must be brought into alignment or identified as a different benchmark rather than presented as the upstream workload.

2. **HIP and CUDA share one source.** Each benchmark is a single `.hip` file that compiles under both `hipcc` (AMD) and `nvcc` (NVIDIA) via `common/hip_cuda_compat.h`. There are no per-vendor copies that could diverge or acquire vendor-specific overhead.

3. **Minimize common code; keep it header-only.** The shared infrastructure is three headers (`common/hip_cuda_compat.h`, `common/bench_timing.h`, `common/bench_warmup.h`) — nothing to build or link separately, so every benchmark compiles standalone with one compiler invocation.

**Measurement:**

4. **One measurement per execution.** A benchmark binary executes its operation **exactly once** and times it — no in-program warmup or iteration loop. The single run is the **full algorithm**: an iterative benchmark still runs its complete algorithm (BFS its whole traversal, k-means its whole `max_iter` refinement, a stencil all its timesteps) — "once" means the algorithm is not *repeated*, not that it is truncated. The untimed `cache_latency` primer is an explicit entry-policy exception described below.

5. **Cold cache, warm clock is the default entry policy.** Immediately before the timed operation, run the **vector-add warmer**: a grid-stride read-modify-write kernel over a 768 MiB scratch buffer, repeated toward a target of **~100 ms of sustained load**.

   - Sustained ALU and memory traffic is intended to raise clocks from idle.
   - Touching the separate scratch buffer is intended to evict the benchmark's working set from the last-level cache without priming its own buffers.

   Verify that the scratch footprint exceeds the relevant cache capacity and qualify clock/cache effects on the target GPU. Benchmarks whose subject requires resident data must document their entry-policy exception. [`cache_latency`](tier1/cache_latency/README.md) runs the warmer first, then an untimed pointer-chase primer, then the timed chase. The primer remains when the warmer is disabled and must be replayed without counting its time in a simulator; actual cache residency still requires target qualification. The warmer is controlled by one measurement parameter read from the environment:

   | Parameter | Env var | Default | Meaning |
   |-----------|---------|---------|---------|
   | `vector_add_warmup` | `BENCH_PARAM_vector_add_warmup` | `true` | Run the untimed vector-add warmer before measurement. `false` disables it; it does not promise a particular cache or clock state. |

6. **The timed bracket contains only the operation under test.** Host-side setup, `hipMalloc`, initial Host→Device uploads, and result verification all happen outside the timed region. For most benchmarks that means **we only care about kernel time** — the bracket contains kernel launches and nothing else. The only exception is a few Tier 1 benchmarks whose subject *is* the transfer (`memory_bandwidth`, `pcie_bandwidth`): there the bracket times the memory-copy operations, because copy time/bandwidth is what they measure.

7. **GPU event timing, raw output.** Time is measured with **GPU events** (`hipEvent`/`cudaEvent`) bracketing exactly the enqueued operation — no host wall-clock timing is needed. Benchmarks never compute GFLOPS, GB/s, or latency — they emit the raw time plus the work-descriptor parameters as JSON lines on stdout, and all derived metrics are computed downstream.

8. **Per-kernel records; their sum is the benchmark's execution time.** Each kernel's time is measured and recorded individually (accumulated across that kernel's launches within the run), one record per kernel. The sum of all per-kernel times is emitted as `summary.total_time_ms` — the benchmark's execution time. Data reuse *between* kernels within the run (kernel B reading kernel A's output) is left intact — it is part of the algorithm being measured; only the state *entering* the run is conditioned by the entry policy.

9. **Deterministic, equivalent runs.** Benchmarks that mutate their inputs re-initialize device state before the timed run, so the measured work is identical on every execution.

10. **Every declared parameter drives execution.** A parameter in `params.json` is read by the benchmark and changes what runs; parameters that would merely be echoed into records are not declared.

### Parameter sweeping rules

Each benchmark declares its parameters in `params.json`; the sweep is structured around one scaling axis plus a small set of configuration knobs:

1. **Exactly one scaling parameter per benchmark.** It is the single axis that controls *how much* work the benchmark does (matrix dimension, array bytes, node count, …). It is chosen so performance is a predictable function of it (linear, polynomial, or exponential) — sweeping it traces the benchmark's performance curve.

2. **Scaling values span the interesting range densely.** Typically 20–30 ordered values (ideally at least 10), from sizes small enough to be cache-resident / under-saturate the device up to sizes that far exceed the last-level cache / fully saturate it — so the sweep crosses the hardware's capacity boundaries rather than sampling one regime. Both aligned (power-of-two) and non-aligned sizes are worth including. No device-specific ceiling is assumed (a size that saturates one GPU may not saturate another); the practical upper bound is the harness timeout per run. The value counts here and in rule 3 are empirical guidance, not hard requirements — a benchmark whose meaningful range is narrower may declare fewer values (`gpubench validate` flags shortfalls as warnings, not errors).

3. **Non-scaling parameters change *how* the work is done, not how much.** Block size, mode, precision, algorithm variant — typically 3–5 values each, with a required default. By default the harness sweeps their cartesian product; `--sweep-non-scaling=false` pins them to defaults.

4. **Every swept value must matter and must work.** Each value changes execution (no placeholder values whose runs would be identical), and every combination respects the benchmark's intrinsic constraints so it runs correctly — a value list that silently produces broken or duplicate runs is a bug in `params.json`.

5. **Measurement controls are not sweep parameters.** `vector_add_warmup` is fixed per run (see harness rules), never crossed into the sweep.

### Harness rules (Go, `gpubench`)

1. **Orchestrate, never alter.** The harness selects parameter configurations, launches benchmark processes with `BENCH_PARAM_*` environment variables, and stores their stdout — it contains no measurement logic and never post-processes a time.

2. **Repetition = process re-execution.** Because each benchmark execution is one complete, independent measurement, `gpubench run --reps N` simply runs the binary N times from scratch (fresh `hipMalloc`, fresh context, fresh clock state) — every sample is fully independent, capturing process- and allocation-level variance. Take the **minimum** GPU time across reps for the least-perturbed sample; use higher rep counts only when characterizing run-to-run variance itself.

3. **Measurement controls are fixed, not swept.** `vector_add_warmup` is a measurement control, not a workload parameter, so it is excluded from the non-scaling cartesian sweep — it is fixed for a whole run to its default unless overridden globally: `gpubench run --vector-add-warmup true|false`. The value used is recorded per run in the database.

4. **Store raw, derive downstream.** Every run's parameters, per-kernel times, and summary land in SQLite exactly as emitted; GFLOPS/GB/s/latency are computed later by analysis scripts, never at collection time.

5. **Traces omit the warmer.** During NVBit trace collection the harness forces `vector_add_warmup=false`. The raw capture retains benchmark-launched kernels, including `cache_latency`'s untimed primer. Replay must preserve its primed state while excluding primer work from measured latency; see the [cache-latency replay requirements](tier1/cache_latency/README.md).

---

## 5-Tier Structure

| Tier | Name | Description |
|------|------|-------------|
| **Tier 1** | Microbenchmarks | Low-level hardware characterization: memory bandwidth, compute throughput (FP32/FP64/FP16/INT32), cache latency, PCIe transfer rates, atomic ops, warp shuffles |
| **Tier 2** | Application Benchmarks | Classic GPU benchmark suites: SHOC, PolyBench-GPU, Rodinia, Parboil, HeteroMark, CUDA SDK Extended, Tango, NPB-GPU, Altis, Lonestar GPU, Graph Analytics, Chai |
| **Tier 3** | Library Benchmarks | Vendor and third-party library kernels: cuBLAS, CUTLASS, cuFFT, cuSPARSE, cuDNN |
| **Tier 4** | Alternative Languages | Non-CUDA implementations: Triton, BabelStream |
| **Tier 5** | ML Workloads | End-to-end machine learning training/inference and LLM benchmarks: ResNet-50, BERT, GPT-2, ViT, NanoGPT, SDPA, torch.compile, vLLM, llama.cpp, Ollama, LLaMA, DeepSeek, Qwen, GLM, Phi, Mistral, BERT fine-tune, Whisper, Stable Diffusion, MoE, RAG, CoT, Multi-Agent, SDXL, GPT-NeoX, DLRM, GNN, GPT-J, RetinaNet, UNet3D |

---

## Directory Layout

```
gpu_benchmarks/
├── Makefile                        # Root build file
├── README.md                       # This file
├── tier1/                          # Microbenchmarks (13)
│   ├── Makefile
│   ├── memory_bandwidth/
│   ├── fp32_throughput/
│   ├── fp64_throughput/
│   ├── fp16_throughput/
│   ├── int32_throughput/
│   ├── cache_latency/
│   ├── cache_bandwidth/
│   ├── pcie_bandwidth/
│   ├── mixbench/
│   ├── shared_mem_bandwidth/
│   ├── atomic_operations/
│   ├── warp_shuffle/
│   └── empty_kernel/
├── tier2/                          # Application benchmarks (63)
│   ├── Makefile
│   ├── shoc_devicememory/
│   ├── shoc_fft/
│   ├── shoc_gemm/
│   ├── shoc_reduction/
│   ├── shoc_scan/
│   ├── shoc_sort/
│   ├── shoc_spmv/
│   ├── shoc_stencil2d/
│   ├── shoc_triad/
│   ├── polybench_2dconv/
│   ├── polybench_2mm/
│   ├── polybench_3dconv/
│   ├── polybench_3mm/
│   ├── polybench_atax/
│   ├── polybench_bicg/
│   ├── polybench_correlation/
│   ├── polybench_fdtd2d/
│   ├── polybench_gemm/
│   ├── polybench_gramschmidt/
│   ├── polybench_jacobi2d/
│   ├── polybench_mvt/
│   ├── polybench_syr2k/
│   ├── rodinia_backprop/
│   ├── rodinia_bfs/
│   ├── rodinia_gaussian/
│   ├── rodinia_hotspot/
│   ├── rodinia_hotspot3d/
│   ├── rodinia_kmeans/
│   ├── rodinia_lavamd/
│   ├── rodinia_lud/
│   ├── rodinia_nw/
│   ├── rodinia_pathfinder/
│   ├── rodinia_srad/
│   ├── parboil_cutcp/
│   ├── parboil_histogram/
│   ├── parboil_lbm/
│   ├── parboil_sgemm/
│   ├── parboil_spmv/
│   ├── parboil_stencil/
│   ├── cuda_transpose/
│   ├── heteromark_aes/
│   ├── heteromark_fir/
│   ├── heteromark_pagerank/
│   ├── cuda_nbody/
│   ├── cuda_convolution_separable/
│   ├── cuda_scan_large/
│   ├── tango_blackscholes/
│   ├── tango_binomial_options/
│   ├── npb_cg/
│   ├── npb_ep/
│   ├── npb_is/
│   ├── npb_mg/
│   ├── altis_cfd/
│   ├── altis_gups/
│   ├── altis_particlefilter/
│   ├── altis_raytracing/
│   ├── lonestar_bh/
│   ├── lonestar_sssp/
│   ├── graph_cc/
│   ├── graph_pr/
│   ├── graph_tc/
│   ├── chai_hsto/
│   └── chai_cedd/
├── tier3/                          # Library benchmarks (11)
│   ├── Makefile
│   ├── cublas_sgemm/
│   ├── cublas_dgemm/
│   ├── cublas_hgemm/
│   ├── cublas_gemv/
│   ├── cublas_batched_gemm/
│   ├── cudnn_conv/
│   ├── cudnn_batchnorm/
│   ├── cufft_1d/
│   ├── cufft_2d/
│   ├── cusparse_spmv/
│   └── cutlass_gemm/
├── tier4/                          # Alternative language benchmarks (7)
│   ├── Makefile
│   ├── babelstream/
│   ├── triton_vector_add/
│   ├── triton_gemm/
│   ├── triton_softmax/
│   ├── triton_layer_norm/
│   ├── triton_dropout/
│   └── triton_flash_attention/
├── tier5/                          # ML workloads (31)
│   ├── Makefile
│   ├── resnet50_train/
│   ├── resnet50_infer/
│   ├── bert_inference/
│   ├── bert_finetune/
│   ├── nanogpt_train/
│   ├── gpt2_generate/
│   ├── vit_inference/
│   ├── sdpa_bench/
│   ├── torch_compile_bench/
│   ├── vllm_offline/
│   ├── llamacpp_bench/
│   ├── ollama_bench/
│   ├── llama_inference/
│   ├── deepseek_inference/
│   ├── qwen_inference/
│   ├── glm_inference/
│   ├── phi_inference/
│   ├── mistral_inference/
│   ├── whisper_inference/
│   ├── stable_diffusion_bench/
│   ├── moe_inference/
│   ├── rag_pipeline/
│   ├── cot_reasoning/
│   ├── multi_agent_bench/
│   ├── sdxl_inference/
│   ├── gptneox_train/
│   ├── dlrm_inference/
│   ├── gnn_inference/
│   ├── gptj_inference/
│   ├── retinanet_inference/
│   └── unet3d_inference/
├── cmd/
│   └── gpubench/                   # gpubench CLI (Go)
└── docs/                          # Detailed specifications
    ├── cli_design.md
    ├── output_protocol.md
    ├── params_format.md
    └── result_sharing.md
```

---

## Tier 1 — Microbenchmarks

The metric column names downstream quantities. See [measurement interpretation](docs/measurement_interpretation.md) for work-count qualification, single-block versus device-wide results, timing resolution, and transfer semantics.

| Benchmark | Metric | Description |
|-----------|--------|-------------|
| `memory_bandwidth` | GB/s | Global memory bandwidth (H2D, D2H, D2D) |
| `fp32_throughput` | GFLOPS | Single-precision FP32 FMA throughput (declared grid: one block) |
| `fp64_throughput` | GFLOPS | Double-precision FP64 FMA throughput (declared grid: one block) |
| `fp16_throughput` | GFLOPS | Half-precision FP16 FMA throughput (declared grid: one block) |
| `int32_throughput` | GIOPS | Integer INT32 multiply-add throughput (declared grid: one block) |
| `cache_latency` | ns | L1/L2/global cache latency via pointer-chasing |
| `pcie_bandwidth` | GB/s | PCIe/Thunderbolt host-device transfer bandwidth |
| `mixbench` | GFLOPS/GB/s | Nominal compute/logical-byte intensity sweep |
| `shared_mem_bandwidth` | GB/s | Shared/threadgroup memory bandwidth (single SM/CU) |
| `atomic_operations` | Gops/s | Atomic operation throughput (Add/CAS/Min) |
| `warp_shuffle` | Gops/s | Warp/SIMD shuffle issue rate (single SM/CU, dependent chain) |
| `cache_bandwidth` | GB/s | Cache-hierarchy bandwidth vs working-set size (shared/private modes) |
| `empty_kernel` | µs | Empty-dispatch GPU event interval and host bracket time |

---

## Tier 2 — Application Benchmarks

| Suite | Benchmark | Description |
|-------|-----------|-------------|
| SHOC | `shoc_devicememory` | Device memory read/write bandwidth |
| SHOC | `shoc_fft` | Fast Fourier Transform |
| SHOC | `shoc_gemm` | Dense matrix multiplication |
| SHOC | `shoc_reduction` | Parallel reduction |
| SHOC | `shoc_scan` | Parallel prefix sum (scan) |
| SHOC | `shoc_sort` | Parallel sort |
| SHOC | `shoc_spmv` | Sparse matrix-vector multiply |
| SHOC | `shoc_stencil2d` | 2D stencil computation |
| SHOC | `shoc_triad` | Stream triad (a = b + α·c) |
| PolyBench | `polybench_2dconv` | 2D convolution |
| PolyBench | `polybench_3mm` | Three matrix multiplications |
| PolyBench | `polybench_atax` | Matrix transpose and vector multiply |
| PolyBench | `polybench_bicg` | BiCG sub-kernel of BiCGStab solver |
| PolyBench | `polybench_fdtd2d` | 2D Finite-Difference Time-Domain |
| PolyBench | `polybench_gemm` | General matrix multiplication |
| PolyBench | `polybench_gramschmidt` | Gram-Schmidt decomposition |
| PolyBench | `polybench_jacobi2d` | 2D Jacobi stencil |
| PolyBench | `polybench_mvt` | Matrix-vector product and transpose |
| PolyBench | `polybench_2mm` | Two matrix multiplications |
| PolyBench | `polybench_3dconv` | 3D convolution |
| PolyBench | `polybench_correlation` | Correlation computation |
| PolyBench | `polybench_syr2k` | Symmetric rank-2k update |
| Rodinia | `rodinia_backprop` | Back-propagation training |
| Rodinia | `rodinia_bfs` | Breadth-first search |
| Rodinia | `rodinia_gaussian` | Gaussian elimination |
| Rodinia | `rodinia_hotspot` | Thermal simulation |
| Rodinia | `rodinia_kmeans` | K-means clustering |
| Rodinia | `rodinia_lud` | LU decomposition |
| Rodinia | `rodinia_nw` | Needleman-Wunsch sequence alignment |
| Rodinia | `rodinia_pathfinder` | Dynamic programming pathfinder |
| Rodinia | `rodinia_srad` | Speckle Reducing Anisotropic Diffusion |
| Rodinia | `rodinia_hotspot3d` | 3D thermal simulation |
| Rodinia | `rodinia_lavamd` | Molecular dynamics (Lennard-Jones) |
| Parboil | `parboil_histogram` | Histogram computation |
| Parboil | `parboil_sgemm` | Single-precision GEMM |
| Parboil | `parboil_spmv` | Sparse matrix-vector multiply |
| Parboil | `parboil_stencil` | 3D stencil computation |
| Parboil | `parboil_cutcp` | Coulombic potential (cutoff) |
| Parboil | `parboil_lbm` | Lattice-Boltzmann method fluid simulation |
| Other | `cuda_transpose` | Optimized matrix transpose |
| HeteroMark | `heteromark_aes` | AES-256 encryption |
| HeteroMark | `heteromark_fir` | FIR (Finite Impulse Response) filter |
| HeteroMark | `heteromark_pagerank` | PageRank on synthetic graph |
| CUDA SDK Extended | `cuda_nbody` | N-body gravitational simulation |
| CUDA SDK Extended | `cuda_convolution_separable` | Separable 2D convolution |
| CUDA SDK Extended | `cuda_scan_large` | Work-efficient parallel prefix sum |
| Tango | `tango_blackscholes` | Black-Scholes option pricing |
| Tango | `tango_binomial_options` | Binomial option pricing |
| NPB-GPU | `npb_ep` | Embarrassingly Parallel (EP) |
| NPB-GPU | `npb_is` | Integer Sort (IS) |
| NPB-GPU | `npb_cg` | Conjugate Gradient (CG) |
| NPB-GPU | `npb_mg` | Multigrid (MG) |
| Altis | `altis_cfd` | Computational Fluid Dynamics (Euler solver) |
| Altis | `altis_particlefilter` | Particle Filter (SIR Monte Carlo localization) |
| Altis | `altis_raytracing` | Simple ray tracing (Phong shading) |
| Altis | `altis_gups` | Giga Updates Per Second (random memory access) |
| Lonestar GPU | `lonestar_sssp` | Single-Source Shortest Path (Bellman-Ford) |
| Lonestar GPU | `lonestar_bh` | Barnes-Hut N-body (2D quadtree approximation) |
| Graph Analytics | `graph_pr` | PageRank (pull-based iterative) |
| Graph Analytics | `graph_tc` | Triangle Counting (sorted adjacency intersection) |
| Graph Analytics | `graph_cc` | Connected Components (label propagation) |
| Chai | `chai_hsto` | Histogram (shared memory privatization) |
| Chai | `chai_cedd` | Canny Edge Detection (5-stage pipeline) |

---

## Tier 3 — Library Benchmarks

| Benchmark | Library | Description |
|-----------|---------|-------------|
| `cublas_sgemm` | cuBLAS | Single-precision GEMM |
| `cublas_dgemm` | cuBLAS | Double-precision GEMM |
| `cublas_hgemm` | cuBLAS | Half-precision GEMM |
| `cublas_gemv` | cuBLAS | Matrix-vector multiplication |
| `cublas_batched_gemm` | cuBLAS | Batched GEMM |
| `cudnn_conv` | cuDNN | Convolution (forward + backward) |
| `cudnn_batchnorm` | cuDNN | Batch normalization |
| `cufft_1d` | cuFFT | 1D Fast Fourier Transform |
| `cufft_2d` | cuFFT | 2D Fast Fourier Transform |
| `cusparse_spmv` | cuSPARSE | Sparse matrix-vector multiply |
| `cutlass_gemm` | CUTLASS | Template-based GEMM |

---

## Tier 4 — Alternative Languages

| Benchmark | Language/Framework | Description |
|-----------|-------------------|-------------|
| `babelstream` | OpenCL/HIP/SYCL | BabelStream memory bandwidth (multi-backend) |
| `triton_vector_add` | Triton | Vector addition kernel |
| `triton_gemm` | Triton | Matrix multiplication kernel |
| `triton_softmax` | Triton | Fused softmax kernel |
| `triton_layer_norm` | Triton | Fused layer normalization kernel |
| `triton_dropout` | Triton | Fused dropout kernel |
| `triton_flash_attention` | Triton | Flash attention kernel |

---

## Tier 5 — ML Workloads

| Benchmark | Framework | Description |
|-----------|-----------|-------------|
| `resnet50_train` | PyTorch | ResNet-50 training throughput |
| `resnet50_infer` | PyTorch | ResNet-50 inference latency and throughput |
| `bert_inference` | PyTorch | BERT inference latency |
| `nanogpt_train` | PyTorch | NanoGPT training throughput |
| `gpt2_generate` | PyTorch | GPT-2 text generation throughput |
| `vit_inference` | PyTorch | Vision Transformer inference |
| `sdpa_bench` | PyTorch | Scaled dot-product attention benchmark |
| `torch_compile_bench` | PyTorch | torch.compile vs eager mode comparison |
| `vllm_offline` | vLLM | Offline batch inference with PagedAttention |
| `llamacpp_bench` | llama.cpp | llama.cpp subprocess inference throughput |
| `ollama_bench` | Ollama | Ollama HTTP API inference throughput |
| `llama_inference` | PyTorch | LLaMA architecture autoregressive generation |
| `deepseek_inference` | PyTorch | DeepSeek-style model inference (GQA) |
| `qwen_inference` | PyTorch | Qwen2 model autoregressive generation |
| `glm_inference` | PyTorch | ChatGLM-style model autoregressive generation |
| `phi_inference` | PyTorch | Microsoft Phi model inference (GQA) |
| `mistral_inference` | PyTorch | Mistral model inference (Grouped Query Attention) |
| `bert_finetune` | PyTorch | BERT fine-tuning (forward + backward + optimizer) |
| `whisper_inference` | PyTorch | OpenAI Whisper audio model inference |
| `stable_diffusion_bench` | PyTorch/Diffusers | Stable Diffusion UNet denoising loop |
| `moe_inference` | PyTorch | Mixture-of-Experts transformer inference (4 experts, top-2 gating) |
| `rag_pipeline` | PyTorch | RAG pipeline: encoder + retriever + generator (1000 docs) |
| `cot_reasoning` | PyTorch | Chain-of-Thought multi-step reasoning generation |
| `multi_agent_bench` | PyTorch | Multi-agent LLM pipeline (3 agents, sequential) |
| `sdxl_inference` | PyTorch/Diffusers | SDXL-style UNet denoising loop (20 steps) |
| `gptneox_train` | PyTorch/Transformers | GPT-NeoX training (forward + backward + optimizer) |
| `dlrm_inference` | PyTorch | DLRM (Deep Learning Recommendation Model) inference |
| `gnn_inference` | PyTorch | GNN (Graph Neural Network) inference with R-GAT proxy |
| `gptj_inference` | PyTorch | GPT-J architecture autoregressive generation |
| `retinanet_inference` | PyTorch | RetinaNet object detection inference |
| `unet3d_inference` | PyTorch | 3D U-Net medical image segmentation inference |

---

## How to Build

### Prerequisites

- GNU Make ≥ 3.81
- **NVIDIA (CUDA):** CUDA Toolkit ≥ 11.6 (with `nvcc`) and a C++17-capable host compiler. Builds default to `-arch=native` (the GPU present at compile time); override with `make CUDA_ARCH=sm_80` when cross-compiling or building on a GPU-less node.
- **AMD (ROCm):** ROCm ≥ 5.0 (with `hipcc`) installed at `/opt/rocm`

### Platform Detection

The build system auto-detects the platform at build time:

| Condition | Platform selected |
|-----------|------------------|
| `/opt/rocm/bin/hipcc` exists | `rocm` |
| Otherwise | `cuda` |

You can override detection by passing `PLATFORM` explicitly.

### Build Steps

```bash
# Clone the repository
git clone https://github.com/sarchlab/gpu_benchmarks.git
cd gpu_benchmarks

# Build everything (auto-detect platform)
make

# Build with explicit platform
make PLATFORM=cuda    # NVIDIA GPU
make PLATFORM=rocm    # AMD GPU

# Build only a specific tier
make tier1
make tier2

# Clean all build artifacts
make clean
```

---

## gpubench CLI

The suite ships with `gpubench`, a Go CLI tool that discovers, compiles, runs, and validates all benchmarks. Results are stored in an SQLite database.

```bash
# Build the CLI
go build -o gpubench ./cmd/gpubench

# Or use Make (builds automatically when needed)
make list
```

### Commands

| Command | Description |
|---------|-------------|
| `list` | List discovered benchmarks |
| `compile` | Compile C++ benchmarks (Tier 1–3) |
| `run` | Run benchmarks with parameter sweeps; store results in SQLite |
| `validate` | Validate all `params.json` files |
| `info` | Display detected platform information |

### Quick Start

```bash
# Build the CLI binary
go build -o gpubench ./cmd/gpubench

# List all benchmarks
./gpubench list

# List benchmarks in a specific tier
./gpubench list --tier 1

# Show detected platform info
./gpubench info

# Compile all C++ benchmarks (auto-detect platform)
./gpubench compile

# Validate every params.json
./gpubench validate

# Run all benchmarks (7 reps, results → results.db)
./gpubench run

# Run a single benchmark with 3 reps
./gpubench run --benchmark shoc_gemm --reps 3

# Run only Tier 1 benchmarks
./gpubench run --tier 1

# Preview the execution plan without running
./gpubench run --dry-run

# Override scaling parameter values
./gpubench run --benchmark fp32_throughput --scaling-values 64,256,1024

# Use only default non-scaling parameter values
./gpubench run --sweep-non-scaling=false

# Override the cold-cache warmer globally (all tier1/tier2 benchmarks)
./gpubench run --vector-add-warmup false           # warm-cache mode (no warmer)

# Repeat each benchmark process N times (fresh process each rep)
./gpubench run --reps 5
```

### Makefile Shortcuts

```bash
make list       # ./gpubench list
make validate   # ./gpubench validate
make run        # ./gpubench run
make info       # ./gpubench info
```

### JSON-Lines Output Protocol

Benchmarks communicate results to the CLI via **JSON-lines on stdout** — one JSON object per line. Human-readable messages go to stderr.

| Message type | Purpose |
|-------------|---------|
| `kernel` | One accumulated timing record per kernel name (`name`, `time_ms`, `params`) |
| `summary` | Sum of kernel GPU times (`total_time_ms`); optional `metrics` for legacy/higher-tier producers |
| `error` | Fatal or warning-level error report |
| `info` | Optional metadata, accepted but discarded by the parser; not stored or exported |

Minimal valid stdout for one complete `polybench_gemm` execution, with illustrative timing values and optional fields omitted (derived metrics are computed downstream):

```jsonl
{"type":"kernel","name":"polybench_gemm_kernel","time_ms":0.234,"params":{"size":256}}
{"type":"summary","total_time_ms":0.234}
```

### SQLite Results Database

Parsed run results are stored in `results.db` (override with `--db`). The schema contains five tables:

| Table | Contents |
|-------|----------|
| `platforms` | Hardware/software environment (GPU model, driver, OS, etc.) |
| `sessions` | One row per `gpubench run` invocation |
| `runs` | One row per (benchmark × parameter config × repetition) |
| `kernel_results` | One accumulated record per kernel name in each successful run |
| `metrics` | Optional legacy/higher-tier metrics supplied in `summary` events |

Query example — minimum total GPU time across fresh-process repetitions, keeping each session and complete configuration separate:

```sql
SELECT r.session_id, r.scaling_param_name, r.scaling_param_value,
       r.non_scaling_params_json, MIN(r.total_time_ms) AS min_total_ms
FROM runs r
WHERE r.benchmark = 'polybench_gemm' AND r.status = 'ok'
GROUP BY r.session_id, r.scaling_param_name, r.scaling_param_value,
         r.non_scaling_params_json
ORDER BY r.session_id, CAST(r.scaling_param_value AS INTEGER);
```

Use the selected run’s kernel breakdown when analyzing this minimum; per-kernel minima from different runs do not describe one algorithm execution. See [the schema examples](docs/schema.sql) for selecting the complete run.

### Parameterization (`params.json`)

Every benchmark directory contains a `params.json` that describes the benchmark identity, its scaling parameter (a nonempty sweep; ≥ 10 values is advisory), optional non-scaling parameters, and expected kernel names.

The CLI passes parameters to benchmarks via environment variables:

```bash
BENCH_PARAM_matrix_size=1024 BENCH_PARAM_block_size=16 ./benchmark
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All runs passed |
| 1 | Some runs failed (results still saved) |
| 2 | CLI usage error |
| 3 | No benchmarks matched filters |
| 4 | Database error |
| 130 | Interrupted (Ctrl+C) — partial results saved |

---

## Reference Documents

- [`docs/cli_design.md`](docs/cli_design.md) — Full CLI command reference
- [`docs/output_protocol.md`](docs/output_protocol.md) — JSON-lines output protocol specification
- [`docs/params_format.md`](docs/params_format.md) — `params.json` format specification
- [`docs/result_sharing.md`](docs/result_sharing.md) — Result export guide (JSON format spec)
- [`benchmark_report.md`](benchmark_report.md) — Research report: benchmark selection and rationale

---

## License

See [LICENSE](LICENSE) for details.
