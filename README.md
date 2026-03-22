# GPU Benchmarks

A comprehensive GPGPU benchmark suite covering microbenchmarks, application workloads, library kernels, alternative language implementations, and ML workloads.

---

## Overview

This suite is organized into five tiers of increasing complexity and scope, allowing researchers and engineers to evaluate GPU hardware and software stacks at every level — from raw memory bandwidth to end-to-end ML training throughput.

---

## 5-Tier Structure

| Tier | Name | Description |
|------|------|-------------|
| **Tier 1** | Microbenchmarks | Low-level hardware characterization: memory bandwidth, compute throughput (FP32/FP64/FP16/INT32), cache latency, PCIe transfer rates, atomic ops, warp shuffles |
| **Tier 2** | Application Benchmarks | Classic GPU benchmark suites: SHOC, PolyBench-GPU, Rodinia, Parboil, HeteroMark, CUDA SDK Extended, Tango, NPB-GPU, Altis, Lonestar GPU, Graph Analytics, Chai |
| **Tier 3** | Library Benchmarks | Vendor and third-party library kernels: cuBLAS, CUTLASS, cuFFT, cuSPARSE, cuDNN |
| **Tier 4** | Alternative Languages | Non-CUDA implementations: Triton, BabelStream |
| **Tier 5** | ML Workloads | End-to-end machine learning training/inference and LLM benchmarks: ResNet-50, BERT, GPT-2, ViT, NanoGPT, SDPA, torch.compile, vLLM, llama.cpp, Ollama, LLaMA, DeepSeek, Qwen, GLM, Phi, Mistral, BERT fine-tune, Whisper, Stable Diffusion, MoE, RAG, CoT, Multi-Agent, SDXL, GPT-NeoX |

---

## Directory Layout

```
gpu_benchmarks/
├── Makefile                        # Root build file
├── README.md                       # This file
├── tier1/                          # Microbenchmarks (12)
│   ├── Makefile
│   ├── memory_bandwidth/
│   ├── fp32_throughput/
│   ├── fp64_throughput/
│   ├── fp16_throughput/
│   ├── int32_throughput/
│   ├── cache_latency/
│   ├── pcie_bandwidth/
│   ├── tensor_core_throughput/
│   ├── mixbench/
│   ├── shared_mem_bandwidth/
│   ├── atomic_operations/
│   └── warp_shuffle/
├── tier2/                          # Application benchmarks (51)
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
│   ├── polybench_3mm/
│   ├── polybench_atax/
│   ├── polybench_bicg/
│   ├── polybench_fdtd2d/
│   ├── polybench_gemm/
│   ├── polybench_gramschmidt/
│   ├── polybench_jacobi2d/
│   ├── polybench_mvt/
│   ├── rodinia_backprop/
│   ├── rodinia_bfs/
│   ├── rodinia_gaussian/
│   ├── rodinia_hotspot/
│   ├── rodinia_kmeans/
│   ├── rodinia_lud/
│   ├── rodinia_nw/
│   ├── rodinia_pathfinder/
│   ├── rodinia_srad/
│   ├── parboil_histogram/
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
│   ├── npb_ep/
│   ├── npb_is/
│   ├── altis_cfd/
│   ├── altis_particlefilter/
│   ├── altis_raytracing/
│   ├── lonestar_sssp/
│   ├── lonestar_bh/
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
├── tier5/                          # ML workloads (26)
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
│   └── gptneox_train/
├── cmd/
│   └── gpubench/                   # gpubench CLI (Go)
└── docs/                          # Detailed specifications
    ├── cli_design.md
    ├── output_protocol.md
    ├── params_format.md
    └── schema.sql
```

---

## Tier 1 — Microbenchmarks

| Benchmark | Metric | Description |
|-----------|--------|-------------|
| `memory_bandwidth` | GB/s | Global memory bandwidth (H2D, D2H, D2D) |
| `fp32_throughput` | GFLOPS | Single-precision FP32 FMA throughput |
| `fp64_throughput` | GFLOPS | Double-precision FP64 FMA throughput |
| `fp16_throughput` | GFLOPS | Half-precision FP16 FMA throughput |
| `int32_throughput` | GIOPS | Integer INT32 multiply-add throughput |
| `cache_latency` | ns | L1/L2/global cache latency via pointer-chasing |
| `pcie_bandwidth` | GB/s | PCIe/Thunderbolt host-device transfer bandwidth |
| `tensor_core_throughput` | TFLOPS | Tensor core / matrix FMA throughput |
| `mixbench` | GFLOPS/GB/s | Compute-memory roofline sweep |
| `shared_mem_bandwidth` | GB/s | Shared/threadgroup memory bandwidth |
| `atomic_operations` | Gops/s | Atomic operation throughput (Add/CAS/Min) |
| `warp_shuffle` | GB/s | Warp/SIMD shuffle throughput |

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
| Rodinia | `rodinia_backprop` | Back-propagation training |
| Rodinia | `rodinia_bfs` | Breadth-first search |
| Rodinia | `rodinia_gaussian` | Gaussian elimination |
| Rodinia | `rodinia_hotspot` | Thermal simulation |
| Rodinia | `rodinia_kmeans` | K-means clustering |
| Rodinia | `rodinia_lud` | LU decomposition |
| Rodinia | `rodinia_nw` | Needleman-Wunsch sequence alignment |
| Rodinia | `rodinia_pathfinder` | Dynamic programming pathfinder |
| Rodinia | `rodinia_srad` | Speckle Reducing Anisotropic Diffusion |
| Parboil | `parboil_histogram` | Histogram computation |
| Parboil | `parboil_sgemm` | Single-precision GEMM |
| Parboil | `parboil_spmv` | Sparse matrix-vector multiply |
| Parboil | `parboil_stencil` | 3D stencil computation |
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
| Altis | `altis_cfd` | Computational Fluid Dynamics (Euler solver) |
| Altis | `altis_particlefilter` | Particle Filter (SIR Monte Carlo localization) |
| Altis | `altis_raytracing` | Simple ray tracing (Phong shading) |
| Lonestar GPU | `lonestar_sssp` | Single-Source Shortest Path (Bellman-Ford) |
| Lonestar GPU | `lonestar_bh` | Barnes-Hut N-body (2D quadtree approximation) |
| Graph Analytics | `graph_pr` | PageRank (pull-based iterative) |
| Graph Analytics | `graph_tc` | Triangle Counting (sorted adjacency intersection) |
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

---

## How to Build

### Prerequisites

- GNU Make ≥ 3.81
- **NVIDIA (CUDA):** CUDA Toolkit ≥ 11.0 (with `nvcc`) and a C++17-capable host compiler
- **AMD (ROCm):** ROCm ≥ 5.0 (with `hipcc`) installed at `/opt/rocm`
- **Apple (Metal):** Xcode ≥ 14 with Metal and MetalPerformanceShaders frameworks

### Platform Detection

The build system auto-detects the platform at build time:

| Condition | Platform selected |
|-----------|------------------|
| Running on macOS (`uname == Darwin`) | `metal` |
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
make PLATFORM=metal   # Apple GPU

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

# Full parameter sweep (scaling × non-scaling Cartesian product)
./gpubench run --sweep-non-scaling
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
| `kernel` | Per-kernel execution timing (`name`, `time_ms`, `params`) |
| `summary` | Aggregate metrics emitted at the end (`total_time_ms`, `metrics`) |
| `error` | Fatal or warning-level error report |
| `info` | Optional benchmark metadata |

Example stdout from a benchmark:

```
{"type":"kernel","name":"gemm_kernel","time_ms":0.234,"params":{"matrix_size":256,"block_size":16}}
{"type":"summary","total_time_ms":0.234,"metrics":[{"name":"GFLOPS","value":143.2}]}
```

### SQLite Results Database

All run data is stored in `results.db` (override with `--db`). The schema contains five tables:

| Table | Contents |
|-------|----------|
| `platforms` | Hardware/software environment (GPU model, driver, OS, etc.) |
| `sessions` | One row per `gpubench run` invocation |
| `runs` | One row per (benchmark × parameter config × repetition) |
| `kernel_results` | Per-kernel timing from `kernel` events |
| `metrics` | Derived metrics from `summary` events (GFLOPS, bandwidth, etc.) |

Query example — average kernel time across scaling values:

```sql
SELECT r.scaling_param_value, AVG(kr.time_ms) AS avg_ms
FROM runs r
JOIN kernel_results kr ON kr.run_id = r.id
WHERE r.benchmark = 'polybench_gemm' AND r.status = 'ok'
GROUP BY r.scaling_param_value
ORDER BY CAST(r.scaling_param_value AS INTEGER);
```

### Parameterization (`params.json`)

Every benchmark directory contains a `params.json` that describes the benchmark identity, its scaling parameter (≥ 10 values), optional non-scaling parameters, and expected kernel names.

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
- [`docs/schema.sql`](docs/schema.sql) — SQLite results database schema
- [`spec.md`](spec.md) — Project specification
- [`roadmap.md`](roadmap.md) — Development roadmap and milestones
- [`benchmark_report.md`](benchmark_report.md) — Research report: benchmark selection and rationale

---

## License

See [LICENSE](LICENSE) for details.
