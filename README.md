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
| **Tier 2** | Application Benchmarks | Classic GPU benchmark suites: SHOC, PolyBench-GPU, Rodinia, Parboil |
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
├── tier2/                          # Application benchmarks (32)
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
│   └── cuda_transpose/
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
└── scripts/                        # Helper scripts
    ├── run_all.sh
    └── report.py
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

## Benchmark Harness Usage

### Run All Benchmarks

```bash
# Run the full suite with auto-detected platform
make run-all

# Run with a specific platform
make run-all PLATFORM=cuda

# Generate a summary report from the latest results
make report
```

### Run Individual Benchmarks

Each benchmark produces a standalone executable in its own subdirectory:

```bash
cd tier1/fp32_throughput
make
./fp32_throughput
```

### CSV Output Format

All benchmarks emit results in CSV format to stdout, making it easy to collect and aggregate data:

```
benchmark,metric,value,unit
fp32_throughput,throughput,12345.6,GFLOPS
```

---

## Reference Documents

- [`spec.md`](spec.md) — Project specification
- [`roadmap.md`](roadmap.md) — Development roadmap and milestones
- [`benchmark_report.md`](benchmark_report.md) — Research report: benchmark selection and rationale

---

## License

See [LICENSE](LICENSE) for details.
