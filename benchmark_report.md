# GPGPU Benchmark Suite — Comprehensive Research Report

## Executive Summary

This report surveys the landscape of GPGPU benchmarks across five tiers: microbenchmarks, application benchmarks, library benchmarks, alternative-language benchmarks, and ML workloads. We identify **60+ individual benchmarks and benchmark suites** spanning the full GPU evaluation stack — from single-instruction latency tests to end-to-end large-language-model training.

### Key Findings

1. **Microbenchmarks (Tier 1)** are essential for understanding raw hardware capabilities. The CUDA Samples `bandwidthTest` and `deviceQuery`, combined with community tools like Mixbench and gpu-bench, can cover memory bandwidth, compute throughput, and latency with low implementation effort.

2. **Application benchmarks (Tier 2)** are well-established. Rodinia and SHOC are the most widely cited and should be included first. PolyBench/GPU and Parboil add important coverage for data-parallel patterns and scientific kernels.

3. **Library benchmarks (Tier 3)** test the optimized building blocks that production workloads depend on. cuBLAS GEMM, CUTLASS, and cuDNN convolutions are the highest priority since they dominate ML and HPC runtimes.

4. **Alternative-language benchmarks (Tier 4)** ensure vendor-neutral evaluation. Triton kernels are the highest priority due to growing adoption in the ML ecosystem. OpenCL and HIP provide cross-platform coverage; SYCL/DPC++ targets Intel GPUs.

5. **ML workloads (Tier 5)** are the primary driver of GPU demand. MLPerf Training and Inference are the gold standard. Complementing them with PyTorch's torchbenchmark and model-specific benchmarks (ResNet-50, BERT, LLaMA) provides practical, reproducible coverage.

### Top Recommendations

- **Start with Tier 1 and Tier 2** — low complexity, high value, fast feedback loop.
- **Prioritize cuBLAS GEMM and CUTLASS** in Tier 3 — they are the foundation of every higher-level ML benchmark.
- **Include Triton** in Tier 4 — it is rapidly becoming the default GPU programming model for ML researchers.
- **Use MLPerf as the Tier 5 anchor** — it is the industry standard and enables apples-to-apples comparisons.

---

## Tier 1 — Microbenchmarks

Microbenchmarks isolate individual hardware characteristics: memory bandwidth, latency, compute throughput, instruction-level behavior, and interconnect performance. They are fast, self-contained, and essential for hardware characterization.

### Benchmark Table

| # | Name | Source | Description | Complexity | Priority |
|---|------|--------|-------------|------------|----------|
| 1 | **CUDA bandwidthTest** | NVIDIA CUDA Samples | Measures host-to-device, device-to-host, and device-to-device memory bandwidth via `cudaMemcpy`. Pageable and pinned memory variants. | Low | **High** |
| 2 | **CUDA deviceQuery** | NVIDIA CUDA Samples | Enumerates GPU properties (SM count, clock rates, memory size, compute capability). Not a performance test per se but essential for baseline. | Low | **High** |
| 3 | **Global Memory Bandwidth** | Custom / CUDA Samples | Large contiguous reads/writes to global memory. Measures peak achievable bandwidth vs. theoretical. | Low | **High** |
| 4 | **Shared Memory Bandwidth** | Custom | Bank-conflict-free vs. conflicted access patterns. Measures shared memory throughput. | Low | **High** |
| 5 | **L1/L2 Cache Bandwidth & Latency** | Custom / GPU Microbenchmark Suite (Jia et al.) | Pointer-chasing and streaming tests to measure cache line sizes, hit/miss latency, and bandwidth at each cache level. | Medium | **High** |
| 6 | **Memory Latency (Global/Shared/L1/L2)** | Custom / Dissecting GPU Memory Hierarchy (Mei & Chu) | Pointer-chasing benchmarks with varying strides and working set sizes. Reveals cache hierarchy characteristics. | Medium | **High** |
| 7 | **Mixbench** | Konstantinidis & Cotronis (GitHub: ekondis/mixbench) | Parametric benchmark varying compute-to-memory ratio. Plots the roofline. Supports CUDA, OpenCL, HIP. | Low | **High** |
| 8 | **Compute Throughput (FP32)** | Custom | FMA-heavy kernels maximizing single-precision throughput. | Low | **High** |
| 9 | **Compute Throughput (FP64)** | Custom | Double-precision FMA kernels. Important for HPC workloads. | Low | **High** |
| 10 | **Compute Throughput (INT32)** | Custom | Integer arithmetic throughput. | Low | Medium |
| 11 | **Compute Throughput (FP16/BF16)** | Custom | Half-precision and bfloat16 throughput using CUDA `__half` / `__nv_bfloat16`. | Low | **High** |
| 12 | **Tensor Core Throughput** | Custom | WMMA API or mma.sync PTX to measure tensor core TOPS (FP16, INT8, TF32). | Medium | **High** |
| 13 | **Atomic Operations** | Custom | `atomicAdd`, `atomicCAS`, `atomicMin` on global and shared memory. Measures throughput and contention effects. | Low | Medium |
| 14 | **Warp Shuffle Throughput** | Custom | `__shfl_sync`, `__shfl_xor_sync` throughput and latency. | Low | Medium |
| 15 | **Occupancy Experiment** | Custom | Varies shared memory usage and register count to show occupancy effects on throughput. | Medium | Medium |
| 16 | **PCIe Bandwidth** | CUDA Samples / Custom | Uni- and bi-directional PCIe transfer bandwidth with varying transfer sizes. | Low | **High** |
| 17 | **NVLink Bandwidth** | Custom / p2pBandwidthLatencyTest (CUDA Samples) | Peer-to-peer bandwidth between GPUs over NVLink. | Low | Medium |
| 18 | **Instruction Throughput** | Custom / GPU Microbenchmark Suite | Measures per-instruction throughput for arithmetic, transcendental, type-conversion instructions. | Medium | Medium |
| 19 | **gpu_burn** | Wiliber Pol (GitHub: wiliber/gpu-burn) | Stress test that maximizes power draw and thermal load. Useful for stability and sustained performance testing. | Low | Low |

### Selection Rationale

- **Memory bandwidth and latency** are the most critical bottlenecks for GPU workloads. Items 1–6 provide comprehensive memory subsystem characterization.
- **Mixbench** is uniquely valuable because it maps the entire compute-memory spectrum, enabling roofline analysis.
- **Tensor core throughput** is essential for modern ML hardware evaluation (Volta, Ampere, Hopper, and beyond).
- **PCIe/NVLink** tests are critical for multi-GPU and distributed workloads.
- Lower-priority items (atomics, shuffles, occupancy) are still valuable for optimization work but less critical for initial hardware characterization.

---

## Tier 2 — Application Benchmarks

Application benchmarks use realistic kernels drawn from scientific computing, image processing, data mining, and physics simulation. They stress the GPU across multiple dimensions simultaneously.

### Benchmark Table

| # | Suite / Benchmark | Source | Description | Complexity | Priority |
|---|-------------------|--------|-------------|------------|----------|
| **SHOC (Scalable HeterOgeneous Computing)** | | | | | |
| 1 | SHOC — BusSpeedDownload/Upload | github.com/vetter/shoc | PCIe transfer bandwidth | Low | **High** |
| 2 | SHOC — DeviceMemory | SHOC | Global memory bandwidth (read, write, read-write) | Low | **High** |
| 3 | SHOC — MaxFlops | SHOC | Peak FLOPS (SP/DP) | Low | **High** |
| 4 | SHOC — FFT | SHOC | Fast Fourier Transform | Medium | **High** |
| 5 | SHOC — GEMM | SHOC | Dense matrix multiplication | Medium | **High** |
| 6 | SHOC — MD (Molecular Dynamics) | SHOC | Lennard-Jones force computation | Medium | Medium |
| 7 | SHOC — Reduction | SHOC | Parallel sum reduction | Low | **High** |
| 8 | SHOC — Scan | SHOC | Prefix sum / scan | Low | **High** |
| 9 | SHOC — Sort | SHOC | Radix sort | Medium | Medium |
| 10 | SHOC — Spmv | SHOC | Sparse matrix–vector multiplication | Medium | **High** |
| 11 | SHOC — Stencil2D | SHOC | 2D stencil computation | Medium | Medium |
| 12 | SHOC — Triad | SHOC | STREAM Triad (a[i] = b[i] + s*c[i]) | Low | **High** |
| **Rodinia** | | | | | |
| 13 | Rodinia — HotSpot | rodinia.cs.virginia.edu | Iterative thermal simulation on a 2D grid (structured grid) | Medium | **High** |
| 14 | Rodinia — CFD (Euler3D) | Rodinia | Computational fluid dynamics — unstructured grid solver | High | **High** |
| 15 | Rodinia — LUD | Rodinia | LU decomposition | Medium | Medium |
| 16 | Rodinia — Needleman-Wunsch | Rodinia | Dynamic programming for sequence alignment | Medium | **High** |
| 17 | Rodinia — Back Propagation | Rodinia | Neural network training (backpropagation) | Medium | Medium |
| 18 | Rodinia — SRAD | Rodinia | Speckle-Reducing Anisotropic Diffusion (image processing) | Medium | Medium |
| 19 | Rodinia — PathFinder | Rodinia | Dynamic programming on a 2D grid | Medium | Medium |
| 20 | Rodinia — BFS | Rodinia | Breadth-first search on graphs | Medium | **High** |
| 21 | Rodinia — Gaussian Elimination | Rodinia | Dense linear algebra | Medium | Medium |
| 22 | Rodinia — Leukocyte | Rodinia | Cell tracking in video — image processing + tracking | High | Medium |
| 23 | Rodinia — K-Means | Rodinia | Clustering algorithm (data mining) | Medium | Medium |
| 24 | Rodinia — StreamCluster | Rodinia | Online clustering of streaming data | High | Low |
| **PolyBench/GPU** | | | | | |
| 25 | PolyBench — 2DConvolution | web.cs.ucla.edu/~pouchet/software/polybench/ | 2D image convolution | Low | **High** |
| 26 | PolyBench — 3mm | PolyBench | Chain of 3 matrix multiplications | Medium | **High** |
| 27 | PolyBench — ATAX | PolyBench | Matrix transpose × vector | Low | Medium |
| 28 | PolyBench — BICG | PolyBench | BiCG sub-kernel of BiCGStab | Low | Medium |
| 29 | PolyBench — GEMM | PolyBench | General matrix multiplication | Low | **High** |
| 30 | PolyBench — GRAMSCHMIDT | PolyBench | Gram-Schmidt orthogonalization | Medium | Medium |
| 31 | PolyBench — MVT | PolyBench | Matrix-vector product and transpose | Low | Medium |
| 32 | PolyBench — SYR2K / SYRK | PolyBench | Symmetric rank-2k / rank-k update | Medium | Medium |
| 33 | PolyBench — FDTD-2D | PolyBench | Finite-difference time-domain (2D) | Medium | **High** |
| 34 | PolyBench — JACOBI-2D | PolyBench | 2D Jacobi stencil iteration | Medium | Medium |
| **Parboil** | | | | | |
| 35 | Parboil — SGEMM | impact.crhc.illinois.edu/parboil | Single-precision GEMM | Medium | **High** |
| 36 | Parboil — Stencil | Parboil | 3D stencil computation | Medium | **High** |
| 37 | Parboil — SpMV | Parboil | Sparse matrix–vector multiplication | Medium | **High** |
| 38 | Parboil — MRI-Q | Parboil | MRI reconstruction (non-uniform FFT) | High | Medium |
| 39 | Parboil — MRI-FHD | Parboil | MRI reconstruction (FHD computation) | High | Medium |
| 40 | Parboil — Histogramming | Parboil | Histogram computation | Low | Medium |
| 41 | Parboil — SAD | Parboil | Sum of absolute differences (video encoding) | Medium | Medium |
| 42 | Parboil — BFS | Parboil | Breadth-first search | Medium | Medium |
| 43 | Parboil — TPACF | Parboil | Two-point angular correlation function (cosmology) | High | Low |
| **Other Notable Benchmarks** | | | | | |
| 44 | CUDA Samples — matrixMul | NVIDIA CUDA Samples | Shared-memory-tiled matrix multiplication | Low | **High** |
| 45 | CUDA Samples — reduction | NVIDIA CUDA Samples | Multi-pass parallel reduction (7 optimization levels) | Low | **High** |
| 46 | CUDA Samples — transpose | NVIDIA CUDA Samples | Matrix transpose with bank-conflict avoidance | Low | **High** |
| 47 | CUDA Samples — convolutionSeparable | NVIDIA CUDA Samples | Separable 2D convolution | Medium | Medium |
| 48 | CUDA Samples — histogram | NVIDIA CUDA Samples | 64- and 256-bin histograms | Low | Medium |
| 49 | HeteroMark | Northeastern U. | Heterogeneous benchmark suite (AES, FIR, KMeans, PageRank, etc.) | Medium | Low |
| 50 | Lonestar GPU | UT Austin (ISS Group) | Irregular applications: BFS, SSSP, MST, DMR on GPUs | High | Medium |
| 51 | GraphBIG-GPU | Georgia Tech / Intel | Graph analytics: BFS, PageRank, SSSP, Triangle Counting | High | Medium |

### Selection Rationale

- **SHOC** covers both low-level (bandwidth, FLOPS) and application-level kernels. Its hierarchical structure makes it ideal for systematic evaluation.
- **Rodinia** is the single most cited GPU benchmark suite in the academic literature. HotSpot, CFD, BFS, and Needleman-Wunsch are the most widely used kernels.
- **PolyBench/GPU** provides regular, polyhedral loop nests — excellent for compiler and auto-tuning research.
- **Parboil** adds real-world scientific workloads (MRI, cosmology) with professionally maintained input datasets of varying sizes.
- **CUDA Samples** provide minimal, well-understood reference implementations of key patterns (GEMM, reduction, transpose).

---

## Tier 3 — Library Benchmarks

GPU-accelerated libraries are the performance backbone of production workloads. Benchmarking them measures the efficiency of vendor-optimized code and evaluates how well hardware executes tuned kernels.

### Benchmark Table

| # | Library / Benchmark | Source | Description | Complexity | Priority |
|---|---------------------|--------|-------------|------------|----------|
| **cuBLAS** | | | | | |
| 1 | SGEMM (FP32 GEMM) | NVIDIA cuBLAS | Single-precision general matrix multiplication. The most important single GPU benchmark for ML/HPC. | Low | **High** |
| 2 | DGEMM (FP64 GEMM) | NVIDIA cuBLAS | Double-precision GEMM. Critical for scientific computing. | Low | **High** |
| 3 | HGEMM (FP16 GEMM) | NVIDIA cuBLAS | Half-precision GEMM using tensor cores. | Low | **High** |
| 4 | Batched GEMM | NVIDIA cuBLAS | Many small GEMMs batched together. Common in attention layers. | Low | **High** |
| 5 | GEMV | NVIDIA cuBLAS | Matrix–vector multiplication. Memory-bandwidth-bound. | Low | Medium |
| 6 | AXPY | NVIDIA cuBLAS | Vector scaling and addition. Memory-bandwidth-bound. | Low | Medium |
| **CUTLASS** | | | | | |
| 7 | CUTLASS GEMM (FP32) | NVIDIA CUTLASS (GitHub: NVIDIA/cutlass) | Templated GEMM with configurable tiling, warp shape, and pipeline depth. | Medium | **High** |
| 8 | CUTLASS GEMM (FP16 + Tensor Core) | CUTLASS | FP16 GEMM using tensor core mma instructions. | Medium | **High** |
| 9 | CUTLASS GEMM (INT8 + Tensor Core) | CUTLASS | INT8 GEMM for quantized inference. | Medium | **High** |
| 10 | CUTLASS Grouped GEMM | CUTLASS | Multiple GEMMs of different sizes scheduled together. | Medium | Medium |
| 11 | CUTLASS Conv2d (Implicit GEMM) | CUTLASS | 2D convolution via implicit GEMM formulation. | Medium | **High** |
| **cuDNN** | | | | | |
| 12 | Convolution Forward | NVIDIA cuDNN | Forward convolution (various algorithms: implicit GEMM, Winograd, FFT). | Medium | **High** |
| 13 | Convolution Backward (Data + Filter) | NVIDIA cuDNN | Backward-pass gradients for training. | Medium | **High** |
| 14 | Pooling | cuDNN | Max and average pooling. | Low | Medium |
| 15 | BatchNorm | cuDNN | Batch normalization forward and backward. | Low | **High** |
| 16 | Softmax | cuDNN | Softmax forward and backward. | Low | Medium |
| 17 | RNN / LSTM | cuDNN | Recurrent neural network forward and backward. | Medium | Medium |
| 18 | Multi-Head Attention | cuDNN (v8.x+) / Flash Attention | Fused multi-head attention. Critical for Transformers. | Medium | **High** |
| **cuSPARSE** | | | | | |
| 19 | SpMV (CSR) | NVIDIA cuSPARSE | Sparse matrix–vector multiplication in CSR format. | Low | **High** |
| 20 | SpMM | cuSPARSE | Sparse × dense matrix multiplication. | Medium | Medium |
| 21 | SpGEMM | cuSPARSE | Sparse × sparse matrix multiplication. | High | Medium |
| **cuFFT** | | | | | |
| 22 | 1D FFT | NVIDIA cuFFT | Complex-to-complex and real-to-complex 1D FFT. | Low | **High** |
| 23 | 2D FFT | cuFFT | 2D FFT — common in image processing and PDE solvers. | Low | **High** |
| 24 | 3D FFT | cuFFT | 3D FFT — molecular dynamics, climate modeling. | Low | Medium |
| 25 | Batched FFT | cuFFT | Many small FFTs batched together. | Low | Medium |
| **cuRAND** | | | | | |
| 26 | PRNG Throughput | NVIDIA cuRAND | Random number generation throughput (XORWOW, MRG32k3a, Philox). | Low | Medium |
| **Other Libraries** | | | | | |
| 27 | Thrust — Sort | NVIDIA Thrust | Radix sort on GPU. | Low | Medium |
| 28 | Thrust — Reduce | Thrust | Parallel reduction. | Low | Medium |
| 29 | CUB — Device-wide primitives | NVIDIA CUB | Sort, scan, reduce, select, histogram. High-performance building blocks. | Low | Medium |
| 30 | NCCL — AllReduce | NVIDIA NCCL | Multi-GPU collective: AllReduce bandwidth and latency. Critical for distributed training. | Medium | **High** |
| 31 | Flash Attention | Dao-AILab (GitHub: Dao-AILab/flash-attention) | Fused, IO-aware attention. De facto standard for Transformer attention. | Medium | **High** |
| **AMD Equivalents** | | | | | |
| 32 | rocBLAS GEMM | AMD rocBLAS | GEMM on AMD GPUs (MI100, MI210, MI300). | Medium | Medium |
| 33 | MIOpen Convolution | AMD MIOpen | Convolution and other DNN primitives on AMD GPUs. | Medium | Medium |
| 34 | hipFFT | AMD hipFFT | FFT on AMD GPUs via HIP. | Medium | Low |

### Selection Rationale

- **cuBLAS GEMM** (items 1–4) is the single most important library benchmark — GEMM performance dominates ML training and HPC performance.
- **CUTLASS** allows fine-grained control over GEMM tiling and scheduling, making it invaluable for understanding performance sensitivity to kernel design choices.
- **cuDNN convolution and attention** directly measure the building blocks of CNNs and Transformers.
- **Flash Attention** is included because it has become the standard attention implementation in production ML frameworks (PyTorch 2.x, JAX).
- **NCCL AllReduce** is critical for multi-GPU and distributed training evaluation.
- **AMD equivalents** ensure the suite is not NVIDIA-only. They are lower priority initially but important for cross-vendor evaluation.

---

## Tier 4 — Alternative Language Benchmarks

Non-CUDA programming models are essential for portable GPU computing, vendor-neutral evaluation, and supporting emerging compiler technologies.

### Benchmark Table

| # | Language / Framework | Benchmark | Source | Description | Complexity | Priority |
|---|---------------------|-----------|--------|-------------|------------|----------|
| **Triton (OpenAI)** | | | | | | |
| 1 | Triton — GEMM | OpenAI Triton (GitHub: triton-lang/triton) | Matrix multiplication in Triton's Python DSL. Auto-tunes tile sizes. | Low | **High** |
| 2 | Triton — Softmax | Triton tutorials | Fused online softmax. | Low | **High** |
| 3 | Triton — Flash Attention | Triton tutorials / community | Attention kernel in Triton. | Medium | **High** |
| 4 | Triton — LayerNorm | Triton tutorials | Fused layer normalization. | Low | **High** |
| 5 | Triton — Quantized GEMM (INT8/INT4) | Community implementations | Quantized matrix multiplication for LLM inference. | Medium | Medium |
| 6 | Triton — Vector Addition | Triton tutorials | Simplest possible Triton kernel — useful as a baseline. | Low | **High** |
| **OpenCL** | | | | | | |
| 7 | OpenCL — clpeak | GitHub: krrishnarraj/clpeak | Auto-detects device and measures peak bandwidth, FLOPS. Lightweight. | Low | **High** |
| 8 | OpenCL — SHOC-OpenCL | SHOC (OpenCL backend) | All SHOC benchmarks compiled for OpenCL. | Medium | **High** |
| 9 | OpenCL — cl-mem | Community | OpenCL memory bandwidth benchmark. | Low | Medium |
| 10 | OpenCL — Rodinia-OpenCL | Rodinia (OpenCL ports) | Select Rodinia benchmarks compiled for OpenCL. | Medium | Medium |
| 11 | OpenCL — AMD APP SDK Samples | AMD | AMD's OpenCL sample applications (matrix transpose, bitonic sort, etc.). | Medium | Low |
| **HIP (AMD)** | | | | | | |
| 12 | HIP — SHOC-HIP | SHOC (HIP backend) | SHOC benchmarks ported to HIP. | Medium | Medium |
| 13 | HIP — Rodinia-HIP | Community ports | Rodinia benchmarks ported to HIP. | Medium | Medium |
| 14 | HIP — rocm-bandwidth-test | AMD ROCm | PCIe and XGMI bandwidth test for AMD GPUs. | Low | Medium |
| 15 | HIPified CUDA Samples | AMD hipify tools | CUDA Samples auto-converted to HIP. Tests hipification quality. | Medium | Medium |
| **SYCL / DPC++ (Intel)** | | | | | | |
| 16 | SYCL-Bench | GitHub: hipsycl/sycl-bench | Benchmark suite specifically for SYCL implementations. | Medium | Medium |
| 17 | SYCL — Rodinia-SYCL | Codeplay / community | Rodinia benchmarks ported to SYCL. | Medium | Low |
| 18 | SYCL — oneDNN benchmarks | Intel oneAPI | DNN primitives benchmarked through oneDNN on Intel GPUs. | Medium | Low |
| **Performance Portability Frameworks** | | | | | | |
| 19 | Kokkos — BabelStream | GitHub: UoB-HPC/BabelStream | STREAM benchmark in Kokkos, CUDA, OpenCL, OpenMP, SYCL, etc. Excellent for cross-platform bandwidth comparison. | Low | **High** |
| 20 | Kokkos — Kokkos Kernels Perf Tests | GitHub: kokkos/kokkos-kernels | GEMM, SpMV, graph algorithms via Kokkos Kernels. | Medium | Medium |
| 21 | RAJA — RAJA Perf Suite | GitHub: LLNL/RAJAPerf | Performance portability benchmark from LLNL. Loop-based kernels across CUDA, HIP, OpenMP. | Medium | Medium |
| **OpenMP Target Offload** | | | | | | |
| 22 | SPEC ACCEL (OpenACC/OpenMP) | SPEC | Industry-standard benchmark for accelerator offloading. Requires SPEC license. | High | Medium |
| 23 | SOLLVE V&V Test Suite | ECP SOLLVE | OpenMP 4.5/5.0 target offload validation and performance. | Medium | Low |
| 24 | OpenMP — BabelStream | BabelStream (OpenMP backend) | STREAM bandwidth in OpenMP target offload. | Low | Medium |
| **Vulkan Compute** | | | | | | |
| 25 | vkpeak | GitHub: gortali/vkpeak | Peak FLOPS and bandwidth using Vulkan compute shaders. | Low | Low |
| 26 | Vulkan Kompute | GitHub: KomputeProject/kompute | General-purpose GPU compute framework. | Medium | Low |

### Selection Rationale

- **Triton** (items 1–6) is the highest priority alternative language because it is becoming the standard for ML kernel development (used in PyTorch 2.x `torch.compile`, xformers, and many LLM projects).
- **BabelStream** (item 19) is the single best cross-platform bandwidth benchmark — it supports CUDA, OpenCL, HIP, SYCL, Kokkos, and OpenMP from a single codebase.
- **OpenCL** remains the most portable GPU API and is necessary for evaluating non-NVIDIA hardware.
- **SYCL** and **HIP** are important for Intel and AMD ecosystems respectively, but can be lower initial priority unless multi-vendor support is a goal.
- **Vulkan Compute** is included for completeness but is lowest priority since it is rarely used for HPC/ML benchmarking.

---

## Tier 5 — ML Workloads

ML workloads are the primary driver of GPU hardware demand. This tier benchmarks end-to-end training, inference, and individual model components.

### Benchmark Table

| # | Suite / Benchmark | Source | Description | Complexity | Priority |
|---|-------------------|--------|-------------|------------|----------|
| **MLPerf Training** | | | | | |
| 1 | MLPerf Training — ResNet-50 (Image Classification) | mlcommons.org/benchmarks | Training ResNet-50 on ImageNet to 75.9% top-1 accuracy. | High | **High** |
| 2 | MLPerf Training — BERT (NLP) | MLCommons | Pre-training BERT-Large (masked LM). | High | **High** |
| 3 | MLPerf Training — GPT-3 (LLM) | MLCommons | Large language model training. | High | **High** |
| 4 | MLPerf Training — RetinaNet (Object Detection) | MLCommons | Object detection on COCO dataset. | High | Medium |
| 5 | MLPerf Training — Stable Diffusion (Image Generation) | MLCommons | Text-to-image diffusion model training. | High | Medium |
| 6 | MLPerf Training — DLRM (Recommendation) | MLCommons | Deep learning recommendation model. | High | Medium |
| 7 | MLPerf Training — 3D U-Net (Medical Imaging) | MLCommons | 3D medical image segmentation. | High | Low |
| **MLPerf Inference** | | | | | |
| 8 | MLPerf Inference — ResNet-50 | MLCommons | Image classification inference (offline, server, single-stream scenarios). | Medium | **High** |
| 9 | MLPerf Inference — BERT-Large | MLCommons | NLP question-answering inference. | Medium | **High** |
| 10 | MLPerf Inference — GPT-J / LLaMA-2 70B | MLCommons | LLM inference (latency and throughput). | High | **High** |
| 11 | MLPerf Inference — Stable Diffusion | MLCommons | Image generation inference. | High | Medium |
| 12 | MLPerf Inference — RetinaNet | MLCommons | Object detection inference. | Medium | Medium |
| **PyTorch Benchmarks** | | | | | |
| 13 | torchbenchmark | GitHub: pytorch/benchmark | Collection of ~50 real models (ResNet, BERT, GPT-2, timm models, etc.) with standardized training/inference loops. | Medium | **High** |
| 14 | PyTorch — torch.compile benchmarks | PyTorch project | Measures `torch.compile` (Inductor / Triton) speedups vs. eager mode. | Medium | **High** |
| **TensorRT** | | | | | |
| 15 | TensorRT — trtexec | NVIDIA TensorRT | Command-line tool for benchmarking TensorRT engine inference (latency, throughput). | Medium | **High** |
| 16 | TensorRT — ONNX model zoo benchmarks | NVIDIA / ONNX | Benchmark popular ONNX models (ResNet, BERT, YOLOv5, etc.) through TensorRT. | Medium | **High** |
| **Distributed Training** | | | | | |
| 17 | DeepSpeed — ZeRO benchmarks | Microsoft DeepSpeed (GitHub: microsoft/DeepSpeed) | Measures training throughput with ZeRO-1/2/3 memory optimization stages. | High | **High** |
| 18 | Megatron-LM benchmarks | NVIDIA Megatron-LM (GitHub: NVIDIA/Megatron-LM) | Tensor-parallel and pipeline-parallel training throughput for GPT and BERT. | High | **High** |
| 19 | FSDP benchmarks | PyTorch FSDP | Fully Sharded Data Parallel training throughput. | High | Medium |
| **Model-Specific Benchmarks** | | | | | |
| 20 | ResNet-50 (Training) | torchvision / NVIDIA DeepLearningExamples | Single most common CNN training benchmark. Images/sec metric. | Medium | **High** |
| 21 | ResNet-50 (Inference) | TensorRT / ONNX Runtime | Inference latency and throughput (FP32, FP16, INT8). | Medium | **High** |
| 22 | BERT-Large (Fine-tuning) | HuggingFace / NVIDIA DeepLearningExamples | SQuAD fine-tuning throughput (sequences/sec). | Medium | **High** |
| 23 | GPT-2 (Training) | HuggingFace / nanoGPT | Small-to-medium LLM training. nanoGPT (Karpathy) is an excellent minimal reproduction. | Medium | **High** |
| 24 | LLaMA-2 / LLaMA-3 (Inference) | Meta / HuggingFace | LLM inference: tokens/sec, time-to-first-token, inter-token latency. | High | **High** |
| 25 | Stable Diffusion (Inference) | Stability AI / HuggingFace diffusers | Image generation latency (images/sec at a given resolution). | Medium | Medium |
| 26 | YOLOv8 (Inference) | Ultralytics | Real-time object detection inference. | Medium | Medium |
| 27 | Whisper (Inference) | OpenAI | Speech-to-text inference throughput. | Medium | Medium |
| **Inference Serving** | | | | | |
| 28 | vLLM Benchmark | GitHub: vllm-project/vllm | LLM serving throughput and latency with PagedAttention. | High | **High** |
| 29 | Triton Inference Server — perf_analyzer | NVIDIA Triton Inference Server | End-to-end inference serving performance (latency, throughput, concurrency). | High | Medium |
| 30 | TGI (Text Generation Inference) | HuggingFace | LLM serving benchmark. | High | Medium |

### Selection Rationale

- **MLPerf** (items 1–12) is the industry gold standard for ML hardware benchmarking. It provides reproducible, audited results. ResNet-50, BERT, and GPT-3/LLaMA are the three most important workloads.
- **torchbenchmark** (item 13) provides breadth — ~50 real models with minimal setup. Excellent for compiler and framework evaluation.
- **TensorRT + trtexec** (items 15–16) represents peak optimized inference and is the standard for NVIDIA inference evaluation.
- **DeepSpeed and Megatron-LM** (items 17–18) are essential for evaluating multi-GPU training efficiency — the most commercially important GPU use case today.
- **vLLM** (item 28) is included because LLM serving is the fastest-growing GPU workload and vLLM's PagedAttention is becoming the standard serving engine.
- **nanoGPT** (item 23) is an excellent pedagogical and benchmarking tool — a complete GPT-2 training loop in ~300 lines of PyTorch.

---

## Implementation Strategy

### Recommended Phasing

| Phase | Tiers | Focus | Estimated Effort | Rationale |
|-------|-------|-------|-----------------|-----------|
| **Phase 1** | Tier 1 + Tier 2 (partial) | Memory BW, compute throughput, Mixbench, CUDA Samples (GEMM, reduction, transpose), SHOC core kernels | 2–3 weeks | Fastest to implement. Self-contained CUDA code. Establishes baseline. |
| **Phase 2** | Tier 2 (full) + Tier 3 (partial) | Rodinia (top 5), PolyBench (top 5), cuBLAS GEMM, CUTLASS GEMM | 2–3 weeks | Application-level coverage + library baselines. |
| **Phase 3** | Tier 3 (full) + Tier 4 (Triton) | cuDNN conv/attention, Flash Attention, NCCL, Triton GEMM/softmax/attention | 2–3 weeks | Complete library coverage + most important alternative language. |
| **Phase 4** | Tier 4 (remaining) + Tier 5 (partial) | OpenCL, HIP, BabelStream, ResNet-50 training/inference, BERT, nanoGPT | 3–4 weeks | Cross-platform coverage + key ML workloads. |
| **Phase 5** | Tier 5 (full) | MLPerf workloads, vLLM, DeepSpeed, Megatron-LM, TensorRT | 3–4 weeks | End-to-end ML evaluation. Requires larger infrastructure (datasets, multi-GPU). |

### Implementation Approach

1. **Unified harness**: Create a common runner script that handles compilation, execution, metric collection, and output formatting for all benchmarks.
2. **Containerization**: Use Docker containers (NVIDIA NGC base images) to manage dependencies.
3. **Metric standardization**: All benchmarks should report: throughput (FLOPS or items/sec), latency (ms), memory bandwidth utilization (%), and power consumption (watts, if accessible via NVML).
4. **Dataset management**: Tier 5 benchmarks require large datasets (ImageNet, C4, COCO). Use pre-built dataset containers or synthetic data where possible.
5. **Multi-GPU support**: Tier 3 (NCCL) and Tier 5 (DeepSpeed, Megatron) require multi-GPU setups. Design the harness to detect and use available GPUs.

### Build System

- **Tier 1–2**: CMake + CUDA. Self-contained, no external dependencies beyond CUDA toolkit.
- **Tier 3**: Link against cuBLAS, cuDNN, cuFFT (shipped with CUDA). CUTLASS is header-only. Flash Attention requires PyTorch.
- **Tier 4**: OpenCL requires OpenCL headers/ICD loader. HIP requires ROCm. Triton requires Python + `triton` pip package. SYCL requires Intel oneAPI or hipSYCL.
- **Tier 5**: Python + PyTorch + HuggingFace ecosystem. Use pip/conda environments.

---

## Missing Gaps and Integration Challenges

### Difficult to Integrate

| Benchmark / Suite | Challenge |
|-------------------|-----------|
| **SPEC ACCEL** | Requires commercial SPEC license. Cannot redistribute. |
| **MLPerf (full compliance)** | Full compliance runs require specific dataset preparation, strict rules, and review submission. Using reference implementations for non-submission benchmarking is feasible. |
| **Proprietary vendor tools** (e.g., NVIDIA Nsight Compute automated benchmarks) | Tied to specific tool versions. Not portable. |
| **FPGA comparisons** (Xilinx Vitis benchmarks) | Out of scope but occasionally relevant for heterogeneous computing comparisons. |
| **AMD MI300X benchmarks** | Hardware access required. rocBLAS/MIOpen benchmarks are software-ready but untestable without AMD hardware. |
| **Apple Metal / M-series GPU benchmarks** | Completely different API (Metal). Out of scope unless explicitly requested. |
| **Wafer-scale benchmarks** (Cerebras) | Not a GPU — out of scope. |

### Known Gaps in Coverage

1. **Graph neural networks (GNNs)**: PyG (PyTorch Geometric) and DGL have their own benchmark suites. These are increasingly important but not covered above.
2. **Sparse training**: Benchmarks for structured/unstructured sparsity (e.g., NVIDIA Sparse Tensor Core utilization) are emerging but not yet standardized.
3. **Mixture-of-Experts (MoE)**: MoE models (e.g., Mixtral) require specialized all-to-all communication benchmarks. Coverage is thin.
4. **Memory-bound LLM inference**: Benchmarks focused specifically on KV-cache management, speculative decoding, and quantized inference (GPTQ, AWQ, GGML) are rapidly evolving.
5. **Multi-node distributed**: Benchmarks crossing node boundaries (NCCL over InfiniBand, RoCE) require cluster infrastructure.

---

## References

### Tier 1 — Microbenchmarks
- NVIDIA CUDA Samples: https://github.com/NVIDIA/cuda-samples
- Mixbench: https://github.com/ekondis/mixbench
- Dissecting GPU Memory Hierarchy through Microbenchmarking: Mei, X. & Chu, X. (IEEE TPDS, 2017)
- Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking: Jia, Z. et al. (arXiv:1804.06826)
- gpu-burn: https://github.com/wiliber/gpu-burn

### Tier 2 — Application Benchmarks
- SHOC Benchmark Suite: https://github.com/vetter/shoc
- Rodinia Benchmark Suite: https://rodinia.cs.virginia.edu/
- PolyBench/GPU: https://web.cs.ucla.edu/~pouchet/software/polybench/GPU/
- Parboil Benchmark Suite: http://impact.crhc.illinois.edu/parboil/parboil.aspx
- HeteroMark: Chai, S. et al. (IISWC 2016)
- Lonestar GPU: https://iss.oden.utexas.edu/?p=projects/galois/lonestargpu

### Tier 3 — Library Benchmarks
- cuBLAS: https://developer.nvidia.com/cublas
- CUTLASS: https://github.com/NVIDIA/cutlass
- cuDNN: https://developer.nvidia.com/cudnn
- cuSPARSE: https://developer.nvidia.com/cusparse
- cuFFT: https://developer.nvidia.com/cufft
- cuRAND: https://developer.nvidia.com/curand
- Flash Attention: https://github.com/Dao-AILab/flash-attention
- NCCL: https://github.com/NVIDIA/nccl
- rocBLAS: https://github.com/ROCm/rocBLAS
- MIOpen: https://github.com/ROCm/MIOpen
- Thrust / CUB: https://github.com/NVIDIA/cccl

### Tier 4 — Alternative Languages
- Triton: https://github.com/triton-lang/triton
- BabelStream: https://github.com/UoB-HPC/BabelStream
- clpeak: https://github.com/krrishnarraj/clpeak
- SYCL-Bench: https://github.com/hipsycl/sycl-bench (now AdaptiveCpp)
- Kokkos: https://github.com/kokkos/kokkos
- RAJA Performance Suite: https://github.com/LLNL/RAJAPerf
- SPEC ACCEL: https://www.spec.org/accel/

### Tier 5 — ML Workloads
- MLPerf Training: https://mlcommons.org/benchmarks/training/
- MLPerf Inference: https://mlcommons.org/benchmarks/inference/
- torchbenchmark: https://github.com/pytorch/benchmark
- TensorRT: https://developer.nvidia.com/tensorrt
- DeepSpeed: https://github.com/microsoft/DeepSpeed
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- nanoGPT: https://github.com/karpathy/nanoGPT
- vLLM: https://github.com/vllm-project/vllm
- NVIDIA DeepLearningExamples: https://github.com/NVIDIA/DeepLearningExamples
- HuggingFace Transformers: https://github.com/huggingface/transformers

---

*Report generated for the GPGPU Benchmark Suite project. This document should be used as the foundation for benchmark selection and implementation planning.*
