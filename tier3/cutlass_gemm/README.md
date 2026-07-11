# tier3/cutlass_gemm — CUTLASS-style Tiled FP32 GEMM

A multi-platform benchmark for single-precision (FP32) general matrix multiplication (GEMM)
using a **custom tiled kernel** with shared-memory staging and register tiling.
Self-contained — no external library dependencies.

| Platform | Runtime        | Kernel                               |
|----------|----------------|--------------------------------------|
| NVIDIA   | **CUDA**       | Custom HIP kernel (compiled via nvcc) |
| AMD      | **ROCm/HIP**   | Custom HIP kernel (compiled via hipcc) |

## Algorithm

Computes **C = A × B** where:

- A is M × K (row-major, FP32)
- B is K × N (row-major, FP32)
- C is M × N (row-major, FP32)
- M = N = K = **4096**

### Tiled GEMM with Register Blocking

This is a **CUTLASS-style** implementation that demonstrates high-performance
GPU GEMM optimization without relying on vendor libraries:

1. **Block tiling**: Each thread block computes a **BM × BN = 64 × 64** tile of C.
2. **K-dimension tiling**: The K-dimension is traversed in tiles of **BK = 8**.
3. **Shared-memory staging**: Each K-tile loads BM × BK elements of A and BK × BN
   elements of B into shared memory cooperatively.
4. **Register tiling**: Each thread computes a **TM × TN = 8 × 8** sub-tile in
   registers using outer-product accumulation over the BK dimension.

Thread block layout:
- Threads per block: **(BM/TM) × (BN/TN) = 8 × 8 = 64**
- Grid: **(N/BN, M/BM) = (64, 64)** for M=N=4096

Shared memory per block:
- smA: BM × BK × 4 bytes = **2 KB**
- smB: BK × BN × 4 bytes = **2 KB**
- Total: **4 KB**

## Files

| File                      | Description                                      |
|---------------------------|--------------------------------------------------|
| `cutlass_gemm.hip`        | CUDA/ROCm source (`#ifdef USE_CUDA` / `#elif defined(USE_HIP)`) |
| `Makefile`                | Auto-detecting build system                      |
| `README.md`               | This file                                        |

## Build

Auto-detect (recommended):

```bash
make
```

Explicit platform:

```bash
make PLATFORM=cuda    # NVIDIA CUDA
make PLATFORM=rocm    # AMD ROCm/HIP
```

Clean:

```bash
make clean
```

## Run

```bash
./cutlass_gemm
```

## Output Format

The benchmark emits two kinds of output:

**CSV (stdout)** — machine-parseable:

```
cutlass_gemm,4096,<time_ms>,<GFLOPS>
```

Example:

```
cutlass_gemm,4096,45.678,3009.12
```

**Human-readable (stderr):**

```
Device: <device name>
M=4096  N=4096  K=4096  BM=64 BN=64 TM=8 TN=8
threads/group=64  warmup=5  iters=5

  iter 0: 45.301 ms
  iter 1: 46.123 ms
  ...
Verification (first 8×8 corner): PASS

cutlass_gemm  M=4096 N=4096 K=4096  time_ms=45.678  GFLOPS=3009.12
```

## Performance Metric

```
GFLOPS = 2 × M × N × K / time_s / 1e9
```

The factor of 2 accounts for one multiply and one add per element of the
inner-product accumulation.

## Verification

After the timed run, the first 8 × 8 = 64 elements of the GPU result C are
compared against a CPU reference GEMM computed over the same corner of A and B.
The relative error threshold is 1e-3.

The benchmark exits with status 0 on pass and 1 on failure.

## Timing

- **5 warmup** iterations (discarded)
- **5 timed** iterations; the reported time is the average

On CUDA/ROCm, timing uses `cudaEvent` / `hipEvent` (GPU-side timer).

## Key Differences from cublas_sgemm

| Aspect          | cublas_sgemm                | cutlass_gemm                   |
|-----------------|-----------------------------|--------------------------------|
| Kernel          | Vendor library (cuBLAS/etc) | Custom tiled kernel            |
| Dependencies    | cuBLAS / rocBLAS            | None (self-contained)          |
| Optimization    | Vendor-tuned                | Demonstrates tiling principles |
| Expected perf   | Near peak                   | Good but below vendor          |
