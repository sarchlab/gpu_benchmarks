# tier3/cublas_batched_gemm — Batched FP32 GEMM via cuBLAS / rocBLAS / MPS

A multi-platform benchmark for batched single-precision (FP32) general matrix multiplication (GEMM),
wrapping vendor-optimised BLAS libraries:

| Platform | Library                          |
|----------|----------------------------------|
| NVIDIA   | **cuBLAS** (`cublasSgemmBatched`) |
| AMD      | **rocBLAS** (`rocblas_sgemm_batched`) |
| Apple    | **Metal Performance Shaders** (`MPSMatrixMultiplication`, batched encodes) |

## Algorithm

Computes **C[i] = α × A[i] × B[i] + β × C[i]** for i in [0, batch\_count), where:

- Each A[i] is M × K (row-major, FP32)
- Each B[i] is K × N (row-major, FP32)
- Each C[i] is M × N (row-major, FP32)
- batch\_count = **64**, M = N = K = **512**
- α = 1.0, β = 0.0

### Row-major ↔ Column-major trick (CUDA / ROCm)

cuBLAS and rocBLAS use **column-major** order internally.
Row-major `C = A·B` is equivalent to column-major `Cᵀ = Bᵀ·Aᵀ`.
This means we call:

```
cublasSgemmBatched(handle, OP_N, OP_N, N, M, K, &alpha, dBptrs, N, dAptrs, K, &beta, dCptrs, N, batch_count)
```

which computes the column-major product `Cᵀ = Bᵀ·Aᵀ` for each batch and writes the result
into dC — which, when read back in row-major, gives the correct C = A·B.

For Apple Metal, `MPSMatrixMultiplication` natively uses row-major layout, so no
transposition trick is needed. All batch\_count multiplications are encoded into a
single command buffer for efficient execution.

## Files

| File                              | Description                                      |
|-----------------------------------|--------------------------------------------------|
| `cublas_batched_gemm.hip`         | CUDA/ROCm source (`#ifdef USE_CUBLAS` / `#elif defined(USE_ROCBLAS)`) |
| `cublas_batched_gemm_metal.mm`    | Objective-C++ Metal host using `MPSMatrixMultiplication` |
| `Makefile`                        | Auto-detecting build system                      |
| `README.md`                       | This file                                        |

## Build

Auto-detect (recommended):

```bash
make
```

Explicit platform:

```bash
make PLATFORM=cuda    # NVIDIA CUDA + cuBLAS
make PLATFORM=rocm    # AMD ROCm/HIP + rocBLAS
make PLATFORM=metal   # Apple Metal + MPS
```

Clean:

```bash
make clean
```

## Run

```bash
./cublas_batched_gemm
```

## Output Format

The benchmark emits two kinds of output:

**CSV (stdout)** — machine-parseable:

```
cublas_batched_gemm,64x512,<time_ms>,<GFLOPS>
```

Example:

```
cublas_batched_gemm,64x512,8.456,2038.50
```

**Human-readable (stderr):**

```
Device: Apple M2 Pro
batch_count=64  M=512  N=512  K=512  warmup=5  iters=5

  iter 0: 8.401 ms
  iter 1: 8.489 ms
  ...
Verification (first 8×8 corner of batch 0): PASS

cublas_batched_gemm  batch=64 M=512 N=512 K=512  time_ms=8.456  GFLOPS=2038.50
```

## Performance Metric

```
GFLOPS = batch_count × 2 × M × N × K / time_s / 1e9
```

The factor of 2 accounts for one multiply and one add per element of the
inner-product accumulation. The `batch_count` multiplier reflects the total
work across all batches.

## Verification

After the timed run, the first 8 × 8 = 64 elements of the GPU result C[0]
(the first batch) are compared against a CPU reference GEMM computed over the
same corner of A[0] and B[0]. The relative error threshold is 1e-3.

The benchmark exits with status 0 on pass and 1 on failure.

## Timing

- **5 warmup** iterations (discarded)
- **5 timed** iterations; the reported time is the average

On CUDA/ROCm, timing uses `cudaEvent` / `hipEvent` (GPU-side timer).
On Metal, timing uses `mach_absolute_time()` around `waitUntilCompleted`.
