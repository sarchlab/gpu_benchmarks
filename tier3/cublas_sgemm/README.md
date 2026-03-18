# tier3/cublas_sgemm — FP32 GEMM via cuBLAS / rocBLAS / MPS

A multi-platform benchmark for single-precision (FP32) general matrix multiplication (GEMM),
wrapping vendor-optimised BLAS libraries:

| Platform | Library                          |
|----------|----------------------------------|
| NVIDIA   | **cuBLAS** (`cublasSgemm`)        |
| AMD      | **rocBLAS** (`rocblas_sgemm`)     |
| Apple    | **Metal Performance Shaders** (`MPSMatrixMultiplication`) |

## Algorithm

Computes **C = A × B** where:

- A is M × K (row-major, FP32)
- B is K × N (row-major, FP32)
- C is M × N (row-major, FP32)
- M = N = K = **4096**

### Row-major ↔ Column-major trick (CUDA / ROCm)

cuBLAS and rocBLAS use **column-major** order internally.
Row-major `C = A·B` is equivalent to column-major `Cᵀ = Bᵀ·Aᵀ`.
This means we call:

```
cublasSgemm(handle, OP_N, OP_N, N, M, K, &alpha, dB, ldb=N, dA, lda=K, &beta, dC, ldc=N)
```

which computes the column-major product `Cᵀ = Bᵀ·Aᵀ` and writes the result
into dC — which, when read back in row-major, gives the correct C = A·B.

For Apple Metal, `MPSMatrixMultiplication` natively uses row-major layout, so no
transposition trick is needed.

## Files

| File                      | Description                                      |
|---------------------------|--------------------------------------------------|
| `cublas_sgemm.hip`        | CUDA/ROCm source (`#ifdef USE_CUBLAS` / `#elif defined(USE_ROCBLAS)`) |
| `cublas_sgemm_metal.mm`   | Objective-C++ Metal host using `MPSMatrixMultiplication` |
| `Makefile`                | Auto-detecting build system                      |
| `README.md`               | This file                                        |

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
./cublas_sgemm
```

## Output Format

The benchmark emits two kinds of output:

**CSV (stdout)** — machine-parseable:

```
cublas_sgemm,4096,<time_ms>,<GFLOPS>
```

Example:

```
cublas_sgemm,4096,12.345,22180.43
```

**Human-readable (stderr):**

```
Device: Apple M2 Pro
M=4096  N=4096  K=4096  warmup=5  iters=5

  iter 0: 12.301 ms
  iter 1: 12.389 ms
  ...
Verification (first 64 elements): PASS

cublas_sgemm  M=4096 N=4096 K=4096  time_ms=12.345  GFLOPS=22180.43
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
On Metal, timing uses `mach_absolute_time()` around `waitUntilCompleted`.
