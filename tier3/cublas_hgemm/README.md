# tier3/cublas_hgemm — FP16 GEMM via cuBLAS / rocBLAS / MPS

A multi-platform benchmark for half-precision (FP16) general matrix multiplication (GEMM),
wrapping vendor-optimised BLAS libraries:

| Platform | Library                          |
|----------|----------------------------------|
| NVIDIA   | **cuBLAS** (`cublasHgemm`)        |
| AMD      | **rocBLAS** (`rocblas_hgemm`)     |
| Apple    | **Metal Performance Shaders** (`MPSMatrixMultiplication`) |

## Algorithm

Computes **C = α A × B + β C** where:

- A is M × K (row-major, FP16)
- B is K × N (row-major, FP16)
- C is M × N (row-major, FP16)
- M = N = K = **4096**
- α = 1.0, β = 0.0

### Row-major ↔ Column-major trick (CUDA / ROCm)

cuBLAS and rocBLAS use **column-major** order internally.
Row-major `C = A·B` is equivalent to column-major `Cᵀ = Bᵀ·Aᵀ`.
This means we call:

```
cublasHgemm(handle, OP_N, OP_N, N, M, K, &alpha, dB, ldb=N, dA, lda=K, &beta, dC, ldc=N)
```

which computes the column-major product `Cᵀ = Bᵀ·Aᵀ` and writes the result
into dC — which, when read back in row-major, gives the correct C = A·B.

For Apple Metal, `MPSMatrixMultiplication` natively uses row-major layout, so no
transposition trick is needed.

## Files

| File                      | Description                                      |
|---------------------------|--------------------------------------------------|
| `cublas_hgemm.hip`        | CUDA/ROCm source (`#ifdef USE_CUBLAS` / `#elif defined(USE_ROCBLAS)`) |
| `cublas_hgemm_metal.mm`   | Objective-C++ Metal host using `MPSMatrixMultiplication` |
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
./cublas_hgemm
```

## Output Format

The benchmark emits two kinds of output:

**CSV (stdout)** — machine-parseable:

```
cublas_hgemm,4096,<time_ms>,<TFLOPS>
```

Example:

```
cublas_hgemm,4096,6.123,22.44
```

**Human-readable (stderr):**

```
Device: Apple M2 Pro
M=4096  N=4096  K=4096  warmup=5  iters=5

  iter 0: 6.101 ms
  iter 1: 6.145 ms
  ...
Verification (first 8×8 corner): PASS

cublas_hgemm  M=4096 N=4096 K=4096  time_ms=6.123  TFLOPS=22.44
```

## Performance Metric

```
TFLOPS = 2 × M × N × K / time_s / 1e12
```

The factor of 2 accounts for one multiply and one add per element of the
inner-product accumulation.

## Verification

After the timed run, the first 8 × 8 = 64 elements of the GPU result C are
compared against a CPU reference GEMM. The CPU reference converts FP16 inputs
to FP32 for accumulation, then compares against the FP16 GPU output converted
back to FP32. The relative error threshold is 5e-2, relaxed compared to FP32
to account for half-precision quantization.

The benchmark exits with status 0 on pass and 1 on failure.

## Timing

- **5 warmup** iterations (discarded)
- **5 timed** iterations; the reported time is the average

On CUDA/ROCm, timing uses `cudaEvent` / `hipEvent` (GPU-side timer).
On Metal, timing uses `mach_absolute_time()` around `waitUntilCompleted`.
