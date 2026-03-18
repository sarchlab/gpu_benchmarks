# tier3/cublas_dgemm — FP64 GEMM Benchmark

FP64 (double-precision) general matrix-matrix multiplication benchmark.

- **NVIDIA**: wraps `cublasDgemm` from cuBLAS
- **AMD**: wraps `rocblas_dgemm` from rocBLAS
- **Apple**: uses `cblas_dgemm` from the Accelerate framework (CPU-only — Apple Metal has no FP64 support)

## Parameters

| Parameter | Value |
|-----------|-------|
| M, N, K   | 4096  |
| Data type | `double` (FP64) |
| Warmup    | 5 iterations (not timed) |
| Timed     | 5 iterations (averaged) |

## Files

| File | Description |
|------|-------------|
| `cublas_dgemm.hip` | HIP/CUDA source; uses `#ifdef USE_CUBLAS` / `#elif defined(USE_ROCBLAS)` |
| `cublas_dgemm_metal.mm` | ObjC++ host using `cblas_dgemm` from Accelerate (CPU-only) |
| `Makefile` | Auto-detects Darwin→metal, `/opt/rocm`→rocm, else→cuda |
| `README.md` | This file |

## Build & Run

### NVIDIA (CUDA)

```bash
make PLATFORM=cuda
./cublas_dgemm
```

### AMD (ROCm/HIP)

```bash
make PLATFORM=rocm
./cublas_dgemm
```

### Apple (Accelerate cblas_dgemm — CPU-only)

```bash
make PLATFORM=metal
./cublas_dgemm
```

> **Note:** Apple Metal does not support FP64 compute. The Metal build uses
> `cblas_dgemm` from Apple's Accelerate framework, which runs on the CPU with
> highly optimized BLAS routines. There is no GPU acceleration on this platform
> for this benchmark.

### Auto-detect

```bash
make        # detects Darwin → metal, /opt/rocm present → rocm, else → cuda
./cublas_dgemm
```

## Output

```
cublas_dgemm,4096,<time_ms>,<GFLOPS>
```

Additional human-readable output goes to `stderr`.

## Performance Formula

```
GFLOPS = 2 × M × N × K / (time_s × 10⁹)
```

For M=N=K=4096 this is ≈ 137.4 GFLOPs of FP64 work per call.

## Verification

The first 8×8 corner of the result matrix C is compared against a simple
triple-loop CPU DGEMM. For FP64, relative error must be below 1×10⁻⁹.
Verification result is reported to `stderr` as `PASS` or `FAIL`.

## Row-Major BLAS Trick

cuBLAS and rocBLAS use column-major order internally. To compute row-major
`C = A × B` we use the identity:

```
C^T = B^T × A^T
```

and call the BLAS routine with swapped arguments and dimensions `(N, M, K)`:

```c
cublasDgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    N, M, K,
    &alpha, dB, N,   /* B^T with leading dim N */
            dA, K,   /* A^T with leading dim K */
    &beta,  dC, N);  /* C^T with leading dim N */
```
