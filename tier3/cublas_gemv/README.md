# cublas_gemv — FP32 GEMV Benchmark

Benchmark for General Matrix-Vector Multiplication (GEMV): **y = A · x**

- **A**: M×N matrix (row-major, FP32), M = N = 4096
- **x**: N-element input vector
- **y**: M-element output vector

Wraps cuBLAS `cublasSgemv` (NVIDIA) and rocBLAS `rocblas_sgemv` (AMD),
both using the same FP32 GEMV operation.

---

## Files

| File | Description |
|------|-------------|
| `cublas_gemv.hip` | HIP/CUDA source; selects cuBLAS or rocBLAS at compile time via `USE_CUBLAS` / `USE_ROCBLAS` |
| `Makefile` | Auto-detects `/opt/rocm`→rocm, else→cuda |
| `README.md` | This file |

---

## Build

Auto-detect platform:
```bash
make
```

### NVIDIA CUDA
```bash
make PLATFORM=cuda
# Requires: nvcc, CUDA toolkit, cuBLAS
```

### AMD ROCm/HIP
```bash
make PLATFORM=rocm
# Requires: hipcc, ROCm, rocBLAS
```

---

## Run

```bash
./cublas_gemv
```

---

## Output Format

**CSV (stdout):**
```
cublas_gemv,<M>,<time_ms>,<GBs>,<GFLOPS>
```

Example:
```
cublas_gemv,4096,0.3512,189.72,95.36
```

**Human-readable (stderr):**
```
Device: NVIDIA A100-SXM4-80GB (id=0)
M=4096  N=4096  warmup=5  iters=5

  iter 0: 0.3498 ms
  iter 1: 0.3512 ms
  ...
Verification (all 4096 elements): PASS

GB/s: 189.72  GFLOPS: 95.36  (avg 0.3512 ms, ...)
cublas_gemv  M=4096 N=4096  time_ms=0.3512  GB/s=189.72  GFLOPS=95.36
```

---

## Performance Metrics

### GFLOPS
```
GFLOPS = 2 * M * N / time_s / 1e9
```
Each output element requires N multiply-add operations → 2·M·N FLOPs total.

### GB/s
```
GB/s = (M*N + M + N) * sizeof(float) / time_s / 1e9
```
Reads: matrix A (M·N floats) + vector x (N floats); Writes: vector y (M floats).

---

## Verification

The GPU result is compared against a CPU reference GEMV (naive double loop).
All M elements are checked; the benchmark fails if any element exceeds
relative error `1e-3` (threshold = `|ref| + 1e-6`). A threshold of 1e-3
is used (matching the cublas_sgemm benchmark) because FP32 accumulation over
N=4096 elements with different GPU/CPU ordering can produce ~1e-4 differences.

---

## Row-Major / Column-Major Trick (cuBLAS / rocBLAS)

cuBLAS and rocBLAS assume **column-major** storage.
For a row-major A (M×N), the memory layout looks like a column-major matrix of shape (N×M).
To compute `y = A_rm * x`, we use the transpose operation:

```
cublasSgemv(handle, CUBLAS_OP_T, N, M, &alpha, dA, N, dx, 1, &beta, dy, 1)
```

This computes `y = (A_cm)^T * x = A_rm * x` correctly.

---

## Parameters

| Parameter | Value |
|-----------|-------|
| M (rows)  | 4096  |
| N (cols)  | 4096  |
| Warmup iterations | 5 |
| Timed iterations  | 5 |
| Data type | FP32  |
