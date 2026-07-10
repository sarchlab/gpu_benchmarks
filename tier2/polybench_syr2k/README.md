# tier2/polybench_syr2k — Symmetric Rank-2k Update (PolyBench SYR2K)

Benchmarks the symmetric rank-2k matrix operation: **C = α·A·Bᵀ + α·B·Aᵀ + β·C** where A and B are N×M single-precision matrices and C is an N×N symmetric output matrix. Emits per-iteration kernel times; GFLOPS is derived downstream.

Derived from the [PolyBench/GPU benchmark suite](https://sourceforge.net/projects/polybench/) SYR2K workload (BLAS Level 3).

## Algorithm

- Symmetric rank-2k update combining two transposed matrix products
- `C[i][j] = alpha * sum_k(A[i][k]*B[j][k]) + alpha * sum_k(B[i][k]*A[j][k]) + beta * C[i][j]`
- Default parameters: N=1024, M=1024, alpha=1.5, beta=1.2
- Tiled shared-memory kernel: 16×16 thread tiles with 4 shared-memory arrays (A\_row, B\_row, A\_col, B\_col) for efficient data reuse
- GFLOPS = 4·N²·M / time\_sec / 1e9 (derived downstream from the emitted times and params)

## Files

| File | Description |
|------|-------------|
| `polybench_syr2k.hip` | HIP source — tiled SYR2K kernel with HIP/CUDA compat layer |
| `polybench_syr2k.metal` | Metal compute shader — tiled SYR2K using threadgroup memory |
| `polybench_syr2k_metal.mm` | ObjC++ Metal host code |
| `Makefile` | Auto-detecting build (Darwin→metal, ROCm→rocm, else→cuda) |
| `params.json` | Parameter specification for the benchmark harness |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
make PLATFORM=metal
```

## Run

```bash
./polybench_syr2k                     # defaults: N=1024, M=1024
./polybench_syr2k --size 2048
./polybench_syr2k --size 512 --inner_size 256
```

Environment variables override CLI flags:
```bash
BENCH_PARAM_matrix_size=2048 BENCH_PARAM_inner_size=512 ./polybench_syr2k
```

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_matrix_size` | 1024 | Output matrix dimension N |
| `BENCH_PARAM_inner_size` | 1024 | Inner dimension M |
| `BENCH_PARAM_block_size` | 256 | Threads per block (echoed in params) |
| `BENCH_PARAM_alpha` | 1.5 | Alpha scaling factor |
| `BENCH_PARAM_beta` | 1.2 | Beta scaling factor |
| `BENCH_PARAM_warmup` | 3 | Untimed warmup launches |
| `BENCH_PARAM_iteration` | 7 | Timed launches, each recorded individually |

## Output

**stdout** — JSON-lines, one kernel record per timed iteration with GPU event
time (`time_ms`) and host wall-clock time (`host_time_ms`); the summary's
`total_time_ms` is the sum of all iterations' GPU event times. GFLOPS is
derived downstream:
```json
{"type":"kernel","name":"syr2k_kernel","iter":0,"time_ms":12.345678,"host_time_ms":12.401234,"params":{"matrix_size":1024,"inner_size":1024,"block_size":256,"alpha":1.50,"beta":1.20}}
...
{"type":"summary","total_time_ms":86.419746}
```

**stderr** — human-readable results:
```
Device: NVIDIA A100 (id=0)
SYR2K: N=1024, M=1024  |  alpha=1.5  beta=1.2

syr2k_kernel iter 0: gpu = 12.346 ms, host = 12.401 ms
...
Total GPU time: 86.420 ms over 7 iterations
```

## Platform Notes

- **CUDA**: compiled with `nvcc` from `polybench_syr2k.hip`
- **ROCm/HIP**: compiled with `hipcc` from `polybench_syr2k.hip`; includes CUDA compat layer for `nvcc` fallback
- **Metal**: compiled with `clang++` from `polybench_syr2k_metal.mm`; the Metal shader (`polybench_syr2k.metal`) is loaded and compiled at runtime from the same directory as the binary

## Differences from polybench_gemm

| Property | polybench_gemm | polybench_syr2k |
|----------|---------------|-----------------|
| Operation | C = α·A·B + β·C | C = α·A·Bᵀ + α·B·Aᵀ + β·C |
| Matrix shapes | A,B,C all N×N | A,B are N×M; C is N×N |
| Kernel complexity | 1 matrix product | 2 transposed products |
| Shared memory tiles | 2 (sA, sB) | 4 (sA\_row, sB\_row, sA\_col, sB\_col) |
| FLOPs formula | 2·N³ | 4·N²·M |
