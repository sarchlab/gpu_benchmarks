# tier2/parboil_sgemm — Tiled Single-Precision GEMM (Parboil)

Benchmarks tiled single-precision matrix multiplication: **C = α·A·B + β·C** for square N×N matrices. Emits raw per-iteration kernel times; GFLOPS is derived downstream.

Derived from the [Parboil benchmark suite](http://impact.crhc.illinois.edu/parboil/parboil.aspx) SGEMM workload.

## Algorithm

- Square N×N single-precision matrix multiply: **C = alpha·A·B + beta·C**
- Default: N=1024, alpha=1.5, beta=1.2
- Tiled kernel: 16×16 thread tiles load sub-matrices into shared/threadgroup memory to improve cache reuse
- `BENCH_PARAM_num_warmup` warmup launches (default 3), then `BENCH_PARAM_num_iters` timed launches (default 7), each recorded individually
- GFLOPS = 2·N³ / time\_sec / 1e9 (derived downstream from `time_ms` and params)

## Files

| File | Description |
|------|-------------|
| `parboil_sgemm.hip` | HIP/CUDA source — tiled SGEMM kernel + self-contained utilities |
| `parboil_sgemm.metal` | Metal compute shader — tiled SGEMM using threadgroup memory |
| `parboil_sgemm_metal.mm` | ObjC++ Metal host code |
| `Makefile` | Auto-detecting build (Darwin→metal, ROCm→rocm, else→cuda) |
| `README.md` | This file |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
make PLATFORM=metal
```

## Run

```bash
./parboil_sgemm                      # defaults: 1024×1024
./parboil_sgemm --size 2048
BENCH_PARAM_num_iters=10 ./parboil_sgemm --size 512
```

## Output

**stdout** — JSON lines: one `kernel` record per timed iteration with the GPU
event time (`time_ms`) and host wall-clock time (`host_time_ms`), then a
summary whose `total_time_ms` is the sum of all iterations' GPU event times:
```
{"type":"kernel","name":"sgemm_kernel","iter":0,"time_ms":0.495,"host_time_ms":0.520,"params":{"size":1024,"block_size":256,"precision":"float"}}
...
{"type":"summary","total_time_ms":3.465}
```
GFLOPS is derived downstream from `time_ms` and params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
Matrix size: 1024×1024  |  alpha=1.5  beta=1.2

sgemm_kernel iter 0: gpu = 0.495 ms, host = 0.520 ms
...
Total GPU time: 3.465 ms over 7 iterations
Skipping verification for large size
```

## Platform Notes

- **CUDA**: compiled with `nvcc` from `parboil_sgemm.hip` (uses `-x cu` flag)
- **ROCm**: compiled with `hipcc` from `parboil_sgemm.hip`
- **Metal**: compiled with `clang++` from `parboil_sgemm_metal.mm`; the Metal shader (`parboil_sgemm.metal`) is loaded and compiled at runtime from the same directory as the binary
- Correctness verification runs only for N≤256 (CPU reference is O(N³))
