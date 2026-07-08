# tier2/shoc_gemm — Dense Matrix Multiply (SHOC GEMM)

Benchmarks dense matrix multiplication: **C = α·A·B + β·C** for square N×N single-precision matrices. Emits raw per-iteration kernel times; compute throughput (GFLOPS) is derived downstream.

Derived from the [SHOC benchmark suite](https://github.com/vetter/shoc) GEMM workload.

## Algorithm

- Square N×N single-precision matrix multiply
- Tiled/shared-memory kernel: 16×16 thread tiles load sub-matrices into threadgroup memory to improve cache reuse
- GFLOPS = 2·N³ / time\_sec / 1e9 (derived downstream from the emitted times and params)

## Files

| File | Description |
|------|-------------|
| `shoc_gemm.hip` | HIP source — tiled GEMM kernel + self-contained utilities |
| `shoc_gemm.metal` | Metal compute shader — tiled GEMM using threadgroup memory |
| `shoc_gemm_metal.mm` | ObjC++ Metal host code |
| `Makefile` | Auto-detecting build (Darwin→metal, ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
make PLATFORM=metal
```

## Run

```bash
./shoc_gemm                     # defaults: 512×512
./shoc_gemm --size 1024
BENCH_PARAM_size=1024 ./shoc_gemm   # env vars override argv
```

Environment variables: `BENCH_PARAM_size`, `BENCH_PARAM_block_size`,
`BENCH_PARAM_precision`, plus `BENCH_PARAM_num_warmup` (warmup launches,
default 3) and `BENCH_PARAM_num_iters` (timed launches, each recorded,
default 7).

## Output

**stdout** — JSON lines, one kernel record per timed iteration with the GPU
event time (`time_ms`) and host wall-clock time (`host_time_ms`), then a
summary whose `total_time_ms` is the sum of all iterations' GPU event times:
```
{"type":"kernel","name":"gemm_kernel","iter":0,"time_ms":12.345600,"host_time_ms":12.410000,"params":{"size":512,"block_size":16,"precision":"float"}}
...
{"type":"summary","total_time_ms":86.419200}
```

Derived metrics such as GFLOPS are computed downstream from `time_ms` and
the params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
Matrix size: 512×512

gemm_kernel iter 0: gpu = 12.346 ms, host = 12.410 ms
...
Total GPU time: 86.419 ms over 7 iterations
PASS
```

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `shoc_gemm.hip`
- **Metal**: compiled with `clang++` from `shoc_gemm_metal.mm`; the Metal shader (`shoc_gemm.metal`) is loaded and compiled at runtime from the same directory as the binary
