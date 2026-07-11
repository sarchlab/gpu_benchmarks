# tier2/polybench_gemm — Dense Matrix Multiply (PolyBench GEMM)

Benchmarks dense matrix multiplication: **C = α·A·B + β·C** for square N×N single-precision matrices. The kernel runs exactly once per execution and the raw GPU time is emitted; compute throughput (GFLOPS) is derived downstream.

Modeled on the [PolyBench/GPU benchmark suite](https://sourceforge.net/projects/polybench/) GEMM workload; the kernel is a generic tiled shared-memory GEMM reimplementation, not the upstream naive kernel.

## Algorithm

- Square N×N single-precision matrix multiply with scaling: `C = alpha * A * B + beta * C`
- Default parameters: N=1024, alpha=1.5, beta=1.2 (PolyBench default values)
- Tiled/shared-memory kernel: 16×16 thread tiles load sub-matrices into shared memory to improve cache reuse
- GFLOPS = 2·N³ / time\_sec / 1e9 (derived downstream from the emitted `time_ms` and params)

## Files

| File | Description |
|------|-------------|
| `polybench_gemm.hip` | HIP source — tiled GEMM kernel + self-contained utilities |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./polybench_gemm                     # default: 1024×1024
./polybench_gemm --size 2048
BENCH_PARAM_size=2048 ./polybench_gemm
```

| Flag / Env var | Default | Description |
|----------------|---------|-------------|
| `--size N` / `BENCH_PARAM_size` | 1024 | Matrix dimension N (N×N) |
| `BENCH_PARAM_vector_add_warmup` | true | Vector-add warmer (clock boost + LLC eviction) before the timed launch |

alpha (1.5) and beta (1.2) are fixed in the source. The kernel runs exactly
once per execution; repetition is the harness's job (`gpubench run --reps N`).

## Output

**stdout** — JSON lines: one `kernel` record for the single timed launch with
the GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`),
then a summary whose `total_time_ms` equals that GPU event time:
```
{"type":"kernel","name":"polybench_gemm_kernel","iter":0,"time_ms":0.245600,"host_time_ms":0.301200,"params":{"size":1024}}
{"type":"summary","total_time_ms":0.245600}
```

Derived metrics such as GFLOPS are computed downstream from `time_ms` and params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
Matrix size: 1024×1024  |  alpha=1.5  beta=1.2

polybench_gemm_kernel: gpu = 0.246 ms, host = 0.301 ms
Total GPU time: 0.246 ms
```

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `polybench_gemm.hip`

## Differences from shoc_gemm

| Property | shoc_gemm | polybench_gemm |
|----------|-----------|----------------|
| Default N | 512 | 1024 |
| alpha | 1.0 | 1.5 |
| beta | 0.0 | 1.2 |
