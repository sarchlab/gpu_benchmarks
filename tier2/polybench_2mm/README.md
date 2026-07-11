# tier2/polybench_2mm — Two Matrix Multiplications (PolyBench 2MM)

Benchmarks two consecutive dense matrix multiplications:
- **D = α·A·B + β·D**
- **E = α·C·D + β·E**

for square N×N single-precision matrices. The 2MM pass runs exactly once per execution and emits one GPU/host time record per kernel; compute throughput (GFLOPS) is derived downstream.

Modeled on the [PolyBench/GPU benchmark suite](https://sourceforge.net/projects/polybench/) 2MM workload; the kernels are a generic tiled shared-memory GEMM reimplementation, not the upstream naive kernels.

## Algorithm

- Two back-to-back tiled GEMM operations on NxN matrices
- Default parameters: N=1024, alpha=1.5, beta=1.2 (PolyBench default values)
- Tiled/shared-memory kernel: 16×16 thread tiles load sub-matrices into shared memory
- Total FLOPs: 4·N³ (two matrix multiplies, each 2·N³) — GFLOPS is derived downstream from the emitted `time_ms` and params

## Files

| File | Description |
|------|-------------|
| `polybench_2mm.hip` | HIP source — tiled 2MM kernel with HIP/CUDA compat |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |
| `params.json` | Parameter specification |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./polybench_2mm                     # defaults: 1024×1024
./polybench_2mm --size 2048
BENCH_PARAM_size=2048 ./polybench_2mm
```

Environment variables override argv:

| Variable | Description | Default |
|----------|-------------|---------|
| `BENCH_PARAM_size` | Matrix dimension N | 1024 |
| `BENCH_PARAM_alpha` | Scaling factor alpha | 1.5 |
| `BENCH_PARAM_beta` | Scaling factor beta | 1.2 |
| `BENCH_PARAM_vector_add_warmup` | Vector-add warmer (clock boost + LLC eviction) before the timed pass | true |

The 2MM pass runs exactly once per execution; repetition is the harness's job (`gpubench run --reps N`).

## Output

**stdout** — JSON-lines: one `{"type":"kernel",...}` record per kernel (`mm_kernel_1`, `mm_kernel_2`) carrying its GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`), then a summary whose `total_time_ms` is the sum of the two GPU event times:
```json
{"type":"kernel","name":"mm_kernel_1","iter":0,"time_ms":1.234,"host_time_ms":1.301,"params":{"size":1024,...}}
{"type":"kernel","name":"mm_kernel_2","iter":0,"time_ms":1.234,"host_time_ms":1.298,"params":{"size":1024,...}}
{"type":"summary","total_time_ms":2.468}
```

Derived metrics such as GFLOPS (4·N³ / time) are computed downstream from `time_ms` and params. The pristine D and E matrices are re-uploaded at the start of the run (untimed) since the kernels rewrite them in place.

**stderr** — human-readable device info and timing details.

## Differences from polybench_gemm

| Property | polybench_gemm | polybench_2mm |
|----------|---------------|---------------|
| Operation | C = α·A·B + β·C | D = α·A·B + β·D, E = α·C·D + β·E |
| Matrices | 3 (A, B, C) | 5 (A, B, C, D, E) |
| Kernels | 1 | 2 (sequential) |
| FLOPs | 2·N³ | 4·N³ |
