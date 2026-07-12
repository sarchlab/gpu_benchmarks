# tier2/parboil_sgemm — Tiled Single-Precision GEMM (Parboil)

Benchmarks tiled single-precision matrix multiplication: **C = α·A·B + β·C** for square N×N matrices. Emits the raw kernel time of a single launch; GFLOPS is derived downstream.

## Provenance & fidelity

Derived from the [Parboil benchmark suite](http://impact.crhc.illinois.edu/parboil/) — the `sgemm` workload. Independent reimplementation for portable measurement, **not** upstream source, and **not** a faithful reproduction. It differs as follows:

- Row-major textbook tiled GEMM vs Parboil's mysgemmNT column-major register-tiled kernel.

Read this as "an SGEMM benchmark in the spirit of Parboil," not Parboil's `sgemm`.

## Algorithm

- Square N×N single-precision matrix multiply: **C = alpha·A·B + beta·C**
- Default: N=1024, alpha=1.5, beta=1.2 (alpha/beta are fixed)
- Tiled kernel: 16×16 thread tiles load sub-matrices into shared memory to improve reuse
- GFLOPS = 2·N³ / time\_sec / 1e9 (derived downstream from `time_ms` and params)

## Measurement

The kernel launches **exactly once** per process execution (see the root
README "Measurement Methodology"). With `BENCH_PARAM_vector_add_warmup`
(default `true`), a vector-add warmer boosts the GPU clock and evicts the
last-level cache immediately before the timed launch. Repetition =
`gpubench run --reps N`.

## Parameters

| Env var (or `--flag`) | Description | Default |
|---------|-------------|---------|
| `BENCH_PARAM_size` / `--size` | Square matrix dimension N | 1024 |
| `BENCH_PARAM_vector_add_warmup` | Cold-cache clock warmer on/off | true |

## Files

| File | Description |
|------|-------------|
| `parboil_sgemm.hip` | HIP/CUDA source — tiled SGEMM kernel + self-contained utilities |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |
| `README.md` | This file |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./parboil_sgemm                      # defaults: 1024×1024
./parboil_sgemm --size 2048
BENCH_PARAM_size=512 ./parboil_sgemm
```

## Output

**stdout** — JSON lines: one `kernel` record (`iter:0`) with the GPU event
time (`time_ms`) and host wall-clock time (`host_time_ms`), then a summary
whose `total_time_ms` is the sum of all recorded GPU event times:
```
{"type":"kernel","name":"sgemm_kernel","iter":0,"time_ms":0.495000,"host_time_ms":0.520000,"params":{"size":1024}}
{"type":"summary","total_time_ms":0.495000}
```
GFLOPS is derived downstream from `time_ms` and params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
Matrix size: 1024×1024  |  alpha=1.5  beta=1.2

sgemm_kernel: gpu = 0.495 ms, host = 0.520 ms
Total GPU time: 0.495 ms
Skipping verification for large size
```

## Platform Notes

- **CUDA**: compiled with `nvcc` from `parboil_sgemm.hip` (uses `-x cu` flag)
- **ROCm**: compiled with `hipcc` from `parboil_sgemm.hip`
- Correctness verification runs only for N≤256 (CPU reference is O(N³)); on FAIL the process exits non-zero
