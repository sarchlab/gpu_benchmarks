# tier2/polybench_gemm — Dense Matrix Multiply (PolyBench GEMM)

Benchmarks dense matrix multiplication: **C = α·A·B + β·C** for square N×N single-precision matrices. The kernel runs exactly once per execution and the raw GPU time is emitted; compute throughput (GFLOPS) is derived downstream.

## Provenance & fidelity

Derived from the [PolyBench/GPU benchmark suite](https://sourceforge.net/projects/polybench/) — the `gemm` workload. Independent reimplementation for portable measurement, **not** upstream source, and **not** a faithful reproduction. It differs as follows:

- Tiled but math-canonical GEMM (generic tiled shared-memory kernel, not the upstream naive kernel; the arithmetic result is unchanged).

Read this as "a GEMM benchmark in the spirit of PolyBench," not PolyBench's `gemm`.

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
{"type":"kernel","name":"polybench_gemm_kernel","iter":0,"time_ms":0.245600,"host_time_ms":0.301200,"params":{"size":1024,"verification":"full_cpu_v1"}}
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

## Measured-result verification

Every run checks every element of the actual measured `C` against a complete
independent double-precision `alpha*A*B + beta*C_initial` CPU reference. The
original input arrays and the nonzero initial-C contribution are preserved.
Checking happens after timing, with one complete result download that reuses
the consumed host C buffer. No separate GPU verification pass is launched.

A value passes only when actual, reference and error are finite and
`abs(actual-reference) <= 5e-4 * max(1,abs(reference))`. For the declared
nonnegative inputs, fixed positive float coefficients and `N <= 8192`, this
exceeds the conservative `gamma_(N+3)` dot-product/scaling bound with float
unit roundoff `u = 2^-24` and `gamma_k = k*u/(1-k*u)`, including the negligible
double-reference error. Smaller errors are intentionally admitted; this does
not promise detection of arbitrary corruption or qualify every manual input.
The checker counts all mismatches, prints at most ten details to stderr and
exits nonzero on failure. Kernel metadata records `verification="full_cpu_v1"`.

The full CPU product uses at most eight host threads on disjoint output row
tiles, with increasing reduction order for every element. It uses one worker
below 512×512 output cells and otherwise caps the count by hardware concurrency
and available row tiles. If an additional thread cannot be created because of
a system error or thread-state allocation failure, its remaining partitions
run on the calling thread; already-started workers are
joined before the result is consumed. This changes verification cost only;
GPU kernels, inputs, geometry, upload/warmer/event sequence and the declared
size sweep are unchanged.

At N=8192, the reference checks 67,108,864 values and adds 512 MiB of host
storage alongside the existing 768 MiB inputs. The 256 MiB result copy reuses
host C. CPU cost is outside GPU event time but inside the process timeout;
other-host budgets/admission and AMD qualification remain open. This preserves
the documented independent reimplementation and makes no new upstream-fidelity
claim.
