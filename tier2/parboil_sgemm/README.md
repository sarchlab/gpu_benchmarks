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
| `parboil_sgemm.hip` | HIP/CUDA source — timed tiled SGEMM kernel and driver |
| `cpu_reference.h` | Tiled, threaded full CPU reference used by verification and host tests |
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
{"type":"kernel","name":"sgemm_kernel","iter":0,"time_ms":0.495000,"host_time_ms":0.520000,"params":{"size":1024,"verification":"full_cpu_v1"}}
{"type":"summary","total_time_ms":0.495000}
```
GFLOPS is derived downstream from `time_ms` and params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
Matrix size: 1024×1024  |  alpha=1.5  beta=1.2

sgemm_kernel: gpu = 0.495 ms, host = 0.520 ms
Total GPU time: 0.495 ms
PASS: 0 mismatches in 1048576 values
```

## Verification

After the single timed launch, every element of the actual measured GPU matrix
is downloaded and compared with a full CPU reference at every declared size.
Both values must be finite and satisfy the existing criterion
`abs(GPU - CPU) <= 1e-3 * abs(CPU) + 1e-4`. The CPU reference starts from
pristine C, including the nonzero `beta*C` term
with the fixed `alpha=1.5`, `beta=1.2`. The original FP32 increasing-k
accumulation and `alpha*sum + beta*initial_C` expression are preserved. There is
no GPU rerun, sampling, skipped large size or verification toggle. Records
include `verification: "full_cpu_v1"`.

The CPU reference uses 4×32 output tiles and at most eight workers owning
disjoint row tiles. It uses one worker below 512² output elements, otherwise
caps the host's reported concurrency at eight. If some worker threads cannot
start, the caller completes their partitions and joins those already running.
A reference or host-allocation failure exits nonzero. All mismatches are
counted, with at most ten detailed diagnostics. The four host matrices retain
the previous memory footprint (about 1 GiB at 8192²); reference work, downloads and
comparison are outside kernel timing but count toward the whole-process timeout.
CPU capacity on other hosts must be qualified separately.

From the repository root, use a new work directory:

```bash
python3 tests/parboil_sgemm_verification.py --host-only --work-dir /tmp/parboil-sgemm-host
python3 tests/parboil_sgemm_verification.py --platform cuda --sweep --work-dir /tmp/parboil-sgemm-gpu
```

The tests cover complete equivalence to the original CPU recipe, independent
known answers, failed thread creation, automatic worker selection, actual
device-tail corruption and all 22
declared sizes with a 60-second process budget. This verification repair retains
the local GEMM workload and the provenance limitations stated above.

## Platform Notes

- **CUDA**: compiled with `nvcc` from `parboil_sgemm.hip` (uses `-x cu` flag)
- **ROCm**: compiled with `hipcc` from `parboil_sgemm.hip`
