# tier2/shoc_gemm/hip — Dense Matrix Multiply (SHOC GEMM)

Benchmarks dense matrix multiplication: **C = α·A·B + β·C** for square N×N single-precision matrices. Emits one raw kernel time per execution; compute throughput (GFLOPS) is derived downstream.

## Provenance & fidelity

Derived from the SHOC benchmark suite (https://github.com/vetter/shoc) — the `GEMM` workload. This is an independent reimplementation for portable measurement, **not** upstream source, and **not** a faithful reproduction of the upstream kernel. It differs from SHOC's version as follows:

- Naive 16×16 tiled GEMM vs SHOC's cuBLAS N/T path.

Read this as "a dense-GEMM benchmark in the spirit of SHOC," not as SHOC's `GEMM`.

## Algorithm

- Square N×N single-precision matrix multiply
- Tiled/shared-memory kernel: 16×16 thread tiles load sub-matrices into threadgroup memory to improve cache reuse
- GFLOPS = 2·N³ / time\_sec / 1e9 (derived downstream from the emitted times and params)

## Measurement protocol

The GEMM kernel is launched and timed **exactly once** per execution — there is
no in-program warmup or iteration loop. When `BENCH_PARAM_vector_add_warmup` is
`true` (default), the vector-add warmer runs immediately before the timed
launch: ~100 ms of sustained load ramps the GPU clock while its final pass
evicts the last-level cache, so the kernel is measured cold-cache at a boosted
clock. Repetition is the harness's job: `gpubench run --reps N` re-executes the
binary in fresh processes.

## Parameters

| Env var | Default | Description |
|---------|---------|-------------|
| `BENCH_PARAM_size` | 512 | Square matrix dimension N |
| `BENCH_PARAM_vector_add_warmup` | true | Clock-boost + LLC-evict warmer before the timed run |

## Files

| File | Description |
|------|-------------|
| `shoc_gemm.hip` | HIP source — timed tiled GEMM kernel and driver |
| `cpu_reference.h` | Tiled, threaded full CPU reference used by verification and host tests |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./shoc_gemm                     # defaults: 512×512
./shoc_gemm --size 1024
BENCH_PARAM_size=1024 ./shoc_gemm   # env vars override argv
```

## Output

**stdout** — JSON lines: one kernel record with the GPU event time (`time_ms`)
and host wall-clock time (`host_time_ms`), then a summary whose `total_time_ms`
equals that record's GPU event time:
```
{"type":"kernel","name":"gemm_kernel","iter":0,"time_ms":12.345600,"host_time_ms":12.410000,"params":{"size":512,"verification":"full_cpu_v1"}}
{"type":"summary","total_time_ms":12.345600}
```

Derived metrics such as GFLOPS are computed downstream from `time_ms` and
the params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
Matrix size: 512×512

gemm_kernel: gpu = 12.346 ms, host = 12.410 ms
Total GPU time: 12.346 ms
PASS: 0 mismatches in 262144 values
```

## Verification

After the single timed launch, every element of the actual measured GPU matrix
is downloaded and compared with a full CPU reference at every declared size.
Both values must be finite and satisfy the existing criterion
`abs(GPU - CPU) <= 1e-3 * abs(CPU) + 1e-5`. The original FP32 increasing-k
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
python3 tests/shoc_gemm_verification.py --host-only --work-dir /tmp/shoc-gemm-host
python3 tests/shoc_gemm_verification.py --platform cuda --sweep --work-dir /tmp/shoc-gemm-gpu
```

The tests cover complete equivalence to the original CPU recipe, independent
known answers, failed thread creation, actual device-tail corruption and all 22
declared sizes with a 60-second process budget. This verification repair retains
the local GEMM workload and the provenance limitations stated above.

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `shoc_gemm.hip`
