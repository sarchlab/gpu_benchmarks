# tier2/shoc_gemm — Dense Matrix Multiply (SHOC GEMM)

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
| `shoc_gemm.hip` | HIP source — tiled GEMM kernel + self-contained utilities |
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
{"type":"kernel","name":"gemm_kernel","iter":0,"time_ms":12.345600,"host_time_ms":12.410000,"params":{"size":512}}
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
PASS
```

## Verification

After the timed run (measurements already emitted), the GPU result is compared
against a CPU reference — but only when `size <= 512` (the O(N³) CPU reference
is too slow beyond that; larger sizes print `Skipping verification for large
size`). On FAIL the process exits non-zero, so the harness marks the run as
failed without losing the measurements. There is no verification on/off
parameter.

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `shoc_gemm.hip`
