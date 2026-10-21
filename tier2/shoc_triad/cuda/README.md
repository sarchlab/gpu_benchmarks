# tier2/shoc_triad/cuda — SHOC Stream Triad Benchmark (native CUDA)

Native CUDA version of this benchmark. Kernels, driver, verification and
output are identical to [`../hip/`](../hip/); only runtime API names differ.
It builds with `nvcc` only, uses no architecture-specific features, and targets
A100 (sm_80) and H100 (sm_90). Shared timing/warmup headers come from
`common_cuda/`.

## Overview

The **Stream Triad** benchmark measures effective GPU memory bandwidth by executing the classic operation:

```
a[i] = b[i] + scalar * c[i]
```

This pattern reads two arrays (`b`, `c`) and writes one (`a`), for a total of
**3 × N × 4 bytes** of memory traffic. It is derived from the [SHOC benchmark
suite](https://github.com/vetter/shoc).

## Provenance & fidelity

Derived from the SHOC benchmark suite (https://github.com/vetter/shoc) — the `Triad` workload. This is an independent reimplementation for portable measurement, **not** upstream source, and **not** a faithful reproduction of the upstream kernel. It differs from SHOC's version as follows:

- Device-resident triad; drops SHOC's PCIe streaming/overlap pipeline.

Read this as "a Stream-Triad benchmark in the spirit of SHOC," not as SHOC's `Triad`.

## Measurement protocol

The triad kernel is launched and timed **exactly once** per execution — there
is no in-program warmup or iteration loop. When `BENCH_PARAM_vector_add_warmup`
is `true` (default), the vector-add warmer runs immediately before the timed
launch: ~100 ms of sustained load ramps the GPU clock while its final pass
evicts the last-level cache, so the kernel is measured cold-cache at a boosted
clock. Repetition is the harness's job: `gpubench run --reps N` re-executes the
binary in fresh processes.

## Files

| File | Description |
|------|-------------|
| `shoc_triad.cu` | CUDA source — self-contained kernel + host |
| `Makefile` | nvcc build (`CUDA_ARCH`, default `native`) |

## Building

```bash
make                     # -arch=native (builds for the GPU in this machine)
make CUDA_ARCH=sm_80     # A100
make CUDA_ARCH=sm_90     # H100
```

**Requirements:**
- **CUDA**: `nvcc` (CUDA toolkit ≥ 10)

## Running

```bash
./shoc_triad [--array_size N] [--block_size B]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--array_size N` | `4194304` | Number of `float` elements (4 M ≈ 16 MiB per array) |
| `--block_size B` | `256` | GPU thread-block size |

Environment variables override the flags:

| Env var | Default | Description |
|---------|---------|-------------|
| `BENCH_PARAM_array_size` | 4194304 | Number of elements |
| `BENCH_PARAM_block_size` | 256 | Thread-block size |
| `BENCH_PARAM_vector_add_warmup` | true | Clock-boost + LLC-evict warmer before the timed run |

## Output

**stdout** — JSON lines: one `kernel` record for the single timed launch
carrying the GPU event time (`time_ms`) and host wall-clock time
(`host_time_ms`), then a `summary` whose `total_time_ms` equals that record's
GPU event time:
```
{"type":"kernel","name":"triad_kernel","iter":0,"time_ms":0.314200,"host_time_ms":0.330000,"params":{"array_size":4194304,"block_size":256}}
{"type":"summary","total_time_ms":0.314200}
```

**stderr** — human-readable summary:
```
Device: NVIDIA GeForce RTX 4090 (id=0)
Array size: 4194304 floats (16.0 MiB)
Block size: 256

triad_kernel: gpu = 0.314 ms, host = 0.330 ms
Total GPU time: 0.314 ms
PASS
```

## Verification

After the timed run (measurements already emitted), the GPU result `a` is
compared element-wise against a CPU reference. On FAIL the process exits
non-zero, so the harness marks the run as failed without losing the
measurements. There is no verification on/off parameter.

## Bandwidth Formula

```
Bandwidth (GB/s) = 3 × N × sizeof(float) / time_seconds / 1e9
```

The factor of 3 accounts for two reads (`b`, `c`) and one write (`a`).
Derived downstream from the emitted `time_ms` and params.
