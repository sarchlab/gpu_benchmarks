# tier2/shoc_reduction/cuda — SHOC Parallel Sum Reduction Benchmark (native CUDA)

Native CUDA version of this benchmark. Kernels, driver, verification and
output are identical to [`../hip/`](../hip/); only runtime API names differ.
It builds with `nvcc` only, uses no architecture-specific features, and targets
A100 (sm_80) and H100 (sm_90). Shared timing/warmup headers come from
`common_cuda/`.

## Overview

The **Reduction** benchmark performs a parallel sum reduction over a large float
array, collapsing it to a single scalar result, and emits the raw GPU/host
kernel times (effective memory bandwidth is derived downstream). It uses a
**tree-based shared-memory (threadgroup) reduction**:

- Each thread block (threadgroup) processes **2 × block_size** elements.
- Threads cooperatively reduce to a single partial sum per block.
- Multiple kernel levels are applied until one value remains.

This benchmark is derived from the [SHOC benchmark suite](https://github.com/vetter/shoc).

## Provenance & fidelity

Derived from the SHOC benchmark suite (https://github.com/vetter/shoc) — the `Reduction` workload. This is an independent reimplementation for portable measurement, **not** upstream source, and **not** a faithful reproduction of the upstream kernel. It differs from SHOC's version as follows:

- Simplified tree reduction vs SHOC's unrolled/vec4 multi-pass reduction.

Read this as "a sum-reduction benchmark in the spirit of SHOC," not as SHOC's `Reduction`.

## Measurement protocol

The full multi-level reduction runs **exactly once** per execution — there is
no in-program warmup or iteration loop. When `BENCH_PARAM_vector_add_warmup` is
`true` (default), the vector-add warmer runs immediately before the timed run:
~100 ms of sustained load ramps the GPU clock while its final pass evicts the
last-level cache, so the reduction is measured cold-cache at a boosted clock.
Repetition is the harness's job: `gpubench run --reps N` re-executes the binary
in fresh processes.

## Files

| File | Description |
|------|-------------|
| `shoc_reduction.cu` | CUDA source — self-contained kernel + host |
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
./shoc_reduction [--size N] [--block_size B]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--size N` | `4194304` | Number of `float` elements (4 M ≈ 16 MiB) |
| `--block_size B` | `256` | GPU thread-block size |

Environment variables override argv:

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_size` | `4194304` | Number of elements |
| `BENCH_PARAM_block_size` | `256` | Thread-block size |
| `BENCH_PARAM_vector_add_warmup` | `true` | Clock-boost + LLC-evict warmer before the timed run |

## Output

**stdout** — JSON lines: one `{"type":"kernel","name":"reduce_kernel",...}` record carrying the GPU event time (`time_ms`) accumulated over all `reduce_kernel` launches of the multi-level reduction and the host wall-clock time (`host_time_ms`), then a summary whose `total_time_ms` equals that record's GPU event time:
```json
{"type":"kernel","name":"reduce_kernel","iter":0,"time_ms":0.1821,"host_time_ms":0.2010,"params":{"size":4194304,"block_size":256}}
{"type":"summary","total_time_ms":0.1821}
```

Derived metrics such as effective bandwidth are computed downstream from `time_ms` and params.

**stderr** — human-readable summary:
```
Device: NVIDIA GeForce RTX 4090 (id=0)
Array size: 4194304 floats (16.0 MiB)
Block size: 256

gpu = 0.182 ms, host = 0.201 ms
Total GPU time: 0.182 ms
GPU sum: 499500.000000, CPU sum: 499500.000000, relative error: 0.000000e+00
PASS
```

## Verification

After the timed run (measurements already emitted), the GPU sum is compared
against a CPU reference sum. On FAIL the process exits non-zero, so the harness
marks the run as failed without losing the measurements. There is no
verification on/off parameter.

## Bandwidth Formula

Effective bandwidth is derived downstream from the emitted `time_ms` and params as:

```
Bandwidth (GB/s) = 2 × N × sizeof(float) / time_seconds / 1e9
```

The factor of 2 accounts for reading N floats and writing the reduced result.

## Algorithm

### CUDA

```
Level 1: reduce_kernel<<<num_blocks, B, B*sizeof(float)>>>(input, partial, N)
Level 2: reduce_kernel<<<1, B, B*sizeof(float)>>>(partial, result, num_blocks)
```

Each block loads two elements per thread into shared memory, then performs
a binary tree reduction until one partial sum per block is produced.
