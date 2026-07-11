# fp32_throughput — Tier 1 Microbenchmark

## Overview

`fp32_throughput` measures peak single-precision (FP32) floating-point compute
throughput by executing a tight chain of fused multiply-add (FMA) instructions
on every GPU thread.

Because each FMA counts as **2 floating-point operations** (1 multiply + 1
add), and all values live in registers (no memory traffic), the result directly
reflects the arithmetic throughput of the GPU rather than memory bandwidth.

The binary emits the raw kernel time; GFLOPS is derived downstream
from each record's `time_ms` and params:

```
GFLOPS = (num_threads × fmas_per_thread × 2) / elapsed_seconds / 1e9
```

## Platform Support

| Platform | Source file | Compiler |
|----------|-------------|---------|
| NVIDIA CUDA | `fp32_throughput.hip` | `nvcc` |
| AMD ROCm/HIP | `fp32_throughput.hip` | `hipcc` |

## What It Measures

- **FP32 arithmetic throughput** — bounded by the number of FP32 FMA units and
  their clock frequency (e.g. ~82 TFLOPS on an RTX 4090).
- The kernel uses **four independent accumulators** to expose instruction-level
  parallelism and hide per-instruction FP32 latency, driving the hardware toward
  its peak throughput rather than its latency bound.

## Build

The directory is **self-contained** — no CMake or external dependencies.

### Auto-detect platform

```bash
cd tier1/fp32_throughput
make
```

The Makefile auto-detects the platform:
- Linux with `/opt/rocm` → ROCm/HIP
- Otherwise → CUDA

### Explicit platform

```bash
# NVIDIA CUDA
make PLATFORM=cuda

# AMD ROCm/HIP
make PLATFORM=rocm
```

### Clean

```bash
make clean
```

## Run

```bash
# Default 4096 FMAs per thread
./fp32_throughput

# Custom FMA count per thread (increase for faster GPUs)
./fp32_throughput 16384
```

Each execution launches and times the kernel **exactly once** — there is no
in-program warmup or iteration loop. Unless `BENCH_PARAM_vector_add_warmup`
is set to `false` (default `true`), a vector-add warmer runs ~100 ms of
sustained load immediately before the timed launch, boosting the GPU clock
from idle and evicting the last-level cache, so the kernel is measured
cold-cache at a boosted clock. Repeat measurements come from fresh process
re-executions: `gpubench run --reps N`.

## Example Output

Progress and device info go to stderr; stdout carries JSON lines — one record
for the single timed launch plus a final summary:

```
{"type":"kernel","name":"fp32_fma_kernel","iter":0,"time_ms":0.524128,"host_time_ms":0.601342,"params":{"fmas_per_thread":4096,"num_blocks":1,"threads_per_block":256}}
{"type":"summary","total_time_ms":0.524128}
```

- `time_ms` is the GPU event time for the single kernel launch;
  `host_time_ms` is the host wall-clock around launch + sync.
- `total_time_ms` equals the single launch's GPU event time.
- GFLOPS is not printed; it is derived downstream from `time_ms` and params.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BENCH_PARAM_fmas_per_thread` (or argv[1]) | 4096 | FMA iterations per thread (rounded to nearest multiple of 4) |
| `BENCH_PARAM_threads_per_block` | 256 | Threads per block / threadgroup |
| `BENCH_PARAM_num_blocks` | 1 | Number of blocks in the dispatch grid (default 1 = single-SM/CU measurement) |
| `BENCH_PARAM_vector_add_warmup` | true | Run the vector-add warmer (clock boost + LLC eviction) before the timed launch; set `false` to time the kernel as-is |

> **Single-SM/CU measurement.** The default grid is one block
> (`num_blocks = 1`), so the benchmark measures a single SM/CU at boost
> clock. Because FMA pipes are per-SM resources with no cross-SM
> interaction, whole-device peak GFLOPS can be estimated as the per-SM
> value × the GPU's SM/CU count. Treat that as an upper bound: an all-SM
> workload is power/thermal-limited and typically sustains a lower clock
> than a single busy SM. For a device-wide run, set
> `BENCH_PARAM_num_blocks` manually (e.g. ≥ 8 × SM count).

## Notes

- GFLOPS (10⁹ floating-point operations per second) is derived downstream
  from the recorded times; the binary itself only emits raw times.
- The vector-add warmer (on by default) boosts the GPU clock and evicts the
  last-level cache before the single timed launch; there are no untimed
  warm-up launches of the benchmark kernel itself.
- **CUDA/HIP**: CUDA/HIP events are used for high-resolution GPU-side timing.
- The kernel loop is unrolled 4× (four independent accumulators) to saturate
  the throughput path of modern GPU FP32 pipelines.
- `fmas_per_thread` is rounded down to the nearest multiple of 4 at runtime to
  match the unrolled loop structure.
