# int32_throughput — Tier 1 Microbenchmark

## Overview

`int32_throughput` measures peak 32-bit integer (INT32) compute throughput by
executing a tight chain of integer multiply-add (MAD) instructions on every
GPU thread.

Because each MAD counts as **2 integer operations** (1 multiply + 1 add), and
all values live in registers (no memory traffic), the result directly reflects
the arithmetic throughput of the GPU rather than memory bandwidth.

The binary emits raw per-iteration kernel times; GIOPS is derived downstream
from each record's `time_ms` and params:

```
GIOPS = (num_threads × mads_per_thread × 2) / elapsed_seconds / 1e9
```

## Platform Support

| Platform | Source file | Compiler |
|----------|-------------|---------|
| NVIDIA CUDA | `int32_throughput.hip` | `nvcc` |
| AMD ROCm/HIP | `int32_throughput.hip` | `hipcc` |
| Apple Metal | `int32_throughput_metal.mm` + `int32_throughput.metal` | `clang++` |

## What It Measures

- **INT32 arithmetic throughput** — bounded by the number of INT32 ALU units and
  their clock frequency.
- The kernel uses **four independent accumulators** to expose instruction-level
  parallelism and hide per-instruction integer latency, driving the hardware
  toward its peak throughput rather than its latency bound.

## Build

The directory is **self-contained** — no CMake or external dependencies.

### Auto-detect platform

```bash
cd tier1/int32_throughput
make
```

The Makefile auto-detects the platform:
- macOS (Darwin) → Metal
- Linux with `/opt/rocm` → ROCm/HIP
- Otherwise → CUDA

### Explicit platform

```bash
# NVIDIA CUDA
make PLATFORM=cuda

# AMD ROCm/HIP
make PLATFORM=rocm

# Apple Metal
make PLATFORM=metal
```

### Clean

```bash
make clean
```

## Run

```bash
# Default 4096 MADs per thread
./int32_throughput

# Custom MAD count per thread (increase for faster GPUs)
./int32_throughput 16384
```

> **Note (Metal):** The `int32_throughput.metal` shader file must be present in
> the same directory as the compiled binary — the host code loads it at runtime.

## Example Output

Human-readable diagnostic output goes to **stderr**; **stdout** carries JSON
lines — one record per timed iteration plus a final summary:

```
{"type":"kernel","name":"int32_mad_kernel","iter":0,"time_ms":0.524128,"host_time_ms":0.598342,"params":{"mads_per_thread":4096,"num_blocks":1,"threads_per_block":256}}
...
{"type":"summary","total_time_ms":3.668901}
```

- `time_ms` is the GPU event time for that single kernel launch;
  `host_time_ms` is the host wall-clock around launch + sync.
- Every timed iteration is emitted individually — no averaging.
- `total_time_ms` is the sum of all iterations' GPU event times.
- GIOPS is not printed; it is derived downstream from `time_ms` and params.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BENCH_PARAM_mads_per_thread` (or argv[1]) | 4096 | MAD iterations per thread (rounded to nearest multiple of 4) |
| `BENCH_PARAM_threads_per_block` | 256 | Threads per block / threadgroup |
| `BENCH_PARAM_num_blocks` | 1024 | Number of blocks in the dispatch grid (params.json pins 1 for harness sweeps) |
| `BENCH_PARAM_num_warmup` | 3 | Warm-up kernel launches (not timed) |
| `BENCH_PARAM_num_iters` | 7 | Timed kernel launches, each recorded individually |

## Notes

- GIOPS (10⁹ integer operations per second) is derived downstream from the
  recorded times; the binary itself only emits raw times.
- Warm-up kernel launches are performed before timing to exclude driver
  and JIT compilation overhead.
- **CUDA/HIP**: CUDA/HIP events are used for high-resolution GPU-side timing.
- **Metal**: `mach_absolute_time()` wall-clock timing is used; command buffers
  are serialized (`waitUntilCompleted`) for accurate measurement.
- The kernel loop is unrolled 4× (four independent accumulators) to saturate
  the throughput path of modern GPU integer pipelines.
- `mads_per_thread` is rounded down to the nearest multiple of 4 at runtime to
  match the unrolled loop structure.
- Integer overflow is expected and harmless — the goal is to exercise the
  integer ALU, not produce meaningful results.
