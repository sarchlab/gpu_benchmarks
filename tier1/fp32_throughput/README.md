# fp32_throughput — Tier 1 Microbenchmark

## Overview

`fp32_throughput` measures peak single-precision (FP32) floating-point compute
throughput by executing a tight chain of fused multiply-add (FMA) instructions
on every GPU thread.

Because each FMA counts as **2 floating-point operations** (1 multiply + 1
add), and all values live in registers (no memory traffic), the result directly
reflects the arithmetic throughput of the GPU rather than memory bandwidth.

The binary emits raw per-iteration kernel times; GFLOPS is derived downstream
from each record's `time_ms` and params:

```
GFLOPS = (num_threads × fmas_per_thread × 2) / elapsed_seconds / 1e9
```

## Platform Support

| Platform | Source file | Compiler |
|----------|-------------|---------|
| NVIDIA CUDA | `fp32_throughput.hip` | `nvcc` |
| AMD ROCm/HIP | `fp32_throughput.hip` | `hipcc` |
| Apple Metal | `fp32_throughput_metal.mm` + `fp32_throughput.metal` | `clang++` |

## What It Measures

- **FP32 arithmetic throughput** — bounded by the number of FP32 FMA units and
  their clock frequency (e.g. ~20 TFLOPS on an RTX 4090, ~10 TFLOPS on an M3 Max).
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
# Default 4096 FMAs per thread
./fp32_throughput

# Custom FMA count per thread (increase for faster GPUs)
./fp32_throughput 16384
```

> **Note (Metal):** The `fp32_throughput.metal` shader file must be present in
> the same directory as the compiled binary — the host code loads it at runtime.

## Example Output

Progress and device info go to stderr; stdout carries JSON lines — one record
per timed iteration plus a final summary:

```
{"type":"kernel","name":"fp32_fma_kernel","iter":0,"time_ms":0.524128,"host_time_ms":0.601342,"params":{"fmas_per_thread":4096,"num_blocks":1,"threads_per_block":256}}
...
{"type":"summary","total_time_ms":3.668901}
```

- `time_ms` is the GPU event time for that single kernel launch;
  `host_time_ms` is the host wall-clock around launch + sync.
- Every timed iteration is emitted individually — no averaging.
- `total_time_ms` is the sum of all iterations' GPU event times.
- GFLOPS is not printed; it is derived downstream from `time_ms` and params.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BENCH_PARAM_fmas_per_thread` (or argv[1]) | 4096 | FMA iterations per thread (rounded to nearest multiple of 4) |
| `BENCH_PARAM_threads_per_block` | 256 | Threads per block / threadgroup |
| `BENCH_PARAM_num_blocks` | 1024 | Number of blocks in the dispatch grid (params.json pins 1 for harness sweeps) |
| `BENCH_PARAM_num_warmup` | 3 | Warm-up kernel launches (not timed) |
| `BENCH_PARAM_num_iters` | 7 | Timed kernel launches, each recorded individually |

## Notes

- GFLOPS (10⁹ floating-point operations per second) is derived downstream
  from the recorded times; the binary itself only emits raw times.
- Warm-up kernel launches are performed before timing to exclude driver and
  JIT compilation overhead.
- **CUDA/HIP**: CUDA/HIP events are used for high-resolution GPU-side timing.
- **Metal**: `mach_absolute_time()` wall-clock timing is used; command buffers
  are serialized (`waitUntilCompleted`) for accurate measurement.
- The kernel loop is unrolled 4× (four independent accumulators) to saturate
  the throughput path of modern GPU FP32 pipelines.
- `fmas_per_thread` is rounded down to the nearest multiple of 4 at runtime to
  match the unrolled loop structure.
