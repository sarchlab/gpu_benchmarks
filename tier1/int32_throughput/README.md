# int32_throughput — Tier 1 Microbenchmark

## Overview

`int32_throughput` measures peak 32-bit integer (INT32) compute throughput by
executing a tight chain of integer multiply-add (MAD) instructions on every
GPU thread.

Because each MAD counts as **2 integer operations** (1 multiply + 1 add), and
all values live in registers (no memory traffic), the result directly reflects
the arithmetic throughput of the GPU rather than memory bandwidth.

```
GIOPS = (num_threads × mads_per_thread × 2) / elapsed_seconds / 1e9
```

## Platform Support

| Platform | Source file | Compiler |
|----------|-------------|---------|
| NVIDIA CUDA | `int32_throughput.cu` | `nvcc` |
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

### CUDA / HIP

```
Device          : NVIDIA RTX 4090 (id=0)
SMs             : 128
Grid            : 1024 blocks × 256 threads = 262144 total threads
MADs / thread   : 4096
Warmup iters    : 5
Timed iters     : 5

Elapsed time    : 2.620 ms (0.524 ms / kernel)
Total INT ops   : 10737418240
INT32 Throughput : 4097.82 GIOPS

Done.
```

### Metal (Apple M3)

```
Device          : Apple M3 Max
Grid            : 1024 blocks × 256 threads = 262144 total threads
MADs / thread   : 4096
Warmup iters    : 5
Timed iters     : 5

Elapsed time    : 4.307 ms (0.861 ms / kernel)
Total INT ops   : 10737418240
INT32 Throughput : 2493.11 GIOPS

Done.
```

## CSV Output

The benchmark writes a single CSV line to **stdout** for machine parsing:

```
int32_throughput,<total_ops>,<time_ms>,<GIOPS>
```

All human-readable diagnostic output goes to **stderr**.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mads_per_thread` (argv[1]) | 4096 | MAD iterations per thread (rounded to nearest multiple of 4) |
| `THREADS_PER_BLOCK` (compile-time) | 256 | Threads per block / threadgroup |
| `NUM_BLOCKS` (compile-time) | 1024 | Number of blocks in the dispatch grid |
| `NUM_WARMUP` (compile-time) | 5 | Warmup kernel launches before timing |
| `NUM_ITERATIONS` (compile-time) | 5 | Kernel repetitions averaged for timing |

## Notes

- Results are in GIOPS (10⁹ integer operations per second).
- Five warm-up kernel launches are performed before timing to exclude driver
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
