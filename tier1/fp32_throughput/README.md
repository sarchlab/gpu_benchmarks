# fp32_throughput — Tier 1 Microbenchmark

## Overview

`fp32_throughput` measures peak single-precision (FP32) floating-point compute
throughput by executing a tight chain of fused multiply-add (FMA) instructions
on every GPU thread.

Because each FMA counts as **2 floating-point operations** (1 multiply + 1
add), and all values live in registers (no memory traffic), the result directly
reflects the arithmetic throughput of the GPU rather than memory bandwidth.

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

### CUDA / HIP

```
Device          : NVIDIA RTX 4090 (id=0)
SMs             : 128
Grid            : 1024 blocks × 256 threads = 262144 total threads
FMAs / thread   : 4096
Kernel repeats  : 10

Elapsed time    : 5.241 ms (0.524 ms / kernel)
Total FLOPs     : 21474836480
FP32 Throughput : 4097.82 GFLOPS

Done.
```

### Metal (Apple M3)

```
Device          : Apple M3 Max
Grid            : 1024 blocks × 256 threads = 262144 total threads
FMAs / thread   : 4096
Kernel repeats  : 10

Elapsed time    : 8.613 ms (0.861 ms / kernel)
Total FLOPs     : 21474836480
FP32 Throughput : 2493.11 GFLOPS

Done.
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fmas_per_thread` (argv[1]) | 4096 | FMA iterations per thread (rounded to nearest multiple of 4) |
| `THREADS_PER_BLOCK` (compile-time) | 256 | Threads per block / threadgroup |
| `NUM_BLOCKS` (compile-time) | 1024 | Number of blocks in the dispatch grid |
| `NUM_ITERATIONS` (compile-time) | 10 | Kernel repetitions averaged for timing |

## Notes

- Results are in GFLOPS (10⁹ floating-point operations per second).
- A warm-up kernel launch is performed before timing to exclude driver and JIT
  compilation overhead.
- **CUDA/HIP**: CUDA/HIP events are used for high-resolution GPU-side timing.
- **Metal**: `mach_absolute_time()` wall-clock timing is used; command buffers
  are serialized (`waitUntilCompleted`) for accurate measurement.
- The kernel loop is unrolled 4× (four independent accumulators) to saturate
  the throughput path of modern GPU FP32 pipelines.
- `fmas_per_thread` is rounded down to the nearest multiple of 4 at runtime to
  match the unrolled loop structure.
