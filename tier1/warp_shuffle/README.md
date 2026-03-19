# warp_shuffle — Tier 1 Microbenchmark

## Overview

`warp_shuffle` measures warp/wave/SIMD shuffle instruction throughput by
executing a tight chain of shuffle operations on every GPU thread.

Each thread performs `ops_per_thread` shuffle operations in a dependency
chain (each shuffle uses the result of the previous one), and the achieved
throughput is expressed as effective data movement rate:

```
GB/s = (num_threads × ops_per_thread × sizeof(int)) / elapsed_seconds / 1e9
```

Four shuffle variants are benchmarked:

| Variant | HIP Intrinsic | Metal Intrinsic | Description |
|---------|--------------|-----------------|-------------|
| `shfl_xor` | `__shfl_xor` | `simd_shuffle_xor` | Butterfly / XOR shuffle |
| `shfl` | `__shfl` | `simd_shuffle` | Indexed (direct) shuffle |
| `shfl_up` | `__shfl_up` | `simd_shuffle_up` | Shift up by delta |
| `shfl_down` | `__shfl_down` | `simd_shuffle_down` | Shift down by delta |

## Platform Support

| Platform | Source file | Compiler |
|----------|-------------|---------|
| NVIDIA CUDA | `warp_shuffle.hip` | `nvcc -x cu` |
| AMD ROCm/HIP | `warp_shuffle.hip` | `hipcc` |
| Apple Metal | `warp_shuffle_metal.mm` + `warp_shuffle.metal` | `clang++` |

## What It Measures

- **Warp/SIMD shuffle throughput** — the rate at which lanes can exchange data
  via hardware shuffle instructions without going through shared memory or
  global memory.
- Shuffles are a fundamental building block for reductions, scans, transposes,
  and other warp-level primitives.
- The benchmark creates a dependency chain to prevent compiler elimination
  while reflecting the hardware's true shuffle throughput.

## Build

The directory is **self-contained** — no CMake or external dependencies.

### Auto-detect platform

```bash
cd tier1/warp_shuffle
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
# Default 4096 shuffle ops per thread
./warp_shuffle

# Custom ops count per thread
./warp_shuffle 8192
```

> **Note (Metal):** The `warp_shuffle.metal` shader file must be present in
> the same directory as the compiled binary — the host code loads it at runtime.

## Output Format

### Human-readable (stderr)

```
Device          : Apple M2 (id=0)
Grid            : 1024 blocks × 256 threads = 262144 total threads
SIMD width      : 32
Ops / thread    : 4096
Warmup iters    : 5
Timed iters     : 5

shfl_xor     : 12.345 ms total, 2.469 ms/kernel, 428.12 GB/s
shfl         : 12.567 ms total, 2.513 ms/kernel, 420.34 GB/s
shfl_up      : 12.890 ms total, 2.578 ms/kernel, 409.78 GB/s
shfl_down    : 12.456 ms total, 2.491 ms/kernel, 424.01 GB/s

Done.
```

### CSV (stdout)

```
warp_shuffle,shfl_xor,2.469,428.12
warp_shuffle,shfl,2.513,420.34
warp_shuffle,shfl_up,2.578,409.78
warp_shuffle,shfl_down,2.491,424.01
```

Format: `warp_shuffle,<variant>,<time_ms>,<GBs>`

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ops_per_thread` (argv[1]) | 4096 | Shuffle iterations per thread |
| `THREADS_PER_BLOCK` (compile-time) | 256 | Threads per block / threadgroup |
| `NUM_BLOCKS` (compile-time) | 1024 | Number of blocks in the dispatch grid |
| `NUM_WARMUP` (compile-time) | 5 | Warm-up iterations (not timed) |
| `NUM_ITERATIONS` (compile-time) | 5 | Timed kernel repetitions |

## Notes

- Results are in GB/s (effective data moved through shuffle network).
- 5 warm-up kernel launches are performed before timing to exclude driver and
  JIT compilation overhead.
- **CUDA/HIP**: HIP events (`hipEvent_t`) are used for high-resolution
  GPU-side timing. HIP shuffle intrinsics (`__shfl_xor`, etc.) do not require
  the `_sync` suffix — they operate on the full warp/wavefront.
- **Metal**: `mach_absolute_time()` wall-clock timing is used; command buffers
  are serialized (`waitUntilCompleted`) for accurate measurement. Metal SIMD
  width is 32 on Apple GPUs.
- The warp size is 64 on AMD GCN/CDNA GPUs and 32 on NVIDIA GPUs. The HIP
  code automatically selects the correct size at compile time.
- A dependency chain ensures the compiler cannot optimize away the shuffle
  operations; only thread 0 of block 0 writes the final result to prevent
  memory traffic from dominating the measurement.
