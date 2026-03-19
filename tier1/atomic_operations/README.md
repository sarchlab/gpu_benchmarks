# atomic_operations — Tier 1 Microbenchmark

## Overview

`atomic_operations` measures the throughput of atomic operations on global
memory by having all GPU threads contend on a single shared counter. Three
atomic operations are benchmarked:

| Operation   | HIP                | Metal                                         |
|-------------|--------------------|-----------------------------------------------|
| atomicAdd   | `atomicAdd()`      | `atomic_fetch_add_explicit()`                 |
| atomicCAS   | `atomicCAS()`      | `atomic_compare_exchange_weak_explicit()`     |
| atomicMin   | `atomicMin()`      | `atomic_fetch_min_explicit()`                 |

The achieved throughput in Giga-Operations per second is:

```
Gops = (num_threads × ops_per_thread) / elapsed_seconds / 1e9
```

## Platform Support

| Platform       | Source file(s)                                              | Compiler |
|----------------|-------------------------------------------------------------|----------|
| NVIDIA CUDA    | `atomic_operations.hip`                                     | `nvcc`   |
| AMD ROCm/HIP   | `atomic_operations.hip`                                     | `hipcc`  |
| Apple Metal    | `atomic_operations_metal.mm` + `atomic_operations.metal`    | `clang++`|

## What It Measures

- **Atomic unit throughput** — maximum contention scenario where all threads
  atomically update a single global memory location. This measures the peak
  throughput of the GPU's atomic hardware units (L2 cache atomics on NVIDIA,
  dedicated atomic units on AMD/Apple).
- Each operation type is benchmarked independently with 5 warmup and 5 timed
  iterations for stable results.

## Build

The directory is **self-contained** — no CMake or external dependencies.

### Auto-detect platform

```bash
cd tier1/atomic_operations
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
# Default 1024 ops per thread
./atomic_operations

# Custom ops per thread (increase for faster GPUs)
./atomic_operations 4096
```

> **Note (Metal):** The `atomic_operations.metal` shader file must be present in
> the same directory as the compiled binary — the host code loads it at runtime.

## Output Format

### Human-readable (stderr)

```
Device          : Apple M3 Max
Grid            : 1024 blocks × 256 threads = 262144 total threads
Ops / thread    : 1024
Warmup iters    : 5
Timed iters     : 5

atomicAdd    : 125.432 ms total, 25.086 ms/kernel, 10.72 Gops/s
atomicCAS    : 198.765 ms total, 39.753 ms/kernel, 6.78 Gops/s
atomicMin    : 134.210 ms total, 26.842 ms/kernel, 10.03 Gops/s

Done.
```

### CSV (stdout)

```
atomic_ops,atomicAdd_global,25.086,10.72
atomic_ops,atomicCAS_global,39.753,6.78
atomic_ops,atomicMin_global,26.842,10.03
```

CSV columns: `benchmark_name,operation_memory,avg_time_ms,Gops`

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ops_per_thread` (argv[1]) | 1024 | Number of atomic operations each thread performs |
| `THREADS_PER_BLOCK` (compile-time) | 256 | Threads per block / threadgroup |
| `NUM_BLOCKS` (compile-time) | 1024 | Number of blocks in the dispatch grid |
| `NUM_WARMUP` (compile-time) | 5 | Warm-up kernel launches (not timed) |
| `NUM_ITERATIONS` (compile-time) | 5 | Timed kernel repetitions for averaging |

## Notes

- Results are in Gops/s (10⁹ atomic operations per second).
- 5 warm-up launches are performed before timing to exclude driver and JIT
  compilation overhead.
- **CUDA/HIP**: HIP events are used for high-resolution GPU-side timing.
- **Metal**: `mach_absolute_time()` wall-clock timing is used; command buffers
  are serialised (`waitUntilCompleted`) for accurate measurement.
- All threads contend on a single global counter — this is the worst-case
  contention scenario and reflects the hardware atomic throughput limit.
- atomicCAS uses a read-then-CAS pattern: each thread reads the current value
  and attempts a compare-and-swap, regardless of success.
