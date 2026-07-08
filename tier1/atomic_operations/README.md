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

The binary reports raw per-iteration kernel times; throughput in
Giga-Operations per second is derived downstream from the recorded `time_ms`
and params:

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
- Each operation type is benchmarked independently with `BENCH_PARAM_num_warmup`
  (default 3) warmup launches and `BENCH_PARAM_num_iters` (default 7) timed
  iterations; every timed iteration is recorded individually.

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

Progress and device info go to stderr; stdout carries JSON lines, one record
per timed iteration plus a final summary:

```
{"type":"kernel","name":"atomicAdd_kernel","iter":0,"time_ms":25.086432,"host_time_ms":25.201004,"params":{"ops_per_thread":1024,"num_blocks":2048,"threads_per_block":256,"atomic_op":"atomicAdd"}}
...
{"type":"summary","total_time_ms":175.605024}
```

- `time_ms` is the GPU event time for that single kernel launch;
  `host_time_ms` is the host wall-clock around launch + sync.
- Every iteration is emitted individually — no averaging or min-keeping.
- `total_time_ms` is the sum of all iterations' GPU event times.
- No derived metrics are printed; Gops/s is computed downstream from
  `time_ms` and the params in each record.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BENCH_PARAM_ops_per_thread` (or argv[1]) | 1024 | Number of atomic operations each thread performs |
| `BENCH_PARAM_threads_per_block` | 256 | Threads per block / threadgroup |
| `BENCH_PARAM_num_blocks` | 2048 | Number of blocks in the dispatch grid |
| `BENCH_PARAM_atomic_op` | atomicAdd | Atomic operation to benchmark (`atomicAdd`, `atomicCAS`, `atomicMin`) |
| `BENCH_PARAM_num_warmup` | 3 | Warm-up kernel launches (not timed) |
| `BENCH_PARAM_num_iters` | 7 | Timed kernel launches, each recorded individually |

## Notes

- Gops/s (10⁹ atomic operations per second) is derived downstream from the
  recorded times; the binary itself only emits raw times.
- Warm-up launches are performed before timing to exclude driver and JIT
  compilation overhead.
- **CUDA/HIP**: HIP events are used for high-resolution GPU-side timing.
- **Metal**: `mach_absolute_time()` wall-clock timing is used; command buffers
  are serialised (`waitUntilCompleted`) for accurate measurement.
- All threads contend on a single global counter — this is the worst-case
  contention scenario and reflects the hardware atomic throughput limit.
- atomicCAS uses a read-then-CAS pattern: each thread reads the current value
  and attempts a compare-and-swap, regardless of success.
