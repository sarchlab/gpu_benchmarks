# warp_shuffle — Tier 1 Microbenchmark

## Overview

`warp_shuffle` measures warp/wave/SIMD shuffle instruction throughput by
executing a tight chain of shuffle operations on every GPU thread.

Each thread performs `ops_per_thread` shuffle operations in a dependency
chain (each shuffle uses the result of the previous one). The binary emits
raw per-iteration kernel times; the effective data movement rate is derived
downstream from each record's `time_ms` and params:

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

Human-readable progress goes to stderr; stdout carries JSON lines — one
record per timed iteration of each selected variant, plus a final summary:

```
{"type":"kernel","name":"shfl_xor_kernel","iter":0,"time_ms":2.469123,"host_time_ms":2.561234,"params":{"ops_per_thread":4096,"num_blocks":1024,"threads_per_block":256}}
...
{"type":"kernel","name":"shfl_kernel","iter":0,"time_ms":2.513456,"host_time_ms":2.601234,"params":{...}}
...
{"type":"kernel","name":"shfl_up_kernel","iter":0,"time_ms":2.578901,"host_time_ms":2.671234,"params":{...}}
...
{"type":"kernel","name":"shfl_down_kernel","iter":0,"time_ms":2.491234,"host_time_ms":2.581234,"params":{...}}
...
{"type":"summary","total_time_ms":70.512345}
```

- Each variant emits one record per timed iteration — no averaging.
- `time_ms` is the GPU event time for that launch; `host_time_ms` is the
  host wall-clock around launch + sync.
- `total_time_ms` is the sum of all emitted iterations' GPU event times
  across every variant that ran.
- GB/s is not printed; it is derived downstream from `time_ms` and params.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BENCH_PARAM_ops_per_thread` (or argv[1]) | 4096 | Shuffle iterations per thread |
| `BENCH_PARAM_threads_per_block` | 256 | Threads per block / threadgroup |
| `BENCH_PARAM_num_blocks` | 1024 | Number of blocks in the dispatch grid |
| `BENCH_PARAM_shuffle_variant` | all | Variant(s) to run (`shfl_xor`, `shfl`, `shfl_up`, `shfl_down`, `all`) |
| `BENCH_PARAM_num_warmup` | 3 | Warm-up launches per variant (not timed) |
| `BENCH_PARAM_num_iters` | 7 | Timed launches per variant, each recorded individually |

## Notes

- GB/s (effective data moved through the shuffle network) is derived
  downstream from the recorded times; the binary itself only emits raw times.
- Warm-up kernel launches are performed before timing to exclude driver and
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
