# tier2/shoc_reduction — SHOC Parallel Sum Reduction Benchmark

## Overview

The **Reduction** benchmark performs a parallel sum reduction over a large float array, collapsing it to a single scalar result, and emits per-pass GPU/host kernel times (effective memory bandwidth is derived downstream).  It uses a **tree-based shared-memory (threadgroup) reduction**:

- Each thread block (threadgroup) processes **2 × block_size** elements.
- Threads cooperatively reduce to a single partial sum per block.
- Multiple passes are applied until one value remains.

This benchmark is derived from the [SHOC benchmark suite](https://github.com/vetter/shoc).

## Files

| File | Description |
|------|-------------|
| `shoc_reduction.hip` | Self-contained HIP kernel + host (NVIDIA/AMD) |
| `shoc_reduction.metal` | Metal compute shader (Apple) |
| `shoc_reduction_metal.mm` | Objective-C++ Metal host (Apple) |
| `Makefile` | Auto-detects platform; supports `cuda`, `rocm`, `metal` |

## Building

```bash
# Auto-detect (Darwin → metal, /opt/rocm present → rocm, else → cuda)
make

# Explicit platform
make PLATFORM=metal
make PLATFORM=rocm
make PLATFORM=cuda
```

**Requirements:**
- **CUDA**: `nvcc` (CUDA toolkit ≥ 10)
- **ROCm/HIP**: `hipcc` (ROCm ≥ 4.0)
- **Metal**: macOS 12+, Xcode Command Line Tools

## Running

```bash
./shoc_reduction [--size N] [--block_size B] [--precision P]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--size N` | `4194304` | Number of `float` elements (4 M ≈ 16 MiB) |
| `--block_size B` | `256` | GPU thread-block size (HIP/CUDA only) |
| `--precision P` | `float` | One of `float`, `double`, `half` |

Environment variables override argv:

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_size` | `4194304` | Number of elements |
| `BENCH_PARAM_block_size` | `256` | Thread-block size |
| `BENCH_PARAM_precision` | `float` | One of `float`, `double`, `half` |
| `BENCH_PARAM_num_warmup` | `3` | Warmup passes (untimed) |
| `BENCH_PARAM_num_iters` | `7` | Timed passes, each recorded |

> Note: `--block_size` is ignored on Metal (threadgroup size is set automatically from the pipeline).

## Output

**stdout** — JSON-lines: one `{"type":"kernel","name":"reduce_kernel",...}` record per timed pass (one pass = the full multi-level reduction) carrying the pass's accumulated GPU event time (`time_ms`) over all `reduce_kernel` launches and the host wall-clock time (`host_time_ms`), then a summary whose `total_time_ms` is the sum of all recorded GPU event times:
```json
{"type":"kernel","name":"reduce_kernel","iter":0,"time_ms":0.1821,"host_time_ms":0.2010,"params":{"size":4194304,"block_size":256,"precision":"float"}}
{"type":"summary","total_time_ms":1.2747}
```

Derived metrics such as effective bandwidth are computed downstream from `time_ms` and params.

**stderr** — human-readable summary:
```
Device: Apple M4 Pro
Array size: 4194304 floats (16.0 MiB)
Block size: 256  |  3 warmup + 7 timed passes

pass 0: gpu = 0.182 ms, host = 0.201 ms
...
Total GPU time: 1.275 ms over 7 passes
GPU sum: 499500.000000, CPU sum: 499500.000000, relative error: 0.000000e+00
PASS
```

## Bandwidth Formula

Effective bandwidth is derived downstream from the emitted `time_ms` and params as:

```
Bandwidth (GB/s) = 2 × N × sizeof(float) / time_seconds / 1e9
```

The factor of 2 accounts for reading N floats and writing the reduced result.

## Algorithm

### HIP / CUDA

```
Pass 1: reduce_kernel<<<num_blocks, B, B*sizeof(float)>>>(input, partial, N)
Pass 2: reduce_kernel<<<1, B, B*sizeof(float)>>>(partial, result, num_blocks)
```

Each block loads two elements per thread into shared memory, then performs
a binary tree reduction until one partial sum per block is produced.

### Metal

The same two-pass strategy is applied using `threadgroup` memory.  The
`reduce_kernel` shader is compiled at runtime from `shoc_reduction.metal`.
Passes are chained using ping-pong scratch buffers until a single value
remains.

## Example Results

| Platform | Array Size | Avg (ms) | Bandwidth |
|----------|-----------|----------|-----------|
| Apple M4 Pro (Metal) | 4 M floats | ~0.18 | ~184 GB/s |
| NVIDIA RTX 4090 (CUDA) | 4 M floats | ~0.04 | ~820 GB/s |
| AMD RX 7900 XTX (ROCm) | 4 M floats | ~0.06 | ~545 GB/s |

*(Results are indicative and vary by system configuration.)*
