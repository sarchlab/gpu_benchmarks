# tier2/shoc_scan — SHOC Exclusive Prefix Scan Benchmark

## Overview

The **Scan** benchmark times an **exclusive prefix sum** (scan) operation and
emits per-iteration kernel times; effective GPU memory bandwidth is derived
downstream:

```
output[i] = input[0] + input[1] + … + input[i-1]   (output[0] = 0)
```

This is a classic parallel-computing primitive with read–modify–write memory
access patterns.  The implementation uses a **work-efficient (Blelloch)
two-phase scan** (up-sweep + down-sweep) with shared/threadgroup memory, and
supports arrays of arbitrary size via recursive multi-block scan.  Derived from
the [SHOC benchmark suite](https://github.com/vetter/shoc).

## Files

| File | Description |
|------|-------------|
| `shoc_scan.hip` | Self-contained HIP kernel + host (NVIDIA/AMD) |
| `shoc_scan.metal` | Metal compute shader — `scan_block_kernel` + `add_block_sums_kernel` (Apple) |
| `shoc_scan_metal.mm` | Objective-C++ Metal host with multi-pass recursive scan (Apple) |
| `Makefile` | Auto-detects platform; supports `cuda`, `rocm`, `metal` |

## Algorithm

The Blelloch work-efficient scan runs in O(n) work and O(log n) depth:

1. **Up-sweep (reduce)**: tree reduction building partial sums.
2. **Clear + Down-sweep**: zero the last element and propagate the exclusive
   prefix down the tree.

For arrays larger than a single block:
- **Phase 1** — each block performs a local Blelloch scan and saves its block
  total to `block_sums[]`.
- **Phase 2** — recursively scan `block_sums[]` to get `scanned_block_sums[]`.
- **Phase 3** — add `scanned_block_sums[i]` to every element in block `i`.

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
./shoc_scan [--array_size N] [--block_size B] [--precision P]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--array_size N` | `1048576` | Number of `float` elements (1 M ≈ 4 MiB) |
| `--block_size B` | `256` | Threads per block — elements per block = 2× (HIP/CUDA only) |
| `--precision P` | `float` | Precision label (half, float, double) |

Environment variables `BENCH_PARAM_array_size`, `BENCH_PARAM_block_size`, and
`BENCH_PARAM_precision` override the flags. `BENCH_PARAM_num_warmup` (default
3) sets the untimed warmup passes and `BENCH_PARAM_num_iters` (default 7) the
timed passes, each recorded.

> Note: `--block_size` is ignored on Metal (thread-group size is set automatically).

## Output

**stdout** — JSON lines: per timed pass, one `kernel` record per kernel
(`scan_block_kernel`, `add_block_sums_kernel`) with that kernel's accumulated
GPU event time across all recursion levels in the pass, then a `summary`
whose `total_time_ms` is the sum of all recorded GPU event times:
```json
{"type":"kernel","name":"scan_block_kernel","iter":0,"time_ms":0.185000,"host_time_ms":0.210000,"params":{"array_size":1048576,"block_size":256,"precision":"float"}}
{"type":"kernel","name":"add_block_sums_kernel","iter":0,"time_ms":0.030000,"host_time_ms":0.045000,"params":{"array_size":1048576,"block_size":256,"precision":"float"}}
...
{"type":"summary","total_time_ms":1.505000}
```

Derived metrics (e.g. GB/s) are computed downstream from `time_ms` and params.

**stderr** — human-readable summary:
```
Device: Apple M4 Pro
Array size: 1048576 floats (4.0 MiB, padded to 1048576)
Block size: 256

pass 0: scan gpu = 0.185 ms, add gpu = 0.030 ms
...
Total GPU time: 1.505 ms over 7 passes
PASS
```

## Bandwidth Formula

Derived downstream from the emitted per-iteration times:

```
Bandwidth (GB/s) = 2 × N × sizeof(float) / time_seconds / 1e9   (derived downstream)
```

The factor of 2 accounts for one read (input) and one write (output).

## Example Results

| Platform | Array Size | Avg (ms) | Bandwidth |
|----------|-----------|----------|-----------|
| Apple M4 Pro (Metal) | 1 M floats | ~0.22 | ~38 GB/s |
| NVIDIA RTX 4090 (CUDA) | 1 M floats | ~0.05 | ~160 GB/s |
| AMD RX 7900 XTX (ROCm) | 1 M floats | ~0.07 | ~114 GB/s |

*(Results are indicative and vary by system configuration.)*
