# tier2/shoc_triad — SHOC Stream Triad Benchmark

## Overview

The **Stream Triad** benchmark measures effective GPU memory bandwidth by executing the classic operation:

```
a[i] = b[i] + scalar * c[i]
```

This pattern reads two arrays (`b`, `c`) and writes one (`a`), for a total of **3 × N × 4 bytes** of memory traffic per iteration.  It is derived from the [SHOC benchmark suite](https://github.com/vetter/shoc).

## Files

| File | Description |
|------|-------------|
| `shoc_triad.hip` | Self-contained HIP kernel + host (NVIDIA/AMD) |
| `shoc_triad.metal` | Metal compute shader (Apple) |
| `shoc_triad_metal.mm` | Objective-C++ Metal host (Apple) |
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
./shoc_triad [--array_size N] [--block_size B]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--array_size N` | `4194304` | Number of `float` elements (4 M ≈ 16 MiB per array) |
| `--block_size B` | `256` | GPU thread-block size (HIP/CUDA only) |

Environment variables `BENCH_PARAM_array_size`, `BENCH_PARAM_block_size`, and
`BENCH_PARAM_precision` override the flags. In addition:

| Env var | Default | Description |
|---------|---------|-------------|
| `BENCH_PARAM_num_warmup` | 3 | Warmup launches (untimed) |
| `BENCH_PARAM_num_iters` | 7 | Timed launches, each recorded |

> Note: `--block_size` is ignored on Metal (thread-group size is set automatically).

## Output

**stdout** — JSON lines, one `kernel` record per timed iteration carrying the
GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`), then a
`summary` whose `total_time_ms` is the sum of all iterations' GPU event times:
```
{"type":"kernel","name":"triad_kernel","iter":0,"time_ms":0.314200,"host_time_ms":0.330000,"params":{"array_size":4194304,"block_size":256,"precision":"float"}}
...
{"type":"summary","total_time_ms":2.199400}
```

**stderr** — human-readable summary:
```
Device: Apple M4 Pro
Array size: 4194304 floats (16.0 MiB)
Block size: 256

triad_kernel iter 0: gpu = 0.314 ms, host = 0.330 ms
...
Total GPU time: 2.199 ms over 7 iterations
PASS
```

## Bandwidth Formula

```
Bandwidth (GB/s) = 3 × N × sizeof(float) / time_seconds / 1e9
```

The factor of 3 accounts for two reads (`b`, `c`) and one write (`a`).
Derived downstream from the emitted `time_ms` and params.

## Example Results

| Platform | Array Size | Avg (ms) | Bandwidth |
|----------|-----------|----------|-----------|
| Apple M4 Pro (Metal) | 4 M floats | ~0.31 | ~153 GB/s |
| NVIDIA RTX 4090 (CUDA) | 4 M floats | ~0.08 | ~600 GB/s |
| AMD RX 7900 XTX (ROCm) | 4 M floats | ~0.11 | ~450 GB/s |

*(Results are indicative and vary by system configuration.)*
