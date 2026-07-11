# memory_bandwidth — Tier 1 Microbenchmark

## Overview

`memory_bandwidth` measures peak PCIe and device-memory copy bandwidth by
timing transfers in three directions:

| Test | Description |
|------|-------------|
| **H2D** | Host → Device (pinned host memory → GPU global memory) |
| **D2H** | Device → Host (GPU global memory → pinned host memory) |
| **D2D** | Device → Device (GPU global memory → GPU global memory) |

Pinned (page-locked) host memory is used for H2D and D2H tests to eliminate
paging overhead and obtain the best possible PCIe throughput.  The D2D test
measures the internal memory copy bandwidth of the GPU.

## Platform Support

| Platform | Source file | Compiler |
|----------|-------------|---------|
| NVIDIA CUDA | `memory_bandwidth.hip` | `nvcc` |
| AMD ROCm/HIP | `memory_bandwidth.hip` | `hipcc` |

## What It Measures

- **H2D / D2H bandwidth** — primarily bounded by PCIe link speed (e.g. ~16 GB/s
  on PCIe 4.0 x16) and the host's memory subsystem.
- **D2D bandwidth** — primarily bounded by the GPU's HBM or GDDR memory
  bandwidth (e.g. ~900 GB/s on an A100).

## Build

The directory is **self-contained** — no CMake or external dependencies.

### Auto-detect platform

```bash
cd tier1/memory_bandwidth
make
```

The Makefile auto-detects the platform:
- Linux with `/opt/rocm` → ROCm/HIP
- Otherwise → CUDA

### Explicit platform

```bash
# NVIDIA CUDA
make PLATFORM=cuda

# AMD ROCm/HIP
make PLATFORM=rocm
```

### Clean

```bash
make clean
```

## Run

```bash
# Default 256 MiB transfer (D2D)
./memory_bandwidth

# Custom transfer size (in MiB)
./memory_bandwidth 512

# All three directions
BENCH_PARAM_transfer_type=all ./memory_bandwidth
```

## Measurement Protocol

Each selected direction is copied **exactly once** per process execution and
timed with GPU events; there is no in-program warmup or iteration loop. Unless
`BENCH_PARAM_vector_add_warmup=false`, a vector-add warmer runs immediately
before the timed copy to boost the GPU clock and evict the last-level cache, so
the copy is measured cold-cache at a boosted clock. Repetition is done by
re-running the process (`gpubench run --reps N`).

## Example Output

Progress and device info go to stderr; stdout carries JSON lines — one record
per selected direction, plus a final summary (shown here for
`transfer_type=all`):

```
{"type":"kernel","name":"memcpy_h2d","iter":0,"time_ms":10.482133,"host_time_ms":10.598210,"params":{"buffer_size_mb":256,"transfer_type":"all","use_pinned_memory":"true"}}
{"type":"kernel","name":"memcpy_d2h","iter":0,"time_ms":10.213450,"host_time_ms":10.330122,"params":{"buffer_size_mb":256,"transfer_type":"all","use_pinned_memory":"true"}}
{"type":"kernel","name":"memcpy_d2d","iter":0,"time_ms":0.612340,"host_time_ms":0.701234,"params":{"buffer_size_mb":256,"transfer_type":"all","use_pinned_memory":"true"}}
{"type":"summary","total_time_ms":21.307923}
```

- Each selected direction (`memcpy_h2d`, `memcpy_d2h`, `memcpy_d2d`) emits one
  record — the operation runs once, no averaging.
- `time_ms` is the GPU event time for that single transfer; `host_time_ms` is
  the host wall-clock around it.
- `total_time_ms` is the sum of the emitted records' GPU event times.
- Bandwidth (GB/s) is not printed; it is derived downstream from `time_ms`
  and `buffer_size_mb`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BENCH_PARAM_buffer_size_mb` (or argv[1]) | 256 | Transfer buffer size in MiB |
| `BENCH_PARAM_transfer_type` | d2d | Direction(s) to run (`h2d`, `d2h`, `d2d`, `all`) |
| `BENCH_PARAM_use_pinned_memory` | true | Use pinned (page-locked) host memory |
| `BENCH_PARAM_vector_add_warmup` | true | Run the cold-cache clock warmer before the timed copy (`false` to disable) |

## Notes

- Bandwidth in SI gigabytes per second (1 GB = 10⁹ bytes) is derived
  downstream from the recorded times.
- **Byte-accounting convention.** A device-to-device copy generates 2×
  `buffer_size_mb` of HBM traffic (one read + one write), whereas H2D and D2H
  each move 1× across the link. The downstream formulas are:
  - D2D: `GB/s = 2 × buffer_size_mb × 2²⁰ / time_ms / 1e6`
  - H2D / D2H: `GB/s = buffer_size_mb × 2²⁰ / time_ms / 1e6`
- **CUDA/HIP**: CUDA/HIP events are used for high-resolution GPU-side timing.
