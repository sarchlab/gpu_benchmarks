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
| Apple Metal | `memory_bandwidth_metal.mm` + `memory_bandwidth.metal` | `clang++` |

## What It Measures

- **H2D / D2H bandwidth** — primarily bounded by PCIe link speed (e.g. ~16 GB/s
  on PCIe 4.0 x16) and the host's memory subsystem.
- **D2D bandwidth** — primarily bounded by the GPU's HBM or GDDR memory
  bandwidth (e.g. ~900 GB/s on an A100, ~400 GB/s on an M3 Max unified memory).

## Build

The directory is **self-contained** — no CMake or external dependencies.

### Auto-detect platform

```bash
cd tier1/memory_bandwidth
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
# Default 256 MiB transfer
./memory_bandwidth

# Custom transfer size (in MiB)
./memory_bandwidth 512
```

## Example Output

Progress and device info go to stderr; stdout carries JSON lines — one record
per timed iteration of each selected direction, plus a final summary:

```
{"type":"kernel","name":"memcpy_h2d","iter":0,"time_ms":10.482133,"host_time_ms":10.598210,"params":{"buffer_size_mb":256,"transfer_type":"all","use_pinned_memory":"true"}}
...
{"type":"kernel","name":"memcpy_d2h","iter":0,"time_ms":10.213450,"host_time_ms":10.330122,"params":{...}}
...
{"type":"kernel","name":"memcpy_d2d","iter":0,"time_ms":0.612340,"host_time_ms":0.701234,"params":{...}}
...
{"type":"summary","total_time_ms":149.312045}
```

- Each direction (`memcpy_h2d`, `memcpy_d2h`, `memcpy_d2d`) emits one record
  per timed iteration — no averaging.
- `time_ms` is the GPU event time for that single transfer; `host_time_ms` is
  the host wall-clock around it.
- `total_time_ms` is the sum of all emitted iterations' GPU event times.
- Bandwidth (GB/s) is not printed; it is derived downstream from `time_ms`
  and `buffer_size_mb`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BENCH_PARAM_buffer_size_mb` (or argv[1]) | 256 | Transfer buffer size in MiB |
| `BENCH_PARAM_transfer_type` | all | Direction(s) to run (`h2d`, `d2h`, `d2d`, `all`) |
| `BENCH_PARAM_use_pinned_memory` | true | Use pinned (page-locked) host memory |
| `BENCH_PARAM_warmup` | 3 | Warm-up transfers per direction (not timed) |
| `BENCH_PARAM_iteration` | 7 | Timed transfers per direction, each recorded individually |

## Notes

- Bandwidth in SI gigabytes per second (1 GB = 10⁹ bytes) is derived
  downstream from the recorded times.
- Warm-up transfers are performed before timing to exclude one-time setup costs.
- **CUDA/HIP**: CUDA/HIP events are used for high-resolution GPU-side timing.
- **Metal**: `mach_absolute_time()` wall-clock timing is used; command buffers
  are serialized (`waitUntilCompleted`) for accurate measurement.
- **Metal D2D**: Uses the `d2d_copy` compute kernel (float4 vectorized loads/stores).
  The reported bandwidth accounts for both the read and write passes
  (i.e. `2 × bytes × iterations / time`).
- Metal H2D/D2H use `MTLResourceStorageModeShared` (CPU-accessible) ↔
  `MTLResourceStorageModePrivate` (GPU-only) blits via `MTLBlitCommandEncoder`.
