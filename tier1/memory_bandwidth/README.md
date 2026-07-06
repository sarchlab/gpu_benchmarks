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

### CUDA / HIP

```
Device: NVIDIA A100-SXM4-80GB (id=0)
Transfer size: 256 MiB (268435456 bytes)
Iterations per test: 20

H2D bandwidth :    25.34 GB/s
D2H bandwidth :    26.11 GB/s
D2D bandwidth :   871.52 GB/s

Done.
```

### Metal (Apple M3)

```
Device: Apple M3 Max
Transfer size: 256 MiB (268435456 bytes)
Iterations per test: 20

H2D bandwidth :    58.12 GB/s
D2H bandwidth :    62.44 GB/s
D2D bandwidth :   391.87 GB/s

Done.
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `buffer_size_mb` (argv[1]) | 256 | Transfer buffer size in MiB |
| `NUM_ITERATIONS` (compile-time) | 20 | Repetitions averaged per test |

## Notes

- Results are in SI gigabytes per second (1 GB = 10⁹ bytes).
- A warm-up transfer is performed before timing to exclude one-time setup costs.
- **CUDA/HIP**: CUDA/HIP events are used for high-resolution GPU-side timing.
- **Metal**: `mach_absolute_time()` wall-clock timing is used; command buffers
  are serialized (`waitUntilCompleted`) for accurate measurement.
- **Metal D2D**: Uses the `d2d_copy` compute kernel (float4 vectorized loads/stores).
  The reported bandwidth accounts for both the read and write passes
  (i.e. `2 × bytes × iterations / time`).
- Metal H2D/D2H use `MTLResourceStorageModeShared` (CPU-accessible) ↔
  `MTLResourceStorageModePrivate` (GPU-only) blits via `MTLBlitCommandEncoder`.
