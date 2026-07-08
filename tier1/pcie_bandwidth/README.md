# pcie_bandwidth — Tier 1 Microbenchmark

## Overview

`pcie_bandwidth` measures PCIe (or equivalent memory-fabric) transfer bandwidth
between the host (CPU) and the GPU.  Three tests are run in sequence:

| Test | Description |
|------|-------------|
| **H2D (unidirectional)** | Pinned host memory → GPU global memory |
| **D2H (unidirectional)** | GPU global memory → pinned host memory |
| **Bidirectional** | Concurrent H2D and D2H; measures aggregate throughput |

Pinned (page-locked) host memory is used for H2D and D2H tests to eliminate
paging overhead and obtain the best possible transfer throughput.

## Platform Support

| Platform | Source file | Compiler |
|----------|-------------|---------|
| NVIDIA CUDA | `pcie_bandwidth.hip` | `nvcc` |
| AMD ROCm/HIP | `pcie_bandwidth.hip` | `hipcc` |
| Apple Metal | `pcie_bandwidth_metal.mm` | `clang++` |

## What It Measures

- **Unidirectional H2D / D2H** — peak transfer bandwidth in a single direction.
  Typical values are ~16 GB/s on PCIe 4.0 x16 and ~32 GB/s on PCIe 5.0 x16.
- **Bidirectional** — aggregate throughput when both transfer directions are
  active simultaneously.  On links that support full-duplex operation the
  aggregate can approach 2× the unidirectional value.

### Apple Silicon Note

Apple Silicon uses a **unified memory architecture** — there is no discrete
PCIe bus between the CPU and GPU.  Both processors share the same physical
memory fabric.  The Metal version measures the copy overhead between
`MTLResourceStorageModeShared` (CPU-accessible) and
`MTLResourceStorageModePrivate` (GPU-tiled) buffers via `MTLBlitCommandEncoder`.
This provides an equivalent metric for platform comparison.

## Build

The directory is **self-contained** — no CMake or external dependencies.

### Auto-detect platform

```bash
cd tier1/pcie_bandwidth
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
# Default 256 MiB transfer per direction
./pcie_bandwidth

# Custom transfer size (in MiB)
./pcie_bandwidth 512
```

## Example Output

Progress and device info go to stderr; stdout carries JSON lines — one record
per timed iteration of each selected test, plus a final summary:

```
{"type":"kernel","name":"memcpy_h2d","iter":0,"time_ms":10.482133,"host_time_ms":10.598210,"params":{"buffer_size_mb":256,"transfer_type":"all","use_pinned_memory":"true"}}
...
{"type":"kernel","name":"memcpy_d2h","iter":0,"time_ms":10.213450,"host_time_ms":10.330122,"params":{...}}
...
{"type":"kernel","name":"memcpy_bidir","iter":0,"time_ms":10.791022,"host_time_ms":10.902311,"params":{...}}
...
{"type":"summary","total_time_ms":221.412045}
```

- Each test (`memcpy_h2d`, `memcpy_d2h`, `memcpy_bidir`) emits one record per
  timed iteration — no averaging.
- `time_ms` is the GPU event time for that transfer; `host_time_ms` is the
  host wall-clock around it.
- `total_time_ms` is the sum of all emitted iterations' GPU event times.
- GB/s is not printed; it is derived downstream from `time_ms` and
  `buffer_size_mb` (bidirectional moves 2 × buffer bytes per iteration).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BENCH_PARAM_buffer_size_mb` (or argv[1]) | 256 | Transfer buffer size in MiB per direction |
| `BENCH_PARAM_transfer_type` | all | Test(s) to run (`h2d`, `d2h`, `bidir`, `all`) |
| `BENCH_PARAM_use_pinned_memory` | true | Use pinned (page-locked) host memory |
| `BENCH_PARAM_num_warmup` | 3 | Warm-up transfers per test (not timed) |
| `BENCH_PARAM_num_iters` | 7 | Timed transfers per test, each recorded individually |

## Notes

- Bandwidth in SI GB/s (1 GB = 10⁹ bytes) is derived downstream from the
  recorded times.
- Warm-up transfers are performed before each timed loop to exclude
  one-time link-training and TLB-mapping costs.
- **CUDA**: CUDA events provide GPU-side timing with microsecond resolution.
  Pinned host memory is allocated with `cudaMallocHost()`.
- **HIP**: HIP events provide GPU-side timing with microsecond resolution.
  Pinned host memory is allocated with `hipHostMalloc()`.
- **Metal**: `mach_absolute_time()` wall-clock timing is used; command buffers
  are serialized (`waitUntilCompleted`) for accurate measurement.  The
  bidirectional test encodes both H2D and D2H blits in a single command buffer
  so the Metal runtime can overlap them on the blit engine.
- **CUDA/HIP bidirectional**: H2D and D2H transfers are issued on two
  independent async streams, allowing the GPU DMA engine to overlap them
  where the hardware supports full-duplex operation.
