# pcie_bandwidth — Tier 1 Microbenchmark

## Overview

`pcie_bandwidth` measures PCIe (or equivalent memory-fabric) transfer bandwidth
between the host (CPU) and the GPU.  Three tests are run in sequence:

| Test | Description |
|------|-------------|
| **H2D (unidirectional)** | Pinned host memory → GPU global memory |
| **D2H (unidirectional)** | GPU global memory → pinned host memory |
| **Bidirectional** | Concurrent H2D and D2H; reports aggregate GB/s |

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

### CUDA / HIP

```
Device          : NVIDIA A100-SXM4-80GB (id=0)
Transfer size   : 256 MiB (268435456 bytes)
Iterations/test : 20

H2D (unidirectional)          :    25.41 GB/s
D2H (unidirectional)          :    26.08 GB/s
Bidirectional (H2D + D2H)     :    49.73 GB/s  (aggregate)

Done.
```

### Metal (Apple M3)

```
Device          : Apple M3 Max
Transfer size   : 256 MiB (268435456 bytes)
Iterations/test : 20
Note            : Apple Silicon uses unified memory; H2D/D2H
                  measures shared↔private memory fabric copy overhead.

H2D (unidirectional)          :    58.34 GB/s
D2H (unidirectional)          :    61.92 GB/s
Bidirectional (H2D + D2H)     :   112.45 GB/s  (aggregate)

Done.
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `buffer_size_mb` (argv[1]) | 256 | Transfer buffer size in MiB per direction |
| `NUM_ITERATIONS` (compile-time) | 20 | Repetitions averaged per test |

## Notes

- All bandwidth values are in SI GB/s (1 GB = 10⁹ bytes).
- A single warm-up transfer is performed before each timed loop to exclude
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
