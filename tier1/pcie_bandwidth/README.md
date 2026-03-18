# pcie_bandwidth — Tier 1 Microbenchmark

## Overview

`pcie_bandwidth` measures PCIe transfer bandwidth between the host (CPU) and
the GPU.  Three tests are run in sequence:

| Test | Description |
|------|-------------|
| **H2D (unidirectional)** | Pinned host memory → GPU global memory |
| **D2H (unidirectional)** | GPU global memory → pinned host memory |
| **Bidirectional** | Concurrent H2D and D2H on two CUDA streams; reports aggregate GB/s |

Pinned (page-locked) memory is allocated with `cudaMallocHost()` so the DMA
engine has direct access to the host buffer without incurring OS paging
overhead, yielding the maximum achievable PCIe throughput.

## What It Measures

- **Unidirectional H2D / D2H** — peak PCIe bandwidth in a single direction.
  Typical values are ~16 GB/s on PCIe 4.0 x16 and ~32 GB/s on PCIe 5.0 x16.
- **Bidirectional** — aggregate throughput when both PCIe lanes are used
  simultaneously.  On links that support full-duplex operation the aggregate
  can approach 2× the unidirectional value.

## Build

From the repository root:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target pcie_bandwidth
```

The binary is placed in `build/tier1/pcie_bandwidth/pcie_bandwidth`.

## Run

```bash
# Default 256 MiB transfer per direction
./build/tier1/pcie_bandwidth/pcie_bandwidth

# Custom transfer size (in MiB)
./build/tier1/pcie_bandwidth/pcie_bandwidth 512
```

## Example Output

```
Device          : NVIDIA A100-SXM4-80GB (id=0)
PCIe link width : x0  gen 175
Transfer size   : 256 MiB (268435456 bytes)
Iterations/test : 20

H2D (unidirectional)          :    25.41 GB/s
D2H (unidirectional)          :    26.08 GB/s
Bidirectional (H2D + D2H)     :    49.73 GB/s  (aggregate)

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
  one-time PCIe link-training and TLB-mapping costs.
- CUDA events provide GPU-side timing with microsecond resolution.
- The bidirectional test issues `cudaMemcpyAsync` on two independent streams,
  allowing the GPU DMA engine to overlap H2D and D2H transfers where the
  hardware supports full-duplex PCIe operation.
