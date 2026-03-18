# memory_bandwidth — Tier 1 Microbenchmark

## Overview

`memory_bandwidth` measures peak PCIe and device-memory copy bandwidth by
timing `cudaMemcpy` transfers in three directions:

| Test | Description |
|------|-------------|
| **H2D** | Host → Device (pinned host memory → GPU global memory) |
| **D2H** | Device → Host (GPU global memory → pinned host memory) |
| **D2D** | Device → Device (GPU global memory → GPU global memory) |

Pinned (page-locked) host memory is used for H2D and D2H tests to eliminate
paging overhead and obtain the best possible PCIe throughput.  The D2D test
measures the internal memory copy bandwidth of the GPU.

## What It Measures

- **H2D / D2H bandwidth** — primarily bounded by PCIe link speed (e.g. ~16 GB/s
  on PCIe 4.0 x16) and the host's memory subsystem.
- **D2D bandwidth** — primarily bounded by the GPU's HBM or GDDR memory
  bandwidth (e.g. ~900 GB/s on an A100, ~900 GB/s on an H100).

## Build

From the repository root:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target memory_bandwidth
```

The binary is placed in `build/tier1/memory_bandwidth/memory_bandwidth`.

## Run

```bash
# Default 256 MiB transfer
./build/tier1/memory_bandwidth/memory_bandwidth

# Custom transfer size (in MiB)
./build/tier1/memory_bandwidth/memory_bandwidth 512
```

## Example Output

```
Device: NVIDIA A100-SXM4-80GB (id=0)
Transfer size: 256 MiB (268435456 bytes)
Iterations per test: 20

H2D bandwidth :    25.34 GB/s
D2H bandwidth :    26.11 GB/s
D2D bandwidth :   871.52 GB/s

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
- CUDA events are used for high-resolution GPU-side timing.
