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

## What It Measures

- **Unidirectional H2D / D2H** — peak transfer bandwidth in a single direction.
  Typical values are ~16 GB/s on PCIe 4.0 x16 and ~32 GB/s on PCIe 5.0 x16.
- **Bidirectional** — aggregate throughput when both transfer directions are
  active simultaneously.  On links that support full-duplex operation the
  aggregate can approach 2× the unidirectional value.

## Build

The directory is **self-contained** — no CMake or external dependencies.

### Auto-detect platform

```bash
cd tier1/pcie_bandwidth
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
# Default 256 MiB transfer per direction
./pcie_bandwidth

# Custom transfer size (in MiB)
./pcie_bandwidth 512
```

## Measurement Protocol

Each selected test is run **exactly once** per process execution and timed with
GPU events; there is no in-program warmup or iteration loop. Unless
`BENCH_PARAM_vector_add_warmup=false`, a vector-add warmer runs immediately
before the timed transfer to boost the GPU clock and evict the last-level
cache, so the transfer is measured cold-cache at a boosted clock. Repetition is
done by re-running the process (`gpubench run --reps N`).

## Example Output

Progress and device info go to stderr; stdout carries JSON lines — one record
per selected test, plus a final summary (shown here for `transfer_type=all`):

```
{"type":"kernel","name":"memcpy_h2d","iter":0,"time_ms":10.482133,"host_time_ms":10.598210,"params":{"buffer_size_mb":256,"transfer_type":"all","use_pinned_memory":"true"}}
{"type":"kernel","name":"memcpy_d2h","iter":0,"time_ms":10.213450,"host_time_ms":10.330122,"params":{"buffer_size_mb":256,"transfer_type":"all","use_pinned_memory":"true"}}
{"type":"kernel","name":"memcpy_bidir","iter":0,"time_ms":10.791022,"host_time_ms":10.902311,"params":{"buffer_size_mb":256,"transfer_type":"all","use_pinned_memory":"true"}}
{"type":"summary","total_time_ms":31.486605}
```

- Each test (`memcpy_h2d`, `memcpy_d2h`, `memcpy_bidir`) emits one record — the
  operation runs once, no averaging.
- `time_ms` is the GPU event time for that transfer; `host_time_ms` is the
  host wall-clock around it.
- `total_time_ms` is the sum of the emitted records' GPU event times.
- GB/s is not printed; it is derived downstream from `time_ms` and
  `buffer_size_mb` (bidirectional moves 2 × buffer bytes per transfer).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BENCH_PARAM_buffer_size_mb` (or argv[1]) | 256 | Transfer buffer size in MiB per direction |
| `BENCH_PARAM_transfer_type` | all | Test(s) to run (`h2d`, `d2h`, `bidir`, `all`) |
| `BENCH_PARAM_use_pinned_memory` | true | Use pinned (page-locked) host memory |
| `BENCH_PARAM_vector_add_warmup` | true | Run the cold-cache clock warmer before the timed transfer (`false` to disable) |

## Notes

- Bandwidth in SI GB/s (1 GB = 10⁹ bytes) is derived downstream from the
  recorded times.
- **CUDA**: CUDA events provide GPU-side timing with microsecond resolution.
  Pinned host memory is allocated with `cudaMallocHost()`.
- **HIP**: HIP events provide GPU-side timing with microsecond resolution.
  Pinned host memory is allocated with `hipHostMalloc()`.
- **CUDA/HIP bidirectional**: H2D and D2H transfers are issued on two
  independent async streams, allowing the GPU DMA engine to overlap them
  where the hardware supports full-duplex operation.
