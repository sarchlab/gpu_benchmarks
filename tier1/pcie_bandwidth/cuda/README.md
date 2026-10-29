# pcie_bandwidth — Tier 1 Microbenchmark (native CUDA)

Native CUDA version of this benchmark. Kernels, driver, verification and
output are identical to [`../hip/`](../hip/); only runtime API names differ.
It builds with `nvcc` only, uses no architecture-specific features, and targets
A100 (sm_80) and H100 (sm_90). Shared timing/warmup headers come from
`common_cuda/`.

## Overview

`pcie_bandwidth` records transfer times between host memory and GPU memory.
Three tests run in sequence by default; derive payload rates for each mode:

| Test | Description |
|------|-------------|
| **H2D (unidirectional)** | Host memory → GPU global memory |
| **D2H (unidirectional)** | GPU global memory → host memory |
| **Bidirectional** | H2D and D2H submitted on two streams; aggregate payload rate |

The manifest sweeps pinned (page-locked) and pageable host allocations. Keep
these cases separate: allocation type affects staging, synchronization, and
whether the submitted transfers can overlap.

## Platform Support

| Platform | Source file | Compiler |
|----------|-------------|---------|
| NVIDIA CUDA | `pcie_bandwidth.cu` | `nvcc` |

## What It Measures

- **Unidirectional H2D / D2H** — payload rate for the selected direction,
  allocation type, and buffer size: `B / (time_ms * 1e6)` SI GB/s, where
  `B = buffer_size_mb * 1024 * 1024`.
- **Bidirectional** — aggregate payload rate `2 * B / (time_ms * 1e6)` for the
  submitted pair. The event bracket relies on legacy default-stream ordering
  with two blocking copy streams; qualify completion ordering on the runtime
  and build used for collection. Submission on two streams does not prove
  overlap, especially with pageable memory.

These rates do not measure every bus transaction or staging copy. See [transfer
interpretation](../../../docs/measurement_interpretation.md#transfer-payload-host-allocation-and-overlap)
for the CUDA synchronization references and qualification requirements.

## Build

The directory is **self-contained** — no CMake or external dependencies.

```bash
cd tier1/pcie_bandwidth/cuda
make                     # -arch=native (builds for the GPU in this machine)
make CUDA_ARCH=sm_80     # A100
make CUDA_ARCH=sm_90     # H100
```

```bash
# NVIDIA CUDA
make                     # -arch=native (builds for the GPU in this machine)
make CUDA_ARCH=sm_80     # A100
make CUDA_ARCH=sm_90     # H100
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
before each timed transfer, intended to raise clocks and disturb cache state.
Qualify the resulting clock/cache state on the target GPU. Repetition is
done by re-running the process (`gpubench run --reps N`).

## Example Output

Progress and device info go to stderr; stdout carries JSON lines — one record
per selected test, plus a final summary (illustrative timings for `transfer_type=all`):

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
| `BENCH_PARAM_vector_add_warmup` | true | Run the untimed warmer before each selected transfer; qualify the resulting clock/cache state on the target |

## Notes

- Bandwidth in SI GB/s (1 GB = 10⁹ bytes) is derived downstream from the
  recorded times.
- Pinned buffers use `hipHostMalloc()` (mapped to CUDA host allocation in the
  CUDA build); pageable buffers use `malloc()`.
- Pageable async copies may stage or serialize. Pinned allocation alone does
  not establish full-duplex overlap or a peak PCIe bandwidth claim.
- Report the actual link/topology and qualify the event timer on each target.