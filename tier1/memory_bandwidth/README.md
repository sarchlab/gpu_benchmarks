# memory_bandwidth — Tier 1 Microbenchmark

## Overview

`memory_bandwidth` times device-to-device (D2D) copies in its declared sweep.
`params.json` fixes `transfer_type=d2d` and `use_pinned_memory=true` across
13 buffer sizes, from 1 to 4096 MiB. Each process emits one `memcpy_d2d`
timing and a summary. The host-to-device setup copy is outside timing.

Use [pcie_bandwidth](../pcie_bandwidth/README.md) for the declared host-transfer
sweep: H2D, D2H and bidirectional modes at 22 buffer sizes, with both pinned
and pageable host allocations. These are separate benchmark identities.

For direct-binary compatibility, `memory_bandwidth` also accepts `h2d`, `d2h`
and `all`, plus pageable host memory. Those values are not in this manifest's
sweep. Its three `kernels` entries list supported record names; they do not
require every name to appear in a D2D run. Keep the reported transfer and
allocation parameters with any manually collected result.

## Platform Support

| Platform | Source file | Compiler |
|----------|-------------|---------|
| NVIDIA CUDA | `memory_bandwidth.hip` | `nvcc` |
| AMD ROCm/HIP | `memory_bandwidth.hip` | `hipcc` |

## What It Measures

The recorded interval covers one runtime copy operation. Derive a D2D payload
rate from the buffer bytes, or a logical read-plus-write rate from twice that
amount, and label the convention. Neither establishes peak device bandwidth or
physical HBM/GDDR traffic; cache behavior and the copy implementation need
qualification on the target.

The program allocates two host buffers and two device buffers even in D2D
mode. `use_pinned_memory` controls the host setup allocation, not the device
allocation or the timed D2D copy. Include these untimed allocations and the
optional warmer when checking whether a configuration fits memory.

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

# Direct-binary compatibility mode, outside the declared sweep
BENCH_PARAM_transfer_type=all ./memory_bandwidth

# Declared D2D collection (13 sizes, seven fresh processes per size)
gpubench run --tier 1 --benchmark memory_bandwidth --reps 7
```

## Measurement Protocol

Each selected direction is copied **exactly once** per process execution and
timed with GPU events; there is no in-program warmup or iteration loop. Unless
`BENCH_PARAM_vector_add_warmup=false`, a vector-add warmer runs immediately
before each timed copy, intended to raise clocks and disturb cache state.
Qualify the resulting state on the target. Repetition is done by
re-running the process (`gpubench run --reps N`).

## Example Output

Progress and device info go to stderr; stdout carries JSON lines — one record
per selected direction, plus a final summary. The declared D2D configuration
produces the following record shape (illustrative timings):

```
{"type":"kernel","name":"memcpy_d2d","iter":0,"time_ms":0.612340,"host_time_ms":0.701234,"params":{"buffer_size_mb":256,"transfer_type":"d2d","use_pinned_memory":"true"}}
{"type":"summary","total_time_ms":0.612340}
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
| `BENCH_PARAM_transfer_type` | d2d | Declared sweep: `d2d`; direct-binary modes also accept `h2d`, `d2h`, `all` |
| `BENCH_PARAM_use_pinned_memory` | true | Declared sweep: `true`; direct binary also accepts `false` for pageable host setup |
| `BENCH_PARAM_vector_add_warmup` | true | Run the untimed warmer before each selected copy; fixed per harness run, not swept |

## Notes

- Bandwidth in SI gigabytes per second (1 GB = 10⁹ bytes) is derived
  downstream from the recorded times.
- **Byte-accounting convention.** With `B = buffer_size_mb × 2²⁰` bytes,
  D2D payload rate is `B / time_ms / 1e6` SI GB/s; logical read-plus-write
  rate is `2 × B / time_ms / 1e6`. H2D/D2H payload rates use the first formula.
  These formulas model payload/logical bytes, not every physical transaction.
- Direct `all` mode runs H2D, D2H and D2D sequentially, with one record per
  direction and a sum in the summary. It is not a bidirectional overlap test.
  See [transfer interpretation](../../docs/measurement_interpretation.md#transfer-payload-host-allocation-and-overlap)
  for host allocation, synchronization and target qualification limits.
- **CUDA/HIP**: CUDA/HIP events are used for high-resolution GPU-side timing.
