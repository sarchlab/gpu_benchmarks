# mixbench — Tier 1 Microbenchmark

## Overview

`mixbench` is a parametric roofline benchmark that sweeps the compute-to-memory
ratio across multiple data points, tracing out the GPU's roofline curve.
Inspired by [github.com/ekondis/mixbench](https://github.com/ekondis/mixbench).

Each GPU thread loads one float (4 bytes) from global memory, performs a
variable-length chain of FP32 FMA operations, and stores the result back
(4 bytes).  By varying the number of FMA operations, the benchmark covers
operational intensities from memory-bound to compute-bound regimes.

| Ratio (FLOP/byte) | FMAs/thread | Regime |
|-------------------|-------------|--------|
| 0.5               | 2           | Memory-bound |
| 1                 | 4           | Memory-bound |
| 2                 | 8           | Transition |
| 4                 | 16          | Transition |
| 8                 | 32          | Compute-bound |
| 16                | 64          | Compute-bound |
| 32                | 128         | Compute-bound |
| 64                | 256         | Compute-bound |

## Platform Support

| Platform | Source file | Compiler |
|----------|-------------|---------|
| NVIDIA CUDA | `mixbench.hip` (with `-DUSE_CUDA`) | `nvcc` |
| AMD ROCm/HIP | `mixbench.hip` (with `-DUSE_ROCM`) | `hipcc` |
| Apple Metal | `mixbench_metal.mm` + `mixbench.metal` | `clang++` |

## What It Measures

- **Low ratios (0.5–2 FLOP/byte)** — performance is bounded by memory
  bandwidth.  The achieved GFLOPS will plateau at
  `peak_bandwidth × operational_intensity`.
- **High ratios (8–64 FLOP/byte)** — performance approaches peak compute
  throughput.  The achieved GFLOPS will plateau near the GPU's theoretical
  FP32 peak.
- **Knee of the roofline** — the transition point between memory-bound and
  compute-bound regimes reveals the balance point of the architecture.

## Build

The directory is **self-contained** — no CMake or external dependencies.

### Auto-detect platform

```bash
cd tier1/mixbench
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
# Default 256 MiB array
./mixbench

# Custom array size (in MiB)
./mixbench 512
```

## Example Output

Each invocation runs a single `flops_per_byte` ratio. Progress and device
info go to stderr; stdout carries JSON lines — one record per timed
iteration plus a final summary:

```
{"type":"kernel","name":"mixbench_kernel","iter":0,"time_ms":0.380124,"host_time_ms":0.462201,"params":{"buffer_size_mb":256,"flops_per_byte":4.0,"threads_per_block":256}}
...
{"type":"summary","total_time_ms":2.660831}
```

- Every timed iteration is emitted individually — no averaging.
- `time_ms` is the GPU event time for that single kernel launch;
  `host_time_ms` is the host wall-clock around launch + sync.
- `total_time_ms` is the sum of all iterations' GPU event times.
- GFLOPS is not printed; it is derived downstream from `time_ms` and the
  params in each record. The roofline curve is traced by sweeping
  `flops_per_byte` across invocations.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BENCH_PARAM_buffer_size_mb` (or argv[1]) | 256 | Array size in MiB |
| `BENCH_PARAM_flops_per_byte` | 4.0 | Target FLOP/byte ratio for this invocation |
| `BENCH_PARAM_threads_per_block` | 256 | Threads per block |
| `BENCH_PARAM_warmup` | 3 | Warmup runs (not timed) |
| `BENCH_PARAM_iteration` | 7 | Timed runs, each recorded individually |

## Operational Intensity Calculation

For each thread:
- Memory accessed: 4 bytes (read) + 4 bytes (write) = **8 bytes**
- FLOPs: `num_fmas × 2` (each FMA = 1 multiply + 1 add)
- `num_fmas = flops_per_byte × 4`
- Operational intensity: `(num_fmas × 2) / 8 = flops_per_byte`

## Notes

- The FMA dependency chain prevents the compiler from eliminating work.
- A 256 MiB array provides 64M+ threads — enough to saturate any modern GPU.
- Results at low ratios reflect memory bandwidth; results at high ratios
  reflect peak FP32 compute throughput.
- **CUDA/HIP**: GPU events used for high-resolution timing.
- **Metal**: `mach_absolute_time()` wall-clock timing; command buffers are
  serialized (`waitUntilCompleted`) for accurate measurement.
- The single `.hip` file uses `#ifdef USE_CUDA` / `USE_ROCM` macros to
  abstract over CUDA and HIP APIs, avoiding code duplication.
