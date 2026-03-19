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

### CUDA / HIP

```
Device          : NVIDIA A100-SXM4-80GB (id=0)
Array size      : 256 MiB (67108864 elements, 268435456 bytes)
Warmup iters    : 5
Timed iters     : 5
Ratios          : 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0

FLOP/byte          Time (ms)      GFLOPS
--------           --------      --------
0.5                   0.183       146.50
1.0                   0.185       290.10
2.0                   0.210       512.30
4.0                   0.380       565.20
8.0                   0.740       580.40
16.0                  1.460       587.10
32.0                  2.900       590.80
64.0                  5.780       592.60
mixbench,0.5,0.183,146.50
mixbench,1.0,0.185,290.10
mixbench,2.0,0.210,512.30
mixbench,4.0,0.380,565.20
mixbench,8.0,0.740,580.40
mixbench,16.0,1.460,587.10
mixbench,32.0,2.900,590.80
mixbench,64.0,5.780,592.60

Done.
```

### Metal (Apple M3)

```
Device          : Apple M3 Max
Array size      : 256 MiB (67108864 elements, 268435456 bytes)
Warmup iters    : 5
Timed iters     : 5
Ratios          : 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0

FLOP/byte          Time (ms)      GFLOPS
--------           --------      --------
0.5                   0.350        76.50
1.0                   0.360       148.70
2.0                   0.420       255.20
4.0                   0.710       301.30
8.0                   1.350       317.20
16.0                  2.680       319.50
32.0                  5.340       320.70
64.0                 10.650       321.40
mixbench,0.5,0.350,76.50
mixbench,1.0,0.360,148.70
mixbench,2.0,0.420,255.20
mixbench,4.0,0.710,301.30
mixbench,8.0,1.350,317.20
mixbench,16.0,2.680,319.50
mixbench,32.0,5.340,320.70
mixbench,64.0,10.650,321.40

Done.
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `buffer_size_mb` (argv[1]) | 256 | Array size in MiB |
| `WARMUP_ITERATIONS` (compile-time) | 5 | Warmup runs per ratio |
| `TIMED_ITERATIONS` (compile-time) | 5 | Timed runs averaged per ratio |

## CSV Output Format

One CSV line per ratio (to stdout, interspersed with human-readable output):

```
mixbench,<flops_per_byte>,<time_ms>,<achieved_gflops>
```

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
