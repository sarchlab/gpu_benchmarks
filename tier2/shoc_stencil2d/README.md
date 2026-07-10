# shoc_stencil2d — SHOC 9-Point 2D Stencil Benchmark

## Description

Applies a 9-point star stencil iteratively on a 2D floating-point grid.
Emits raw per-pass kernel times; effective memory bandwidth (GB/s) is
derived downstream.
Derived from the SHOC (Scalable Heterogeneous Computing) benchmark suite.

## Algorithm

The stencil reads 9 neighbors for each interior cell:

```
out[i][j] = 0.5   * in[i][j]
          + 0.1   * (in[i-1][j] + in[i+1][j] + in[i][j-1] + in[i][j+1])
          + 0.025 * (in[i-1][j-1] + in[i-1][j+1] + in[i+1][j-1] + in[i+1][j+1])
```

- **Grid**: N×N floats (default N=2048)
- **Interior only**: rows/columns 1 .. N-2; boundary cells remain 0
- **Thread blocks**: 16×16
- **Ping-pong buffers**: `in_buf` and `out_buf` swapped each iteration
- **Passes**: `BENCH_PARAM_warmup` warmup passes (default 3), then `BENCH_PARAM_iteration` timed passes (default 7), each recorded individually

## Files

| File                        | Description                                |
|-----------------------------|--------------------------------------------|
| `shoc_stencil2d.hip`        | HIP/CUDA kernel + host (NVIDIA & AMD)      |
| `shoc_stencil2d.metal`      | Metal compute shader (Apple)               |
| `shoc_stencil2d_metal.mm`   | Metal Objective-C++ host (Apple)           |
| `Makefile`                  | Multi-platform build (cuda/rocm/metal)     |
| `README.md`                 | This file                                  |

## Build

```bash
# Auto-detect platform
make

# Explicit platform
make PLATFORM=cuda    # NVIDIA CUDA
make PLATFORM=rocm    # AMD ROCm/HIP
make PLATFORM=metal   # Apple Metal

make clean
```

## Run

```bash
# Default: 2048×2048 grid, 5 stencil steps per pass
./shoc_stencil2d

# Custom parameters
BENCH_PARAM_iteration=5 ./shoc_stencil2d --n 4096 --iterations 10
```

### Options

| Option           | Default | Description                         |
|------------------|---------|-------------------------------------|
| `--n N`          | 2048    | Grid dimension (N×N floats)         |
| `--iterations I` | 5       | Stencil steps per timed pass        |
| `--trials T`     | 3       | Legacy trial-count param (echoed in records) |

Environment variables (`BENCH_PARAM_*`) override the flags. In addition,
`BENCH_PARAM_warmup` (default 3) sets the warmup passes and
`BENCH_PARAM_iteration` (default 7) the timed passes.

## Output Format

**stderr**: device info, per-pass timing, correctness check result

**stdout** (JSON lines): one `kernel` record per timed pass with the
accumulated GPU event time (`time_ms`) and host time (`host_time_ms`) of all
stencil steps in that pass, then a summary whose `total_time_ms` is the sum
of all recorded GPU event times:
```json
{"type":"kernel","name":"stencil2d_kernel","iter":0,"time_ms":12.345,"host_time_ms":13.100,"params":{"n":2048,"iterations":5,"trials":3,"block_size":16}}
{"type":"summary","total_time_ms":86.42}
```

## Performance Metric (derived downstream)

Effective memory bandwidth (GB/s), computed downstream from `time_ms` and
params — not reported by the binary. It approximates each stencil step as
reading and writing the full N×N grid:

```
GB/s = ITERATIONS × 2 × N × N × sizeof(float) / (time_s × 1e9)
```

## Correctness Verification

Before the timed run, a single stencil iteration is applied to a small
64×64 grid on both CPU and GPU. The GPU result is compared element-wise
to the CPU reference; the check passes when the maximum relative error
across all interior cells is less than 1×10⁻⁴.
