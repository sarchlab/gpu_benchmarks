# shoc_stencil2d — SHOC 9-Point 2D Stencil Benchmark

## Description

Applies a 9-point star stencil iteratively on a 2D floating-point grid.
Emits one raw kernel time (accumulated over all stencil steps); effective
memory bandwidth (GB/s) is derived downstream.
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
- **Ping-pong buffers**: `in_buf` and `out_buf` swapped each step
- **Steps**: `iterations` stencil time-steps (default 5), all inside the single measured run

## Measurement protocol

The measured run — all `iterations` ping-pong stencil steps over the pristine
input — executes **exactly once** per process; there is no in-program warmup
or repetition loop. Each kernel launch is bracketed with GPU events and the
times are accumulated into one record. When `BENCH_PARAM_vector_add_warmup` is
`true` (default), the vector-add warmer runs after the (untimed) buffer
initialization and immediately before the first timed launch: ~100 ms of
sustained load ramps the GPU clock while its final pass evicts the last-level
cache, so the run is cold-cache at a boosted clock. Repetition is the
harness's job: `gpubench run --reps N` re-executes the binary in fresh
processes.

## Files

| File                        | Description                                |
|-----------------------------|--------------------------------------------|
| `shoc_stencil2d.hip`        | HIP/CUDA kernel + host (NVIDIA & AMD)      |
| `Makefile`                  | Multi-platform build (cuda/rocm)           |
| `README.md`                 | This file                                  |

## Build

```bash
# Auto-detect platform
make

# Explicit platform
make PLATFORM=cuda    # NVIDIA CUDA
make PLATFORM=rocm    # AMD ROCm/HIP

make clean
```

## Run

```bash
# Default: 2048×2048 grid, 5 stencil steps
./shoc_stencil2d

# Custom parameters
./shoc_stencil2d --n 4096 --iterations 10
```

### Options

| Option           | Default | Description                         |
|------------------|---------|-------------------------------------|
| `--n N`          | 2048    | Grid dimension (N×N floats)         |
| `--iterations I` | 5       | Stencil time-steps in the measured run |

Environment variables override the flags:

| Env var | Default | Description |
|---------|---------|-------------|
| `BENCH_PARAM_n` | 2048 | Grid dimension |
| `BENCH_PARAM_iterations` | 5 | Stencil time-steps |
| `BENCH_PARAM_block_size` | 16 | Thread-block dimension (block_size × block_size) |
| `BENCH_PARAM_vector_add_warmup` | true | Clock-boost + LLC-evict warmer before the timed run |

## Output Format

**stderr**: device info, timing, correctness check result

**stdout** (JSON lines): one `kernel` record with the accumulated GPU event
time (`time_ms`) and host time (`host_time_ms`) of all stencil steps in the
run, then a summary whose `total_time_ms` equals that record's GPU event time:
```json
{"type":"kernel","name":"stencil2d_kernel","iter":0,"time_ms":12.345,"host_time_ms":13.100,"params":{"n":2048,"iterations":5,"block_size":16}}
{"type":"summary","total_time_ms":12.345}
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
across all interior cells is less than 1×10⁻⁴. On FAIL the process exits
non-zero (after the measurements have been emitted), so the harness marks
the run as failed without losing the timing data. There is no verification
on/off parameter.
