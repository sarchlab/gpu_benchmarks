# shoc_stencil2d — SHOC 9-Point 2D Stencil Benchmark

## Description

Applies a 9-point star stencil iteratively on a 2D floating-point grid.
Emits one raw kernel time (accumulated over all stencil steps); effective
memory bandwidth (GB/s) is derived downstream.
Derived from the SHOC (Scalable Heterogeneous Computing) benchmark suite.

## Provenance & fidelity

Derived from the SHOC benchmark suite (https://github.com/vetter/shoc) — the `Stencil2D` workload. This is an independent reimplementation for portable measurement, **not** upstream source, and **not** a faithful reproduction of the upstream kernel. It differs from SHOC's version as follows:

- Untiled stencil vs SHOC's shared-memory tiled kernel.

Read this as "a 9-point 2D-stencil benchmark in the spirit of SHOC," not as SHOC's `Stencil2D`.

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
| `cpu_reference.h`         | Complete float CPU reference              |
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
{"type":"kernel","name":"stencil2d_kernel","iter":0,"time_ms":12.345,"host_time_ms":13.100,"params":{"n":2048,"iterations":5,"block_size":16,"verification":"full_cpu_v1"}}
{"type":"summary","total_time_ms":12.345}
```

## Performance Metric (derived downstream)

Effective memory bandwidth (GB/s), computed downstream from `time_ms` and
params — not reported by the binary. It approximates each stencil step as
reading and writing the full N×N grid:

```
GB/s = ITERATIONS × 2 × N × N × sizeof(float) / (time_s × 1e9)
```

## Correctness verification

Ordinary runs check every cell of the actual final measured state after timing.
The CPU reference applies the existing float formula for all requested
`iterations` from the original seed-123 input, with fixed zero boundaries. It
preserves the single-step arithmetic and per-cell grouping of the former
64×64 qualification reference. Interior acceptance remains
`fabsf(gpu-ref)/(fabsf(ref)+1e-6f) <= 1e-4f`; the GPU value, CPU reference and
relative error must all be finite. Boundaries must match exactly.

The returned GPU buffer follows timestep parity. The CPU reference consumes the
original host input before that allocation is reused for one full result
download. No separate GPU verification launch runs. All mismatches contribute
to the failure count, while at most ten individual diagnostics are printed.
Failure returns exit code 1, even though timing records have been emitted;
the CLI marks the run as failed. The kernel record identifies `full_cpu_v1`.
There is no verification on/off parameter.

At N=8192, the two float reference grids need 512 MiB at peak, in addition to
the existing 256 MiB host input. One reference scratch is released before the
256 MiB device-to-host result copy. CPU work and result copying are outside GPU
timing but count toward the process timeout. Qualify full process runtime and
host/device capacity on each collection host.

From the repository root:

```sh
c++ -O3 -std=c++14 -fsanitize=address,undefined tests/shoc_stencil_reference_test.cpp -o /tmp/shoc-stencil-reference
/tmp/shoc-stencil-reference
python3 tests/shoc_stencil_measured_result.py --platform cuda --work-dir /tmp/shoc-stencil-measured-result
python3 tests/finite_verification.py --platform cuda --benchmark shoc_stencil2d --work-dir /tmp/shoc-stencil-finite
```

Each output directory must be new. The host test compares 240 complete float
states bit for bit against an independently indexed recurrence, covering
positive/signed/zero/impulse inputs, partial blocks, one/two/odd/even/50/51-step
sequences, input preservation and exact zero boundaries. Nine exact one-step
impulses isolate the coefficients. The measured GPU regression checks 210
ordinary-path cases (16 controls, 194 faults), including first/middle/last
interior cells, boundaries, finite and NaN/±Inf corruption, one-ULP boundary
changes and capped diagnostics with complete mismatch counts. It modifies only
disposable copies and retains process logs/reports. The existing finite-value
regression now exercises the full measured-state comparator, including bad CPU
references. CI runs the host test and compiles the GPU regressions; executing
them requires a GPU. Use `--platform rocm` on AMD or `--compile-only` without
hardware.

AMD execution, other-host budgets, manual input admission and trace
qualification remain separate. Full result verification does not by itself
establish upstream implementation identity or publication performance.
