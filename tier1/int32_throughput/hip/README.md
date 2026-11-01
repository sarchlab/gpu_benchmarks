# int32_throughput — Tier 1 Microbenchmark

## Overview

`int32_throughput` times an integer multiply-add workload at the selected grid
geometry. The declared grid has one block. Operands stay in registers and each
thread writes a result. The binary emits raw time and work descriptors; the
nominal source count is two integer operations per MAD:

```
GIOPS = (num_blocks * threads_per_block * mads_per_thread * 2) / elapsed_seconds / 1e9
```

Use this count after target instruction qualification. See [measurement interpretation](../../../docs/measurement_interpretation.md) for single-block versus device-wide claims.

## Platform Support

| Platform | Source file | Compiler |
|----------|-------------|---------|
| NVIDIA CUDA | `int32_throughput.hip` | `nvcc` |
| AMD ROCm/HIP | `int32_throughput.hip` | `hipcc` |

## What It Measures

The selected INT32 multiply-add workload uses four independent accumulators to expose
instruction-level parallelism. Occupancy, issue limits, and clock state determine
the achieved rate; the instruction pattern alone does not prove peak throughput.

## Build

The directory is **self-contained** — no CMake or external dependencies.

### Auto-detect platform

```bash
cd tier1/int32_throughput/hip
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
# Default 4096 MADs per thread
./int32_throughput

# Custom MAD count per thread (increase for faster GPUs)
./int32_throughput 16384
```

Each execution launches and times the kernel **exactly once** — there is no
in-program warmup or iteration loop. Unless `BENCH_PARAM_vector_add_warmup`
is set to `false` (default `true`), a vector-add warmer runs ~100 ms of
sustained load immediately before the timed launch, intended to raise clocks
and disturb cache state. Qualify the resulting state on the target GPU. Repeat
measurements come from fresh process re-executions: `gpubench run --reps N`.

## Example Output

Human-readable diagnostic output goes to **stderr**; **stdout** carries JSON
lines — one record for the single timed launch plus a final summary. Illustrative timings:

```
{"type":"kernel","name":"int32_mad_kernel","iter":0,"time_ms":0.524128,"host_time_ms":0.598342,"params":{"mads_per_thread":4096,"num_blocks":1,"threads_per_block":256}}
{"type":"summary","total_time_ms":0.524128}
```

- `time_ms` is the GPU event time for the single kernel launch;
  `host_time_ms` is the host bracket around event recording, launch submission,
  error checking, and synchronization.
- `total_time_ms` equals the single launch's GPU event time.
- GIOPS is not printed; it is derived downstream from `time_ms` and params.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BENCH_PARAM_mads_per_thread` (or argv[1]) | 4096 | MAD iterations per thread (rounded down to a multiple of 4, minimum 4) |
| `BENCH_PARAM_threads_per_block` | 256 | Threads per block / threadgroup |
| `BENCH_PARAM_num_blocks` | 1 | Number of blocks in the dispatch grid (default and declared sweep: one block) |
| `BENCH_PARAM_vector_add_warmup` | true | Run the vector-add warmer before the timed launch; qualify clock/cache effects on the target |

> **Single-block measurement.** The manifest fixes `num_blocks=1`. Report the
> selected block size and work count; one block does not establish SM/CU or
> whole-device saturation. Multiplying this rate by SM/CU count is an
> extrapolation model, without a demonstrated upper-bound guarantee. A
> device-wide claim needs its own declared grid configuration and measured
> result. See [measurement interpretation](../../../docs/measurement_interpretation.md) for counting and qualification limits.

## Notes

- GIOPS (10⁹ integer operations per second) is derived downstream from the
  recorded times; the binary itself only emits raw times.
- The vector-add warmer (on by default) runs before the single timed launch
  to raise clocks and disturb cache state. Qualify the resulting state on the
  target; there are no untimed launches of the benchmark kernel itself.
- **CUDA/HIP**: CUDA/HIP events are used for high-resolution GPU-side timing.
- The kernel loop uses four independent accumulators to expose parallelism;
  saturation remains a measurement question.
- `mads_per_thread` is rounded down to the nearest multiple of 4 at runtime to
  match the unrolled loop structure, with a minimum of four for positive requests.
- The accumulators are **unsigned**, so overflow wraps with well-defined
  semantics (signed overflow would be undefined behavior). The goal is to
  exercise the integer ALU, not to produce meaningful results.
- The multiplier and addend are passed as **runtime kernel arguments** rather
  than compile-time constants. With constant operands the compiler
  strength-reduces the multiply to shifts/adds and can algebraically collapse
  the `a = a*mul + add` recurrence into fewer instructions (e.g. folding four
  `a*3+1` steps into a single `a*81+40`), which would report more MADs than the
  hardware actually executes. Runtime operands force one genuine integer
  multiply-add per iteration.
