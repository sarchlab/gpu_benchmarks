# fp32_throughput — Tier 1 Microbenchmark

## Overview

`fp32_throughput` times an FP32 FMA workload at the selected grid geometry.
The declared grid has one block. The binary emits raw time and work descriptors;
downstream analysis uses the nominal count below only after qualifying that the
compiled kernel preserves the counted work:

```
GFLOPS = (num_blocks * threads_per_block * fmas_per_thread * 2) / elapsed_seconds / 1e9
```

See [measurement interpretation](../../../docs/measurement_interpretation.md) for single-block versus device-wide claims and timing limits.

## Platform Support

| Platform | Source file | Compiler |
|----------|-------------|---------|
| NVIDIA CUDA | `fp32_throughput.hip` | `nvcc` |
| AMD ROCm/HIP | `fp32_throughput.hip` | `hipcc` |

## What It Measures

The selected FP32 FMA workload uses four independent accumulators to expose
instruction-level parallelism. Occupancy, issue limits, and clock state determine
the achieved rate; the instruction pattern alone does not prove peak throughput.

## Build

The directory is **self-contained** — no CMake or external dependencies.

### Auto-detect platform

```bash
cd tier1/fp32_throughput/hip
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
# Default 4096 FMAs per thread
./fp32_throughput

# Custom FMA count per thread (increase for faster GPUs)
./fp32_throughput 16384
```

Each execution launches and times the kernel **exactly once** — there is no
in-program warmup or iteration loop. Unless `BENCH_PARAM_vector_add_warmup`
is set to `false` (default `true`), a vector-add warmer runs ~100 ms of
sustained load immediately before the timed launch, intended to raise clocks
and disturb cache state. Qualify the resulting state on the target GPU. Repeat
measurements come from fresh process re-executions: `gpubench run --reps N`.

## Example Output

Progress and device info go to stderr; stdout carries JSON lines — one record
for the single timed launch plus a final summary. Illustrative timings:

```
{"type":"kernel","name":"fp32_fma_kernel","iter":0,"time_ms":0.524128,"host_time_ms":0.601342,"params":{"fmas_per_thread":4096,"num_blocks":1,"threads_per_block":256}}
{"type":"summary","total_time_ms":0.524128}
```

- `time_ms` is the GPU event time for the single kernel launch;
  `host_time_ms` is the host bracket around event recording, launch submission,
  error checking, and synchronization.
- `total_time_ms` equals the single launch's GPU event time.
- GFLOPS is not printed; it is derived downstream from `time_ms` and params.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BENCH_PARAM_fmas_per_thread` (or argv[1]) | 4096 | FMA iterations per thread (rounded down to a multiple of 4, minimum 4) |
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

- GFLOPS (10⁹ floating-point operations per second) is derived downstream
  from the recorded times; the binary itself only emits raw times.
- The vector-add warmer (on by default) runs before the single timed launch
  to raise clocks and disturb cache state. Qualify the resulting state on the
  target; there are no untimed launches of the benchmark kernel itself.
- **CUDA/HIP**: CUDA/HIP events are used for high-resolution GPU-side timing.
- The kernel loop uses four independent accumulators to expose parallelism;
  saturation remains a measurement question.
- `fmas_per_thread` is rounded down to the nearest multiple of 4 at runtime to
  match the unrolled loop structure, with a minimum of four for positive requests.

## Per-thread output validation

Every launched thread stores its four-accumulator checksum to a distinct output
slot after the FMA loop. Allocation stays outside timing; the final stores are
part of the measured kernel and can matter at small work counts. This makes
all threads' source work observable without adding a reduction or atomic.
Generated instructions and active-lane counts still require target qualification.

On a GPU host, run `make test PLATFORM=cuda` (or `PLATFORM=rocm`). The test
launches the production kernel with sentinel-filled output, checks every thread
against a finite CPU FMA reference, and checks both surrounding sentinels. It
covers the smallest/largest declared FMA counts, every declared block size,
and one/two blocks. It runs separately from the measurement executable and
adds no verification work or extra launch to benchmark measurements.
