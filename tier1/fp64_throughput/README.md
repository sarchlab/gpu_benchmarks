# fp64_throughput

Microbenchmark that times **FP64 FMA work** at the selected grid geometry. The
declared grid is one block; GFLOPS is derived downstream after qualifying that
the compiled kernel preserves the counted work.

Supported platforms: **NVIDIA CUDA** and **AMD ROCm/HIP**.

---

## What it measures

The benchmark exercises FP64 arithmetic with a
dense loop of **fused multiply-add (FMA)** instructions:

```c
a = fma(a, multiplier, addend);   // 2 FLOPs: 1 multiply + 1 add
```

The source uses register operands with a final result store. Counted-work
preservation must be established in the compiled binary before deriving
arithmetic throughput (R-02 in the readiness tracker).

Four independent accumulators expose instruction-level parallelism. The achieved
rate depends on occupancy, instruction issue, and clock state; full issue rate
is a measurement result rather than a property guaranteed by the source pattern.

---

## FP64 versus FP32 comparisons

The supported arithmetic rates depend on the exact GPU model and instruction
path. Compare matched, qualified configurations; a broad product-class ratio
cannot substitute for the measured FP64 and FP32 results.

---

## FLOP counting

| Quantity | Value |
|---|---|
| FLOPs per FMA | **2** (1 multiply + 1 add) |
| GFLOPS formula | `(num_blocks × threads_per_block × fmas_per_thread × 2) / elapsed_s / 1e9` |

The binary does not print GFLOPS; it is derived downstream from the
recorded `time_ms` and effective params using the formula above after work-count
qualification. Positive FMA requests are rounded down to a multiple of four,
with a minimum of four; all declared values already satisfy this constraint.

---

## Build

```bash
# Auto-detect platform (/opt/rocm present → rocm, else cuda)
make

# Explicit platform selection
make PLATFORM=cuda
make PLATFORM=rocm

make clean
```

Requirements by platform:

| Platform | Requirement |
|---|---|
| `cuda` | NVIDIA CUDA Toolkit (`nvcc`) |
| `rocm` | AMD ROCm/HIP (`hipcc` at `/opt/rocm/bin/hipcc`) |

---

## Usage

```
./fp64_throughput [fmas_per_thread]
```

| Argument | Default | Description |
|---|---|---|
| `fmas_per_thread` | `4096` | FMA calls per thread; choose a declared work count within the target timing and runtime budgets. |

Parameters can also be set via environment variables
(`BENCH_PARAM_fmas_per_thread`, `BENCH_PARAM_num_blocks`,
`BENCH_PARAM_threads_per_block`), plus one measurement control:

| Env var | Default | Description |
|---|---|---|
| `BENCH_PARAM_vector_add_warmup` | `true` | Run the vector-add warmer (~100 ms of sustained load) before the timed launch; qualify clock/cache effects on the target |

Each execution launches and times the kernel **exactly once** — there is no
in-program warmup or iteration loop. Repeat measurements come from fresh
process re-executions: `gpubench run --reps N`.

> **Single-block measurement.** The manifest fixes `num_blocks=1`. Report the
> selected block size and work count; one block does not establish SM/CU or
> whole-device saturation. Multiplying this rate by SM/CU count is an
> extrapolation model, without a demonstrated upper-bound guarantee. A
> device-wide claim needs its own declared grid configuration and measured
> result. See [measurement interpretation](../../docs/measurement_interpretation.md) for counting and qualification limits.

---

## Example output

Progress and device info go to stderr; stdout carries JSON lines — one record
for the single timed launch plus a final summary. Illustrative timings:

```
{"type":"kernel","name":"fp64_fma_kernel","iter":0,"time_ms":11.987120,"host_time_ms":12.104532,"params":{"fmas_per_thread":4096,"num_blocks":1,"threads_per_block":256}}
{"type":"summary","total_time_ms":11.987120}
```

`time_ms` is the GPU event time for the single kernel launch, `host_time_ms`
the host bracket around event recording, launch submission, error checking,
and synchronization. `total_time_ms` equals the single launch's GPU event time.
No GFLOPS figure is printed — it is derived downstream.

---

## Files

| File | Purpose |
|---|---|
| `fp64_throughput.hip` | HIP source — AMD ROCm + NVIDIA HIP-on-CUDA |
| `Makefile` | Build rules for all platforms |
| `README.md` | This file |

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
