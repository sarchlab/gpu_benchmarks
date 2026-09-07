# fp16_throughput

Microbenchmark that times **packed FP16 FMA work** at the selected grid geometry.
The declared grid is one block; GFLOPS is derived downstream from qualified work
counts and the raw GPU time.

Supports **NVIDIA (CUDA)** and **AMD ROCm/HIP** targets.

---

## What it measures

The benchmark exercises FP16 arithmetic with a
dense loop of **packed fused multiply-add (FMA)** instructions. The source uses
register operands and writes one result per thread; compiled work and any
register spills still need target qualification.

| Platform | Type | Intrinsic / Built-in |
|---|---|---|
| CUDA | `__half2` | `__hfma2(a, b, c)` |
| HIP (ROCm) | `__half2` | `__hfma2(a, b, c)` |

Each packed FP16 FMA operates on two `half` values simultaneously:

```
FLOPs per packed FMA call = 2 (packed elements) × 2 (mul + add) = 4
```

Four independent accumulators expose instruction-level parallelism. Full issue
rate depends on the target and block occupancy and must be established by measurement.

---

## FLOP counting

| Quantity | Value |
|---|---|
| Packed elements per call | **2** |
| FLOPs per FMA (1 multiply + 1 add) | **2** |
| FLOPs per packed FMA call | **4** |
| GFLOPS formula | `(num_blocks × threads_per_block × fmas_per_thread × 4) / elapsed_s / 1e9` |

The binary does not print GFLOPS; it is derived downstream from the
recorded `time_ms` and effective params using the formula above after work-count
qualification. Positive FMA requests are rounded down to a multiple of four,
with a minimum of four; all declared values already satisfy this constraint.

---

## Build

The Makefile auto-detects the platform:
- **Linux with `/opt/rocm`** → `rocm`
- **Linux otherwise** → `cuda`

```bash
# Auto-detect
make

# Explicit platform
make PLATFORM=cuda
make PLATFORM=rocm

# Clean
make clean
```

---

## Usage

```
./fp16_throughput [fmas_per_thread]
```

| Argument | Default | Description |
|---|---|---|
| `fmas_per_thread` | `4096` | Packed FMA calls per thread; choose a declared work count with a resolvable event interval. |

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
{"type":"kernel","name":"fp16_fma_kernel","iter":0,"time_ms":2.811214,"host_time_ms":2.902341,"params":{"fmas_per_thread":4096,"num_blocks":1,"threads_per_block":256}}
{"type":"summary","total_time_ms":2.811214}
```

`time_ms` is the GPU event time for the single kernel launch, `host_time_ms`
the host wall-clock around launch + sync, and `total_time_ms` equals the
single launch's GPU event time. No GFLOPS figure is printed — it is derived
downstream.

---

## Files

| File | Purpose |
|---|---|
| `fp16_throughput.hip` | HIP source — same kernel, AMD ROCm/HIP APIs |
| `Makefile` | Build rules for all platforms |
| `README.md` | This file |
