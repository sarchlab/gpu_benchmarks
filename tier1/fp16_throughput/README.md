# fp16_throughput

Microbenchmark that measures the peak **half-precision (FP16) floating-point
compute throughput** of a GPU in GFLOPS (10⁹ floating-point operations per
second).

Supports **NVIDIA (CUDA)** and **AMD ROCm/HIP** targets.

---

## What it measures

The benchmark saturates an SM/CU's FP16 arithmetic pipelines by executing a
dense loop of **packed fused multiply-add (FMA)** instructions.  All operands
live in registers — there is no memory traffic beyond one small per-thread
write at the end — so the result reflects raw arithmetic throughput rather than
memory bandwidth.

| Platform | Type | Intrinsic / Built-in |
|---|---|---|
| CUDA | `__half2` | `__hfma2(a, b, c)` |
| HIP (ROCm) | `__half2` | `__hfma2(a, b, c)` |

Each packed FP16 FMA operates on two `half` values simultaneously:

```
FLOPs per packed FMA call = 2 (packed elements) × 2 (mul + add) = 4
```

Four independent accumulators are used per thread to hide the FP16 pipeline
latency and achieve full issue-rate throughput.

---

## FLOP counting

| Quantity | Value |
|---|---|
| Packed elements per call | **2** |
| FLOPs per FMA (1 multiply + 1 add) | **2** |
| FLOPs per packed FMA call | **4** |
| GFLOPS formula | `(threads × fmas_per_thread × 4) / elapsed_s / 1e9` |

The binary does not print GFLOPS; it is derived downstream from the
recorded `time_ms` and params using the formula above.

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
| `fmas_per_thread` | `4096` | Number of packed FMA iterations per thread. Increase on very fast GPUs so the kernel runs for at least ~5 ms. |

Parameters can also be set via environment variables
(`BENCH_PARAM_fmas_per_thread`, `BENCH_PARAM_num_blocks`,
`BENCH_PARAM_threads_per_block`), plus one measurement control:

| Env var | Default | Description |
|---|---|---|
| `BENCH_PARAM_vector_add_warmup` | `true` | Run the vector-add warmer (~100 ms of sustained load boosting the clock and evicting the last-level cache) before the timed launch; set `false` to time the kernel as-is |

Each execution launches and times the kernel **exactly once** — there is no
in-program warmup or iteration loop. Repeat measurements come from fresh
process re-executions: `gpubench run --reps N`.

> **Single-SM/CU measurement.** The default grid is one block
> (`num_blocks = 1`), so the benchmark measures a single SM/CU at boost
> clock. Because FMA pipes are per-SM resources with no cross-SM
> interaction, whole-device peak GFLOPS can be estimated as the per-SM
> value × the GPU's SM/CU count. Treat that as an upper bound: an all-SM
> workload is power/thermal-limited and typically sustains a lower clock
> than a single busy SM. For a device-wide run, set
> `BENCH_PARAM_num_blocks` manually (e.g. ≥ 8 × SM count).

---

## Example output

Progress and device info go to stderr; stdout carries JSON lines — one record
for the single timed launch plus a final summary:

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
