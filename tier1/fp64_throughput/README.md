# fp64_throughput

Microbenchmark that measures the peak **double-precision (FP64) floating-point
compute throughput** of a GPU in GFLOPS (10⁹ floating-point operations per
second).

Supported platforms: **NVIDIA CUDA** and **AMD ROCm/HIP**.

---

## What it measures

The benchmark saturates an SM/CU's FP64 arithmetic pipelines by executing a
dense loop of **fused multiply-add (FMA)** instructions:

```c
a = fma(a, multiplier, addend);   // 2 FLOPs: 1 multiply + 1 add
```

All operands live in registers — there is no memory traffic beyond a single
write at the end — so the result reflects raw arithmetic throughput rather than
memory bandwidth.

Four independent accumulators are used per thread so the compiler can
interleave FMA instructions and hide the FP64 pipeline latency, achieving
full issue-rate throughput.

---

## FP64 vs FP32 throughput (CUDA/HIP platforms)

FP64 throughput varies significantly by GPU class:

| GPU class | FP64 : FP32 ratio |
|---|---|
| Consumer (GeForce RTX) | 1:32 or 1:64 |
| Professional (Quadro / RTX A-series) | 1:16 or 1:32 |
| Datacenter / HPC (V100, A100, H100, MI250X) | ~1:2 |

---

## FLOP counting

| Quantity | Value |
|---|---|
| FLOPs per FMA | **2** (1 multiply + 1 add) |
| GFLOPS formula | `(threads × fmas_per_thread × 2) / elapsed_s / 1e9` |

The binary does not print GFLOPS; it is derived downstream from the
recorded `time_ms` and params using the formula above.

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
| `fmas_per_thread` | `4096` | Number of FMA iterations per thread. Increase on HPC GPUs where FP64 is fast (kernel should run ≥5 ms for a stable measurement). Decrease on consumer GPUs where FP64 is rate-limited. |

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
{"type":"kernel","name":"fp64_fma_kernel","iter":0,"time_ms":11.987120,"host_time_ms":12.104532,"params":{"fmas_per_thread":4096,"num_blocks":1,"threads_per_block":256}}
{"type":"summary","total_time_ms":11.987120}
```

`time_ms` is the GPU event time for the single kernel launch, `host_time_ms`
the host wall-clock around launch + sync, and `total_time_ms` equals the
single launch's GPU event time. No GFLOPS figure is printed — it is derived
downstream.

> **Note:** On consumer GPUs, FP64 throughput is severely limited by hardware.
> Consider reducing `fmas_per_thread` (e.g. to 256 or 512) to keep the
> benchmark runtime reasonable.

---

## Files

| File | Purpose |
|---|---|
| `fp64_throughput.hip` | HIP source — AMD ROCm + NVIDIA HIP-on-CUDA |
| `Makefile` | Build rules for all platforms |
| `README.md` | This file |
