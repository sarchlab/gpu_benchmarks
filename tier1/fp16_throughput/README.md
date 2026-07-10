# fp16_throughput

Microbenchmark that measures the peak **half-precision (FP16) floating-point
compute throughput** of a GPU in GFLOPS (10⁹ floating-point operations per
second).

Supports **NVIDIA (CUDA)**, **AMD ROCm/HIP**, and **Apple Metal** targets.

---

## What it measures

The benchmark saturates the GPU's FP16 arithmetic pipelines by executing a
dense loop of **packed fused multiply-add (FMA)** instructions.  All operands
live in registers — there is no memory traffic beyond a single small write at
the end — so the result reflects raw arithmetic throughput rather than memory
bandwidth.

| Platform | Type | Intrinsic / Built-in |
|---|---|---|
| CUDA | `__half2` | `__hfma2(a, b, c)` |
| HIP (ROCm) | `__half2` | `__hfma2(a, b, c)` |
| Metal | `half2` | `fma(a, b, c)` |

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

The binary does not print GFLOPS; it is derived downstream from each
recorded iteration's `time_ms` and params using the formula above.

---

## Build

The Makefile auto-detects the platform:
- **Darwin** → `metal`
- **Linux with `/opt/rocm`** → `rocm`
- **Linux otherwise** → `cuda`

```bash
# Auto-detect
make

# Explicit platform
make PLATFORM=cuda
make PLATFORM=rocm
make PLATFORM=metal

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
`BENCH_PARAM_threads_per_block`), plus the measurement-harness controls:

| Env var | Default | Description |
|---|---|---|
| `BENCH_PARAM_warmup` | `3` | Warm-up kernel launches (not timed) |
| `BENCH_PARAM_iteration` | `7` | Timed kernel launches, each recorded individually |

---

## Example output

Progress and device info go to stderr; stdout carries JSON lines — one record
per timed iteration plus a final summary:

```
{"type":"kernel","name":"fp16_fma_kernel","iter":0,"time_ms":2.811214,"host_time_ms":2.902341,"params":{"fmas_per_thread":4096,"num_blocks":1,"threads_per_block":256}}
...
{"type":"summary","total_time_ms":19.678498}
```

Every timed iteration is emitted individually (no averaging); `time_ms` is
the GPU event time, `host_time_ms` the host wall-clock around launch + sync,
and `total_time_ms` is the sum of all iterations' GPU event times. No GFLOPS
figure is printed — it is derived downstream.

> **Note:** The Metal binary must be run from the directory containing
> `fp16_throughput.metal`, or the `.metal` file must reside next to the
> executable, so the host can compile the shader at runtime.

---

## Files

| File | Purpose |
|---|---|
| `fp16_throughput.hip` | HIP source — same kernel, AMD ROCm/HIP APIs |
| `fp16_throughput_metal.mm` | Metal host code (Objective-C++) |
| `fp16_throughput.metal` | Metal shader — `half2` FP16 FMA kernel |
| `Makefile` | Build rules for all three platforms |
| `README.md` | This file |
