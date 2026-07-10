# fp64_throughput

Microbenchmark that measures the peak **double-precision (FP64) floating-point
compute throughput** of a GPU in GFLOPS (10⁹ floating-point operations per
second).

Supported platforms: **NVIDIA CUDA**, **AMD ROCm/HIP**, and **Apple Metal**
(FP32 proxy — see [Apple GPU limitation](#apple-gpu-limitation) below).

---

## What it measures

The benchmark saturates the GPU's FP64 arithmetic pipelines by executing a
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

## Apple GPU limitation

> **Apple GPUs do not have native FP64 (double-precision) hardware.**

The Metal Shading Language does not expose a `double` type on Apple Silicon.
When built with `PLATFORM=metal` this benchmark runs **FP32 FMA operations**
instead and clearly labels the output as an FP32 proxy measurement:

```
FP32 Throughput : 2845.67 GFLOPS  (FP64 proxy — Apple GPUs lack native FP64)
```

This result **cannot** be directly compared to FP64 GFLOPS from CUDA or HIP
builds.  For true FP64 throughput, use an NVIDIA or AMD GPU with
`PLATFORM=cuda` or `PLATFORM=rocm`.

---

## FP64 vs FP32 throughput (CUDA/HIP platforms)

FP64 throughput varies significantly by GPU class:

| GPU class | FP64 : FP32 ratio |
|---|---|
| Consumer (GeForce RTX) | 1:32 or 1:64 |
| Professional (Quadro / RTX A-series) | 1:16 or 1:32 |
| Datacenter / HPC (V100, A100, H100, MI250X) | ~1:2 |
| Apple Silicon (M-series) | N/A — no native FP64 |

---

## FLOP counting

| Quantity | Value |
|---|---|
| FLOPs per FMA | **2** (1 multiply + 1 add) |
| GFLOPS formula | `(threads × fmas_per_thread × 2) / elapsed_s / 1e9` |

The binary does not print GFLOPS; it is derived downstream from each
recorded iteration's `time_ms` and params using the formula above.

---

## Build

```bash
# Auto-detect platform (Darwin → metal, /opt/rocm present → rocm, else cuda)
make

# Explicit platform selection
make PLATFORM=cuda
make PLATFORM=rocm
make PLATFORM=metal

make clean
```

Requirements by platform:

| Platform | Requirement |
|---|---|
| `cuda` | NVIDIA CUDA Toolkit (`nvcc`) |
| `rocm` | AMD ROCm/HIP (`hipcc` at `/opt/rocm/bin/hipcc`) |
| `metal` | macOS with Xcode Command Line Tools (`clang++`) |

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
{"type":"kernel","name":"fp64_fma_kernel","iter":0,"time_ms":11.987120,"host_time_ms":12.104532,"params":{"fmas_per_thread":4096,"num_blocks":1,"threads_per_block":256}}
...
{"type":"summary","total_time_ms":83.910041}
```

Every timed iteration is emitted individually (no averaging); `time_ms` is
the GPU event time, `host_time_ms` the host wall-clock around launch + sync,
and `total_time_ms` is the sum of all iterations' GPU event times. No GFLOPS
figure is printed — it is derived downstream.

> **Note:** On consumer GPUs, FP64 throughput is severely limited by hardware.
> Consider reducing `fmas_per_thread` (e.g. to 256 or 512) to keep the
> benchmark runtime reasonable.

---

## Files

| File | Purpose |
|---|---|
| `fp64_throughput.hip` | HIP source — AMD ROCm + NVIDIA HIP-on-CUDA |
| `fp64_throughput_metal.mm` | Apple Metal host code (Objective-C++) |
| `fp64_throughput.metal` | Metal compute kernel (FP32 FMA — Apple FP64 proxy) |
| `Makefile` | Build rules for all three platforms |
| `README.md` | This file |
