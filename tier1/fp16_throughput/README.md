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

---

## Example output (CUDA / HIP)

```
Device          : NVIDIA A100-SXM4-80GB (id=0)
Grid            : 1024 blocks × 256 threads = 262144 total threads
FMAs / thread   : 4096  (__hfma2, 4 FLOPs each)
Kernel repeats  : 10

Elapsed time    : 28.107 ms (2.811 ms / kernel)
Total FLOPs     : 42949672960
FP16 Throughput : 1527.74 GFLOPS

Done.
```

## Example output (Metal)

```
Device          : Apple M3 Pro
Grid            : 1024 blocks × 256 threads = 262144 total threads
FMAs / thread   : 4096  (fma(half2,...), 4 FLOPs each)
Kernel repeats  : 10

Elapsed time    : 12.543 ms (1.254 ms / kernel)
Total FLOPs     : 42949672960
FP16 Throughput : 3424.89 GFLOPS

Done.
```

> **Note:** The Metal binary must be run from the directory containing
> `fp16_throughput.metal`, or the `.metal` file must reside next to the
> executable, so the host can compile the shader at runtime.

---

## Files

| File | Purpose |
|---|---|
| `fp16_throughput.cu` | CUDA source — kernel + driver code |
| `fp16_throughput.hip` | HIP source — same kernel, AMD ROCm/HIP APIs |
| `fp16_throughput_metal.mm` | Metal host code (Objective-C++) |
| `fp16_throughput.metal` | Metal shader — `half2` FP16 FMA kernel |
| `Makefile` | Build rules for all three platforms |
| `README.md` | This file |
