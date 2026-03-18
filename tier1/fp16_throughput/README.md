# fp16_throughput

Microbenchmark that measures the peak **half-precision (FP16) floating-point
compute throughput** of an NVIDIA GPU in GFLOPS (10⁹ floating-point operations
per second).

Requires compute capability **7.0 (Volta) or later** for native FP16 CUDA-core
throughput.

---

## What it measures

The benchmark saturates the GPU's FP16 arithmetic pipelines by executing a
dense loop of **packed fused multiply-add (FMA)** instructions using the
`__half2` type and `__hfma2()` intrinsic:

```c
a = __hfma2(a, multiplier, addend);   // 4 FLOPs: 2 packed FP16 FMAs
```

`__half2` packs two `half` values into a single 32-bit register lane, so one
`__hfma2()` call performs **two simultaneous FP16 FMAs**.  All operands live in
registers — there is no memory traffic beyond a single 4-byte write at the end —
so the result reflects raw arithmetic throughput rather than memory bandwidth.

Four independent `__half2` accumulators are used per thread so the compiler can
interleave FMA instructions and hide the FP16 pipeline latency, achieving full
issue-rate throughput.

---

## FLOP counting

| Quantity | Value |
|---|---|
| Packed elements per `__half2` | **2** |
| FLOPs per FMA (1 multiply + 1 add) | **2** |
| FLOPs per `__hfma2()` call | **4** |
| GFLOPS formula | `(threads × fmas_per_thread × 4) / elapsed_s / 1e9` |

---

## Build

```bash
# From the repository root:
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target fp16_throughput -j$(nproc)
```

To target specific GPU architectures (sm_70 minimum for native FP16):

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="70;80;86;90"
```

---

## Usage

```
./fp16_throughput [fmas_per_thread]
```

| Argument | Default | Description |
|---|---|---|
| `fmas_per_thread` | `4096` | Number of `__hfma2` iterations per thread. Increase on very fast GPUs so the kernel runs for at least ~5 ms. |

---

## Example output

```
Device          : NVIDIA A100-SXM4-80GB (id=0)
Compute cap.    : 8.0
SMs             : 108
Grid            : 1024 blocks × 256 threads = 262144 total threads
FMAs / thread   : 4096  (__hfma2, 4 FLOPs each)
Kernel repeats  : 10

Elapsed time    : 28.107 ms (2.811 ms / kernel)
Total FLOPs     : 42949672960
FP16 Throughput : 1527.74 GFLOPS
```

> **Note:** The benchmark exercises a single SM wave (1024 blocks). True peak
> throughput is achieved when the grid is large enough to keep all SMs busy
> simultaneously.  Adjust `NUM_BLOCKS` in the source if needed for your GPU.

---

## Files

| File | Purpose |
|---|---|
| `fp16_throughput.cu` | CUDA source — kernel + driver code |
| `CMakeLists.txt` | CMake build rules |
| `README.md` | This file |
