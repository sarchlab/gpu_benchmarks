# fp32_throughput

Microbenchmark that measures the peak **single-precision (FP32) floating-point
compute throughput** of an NVIDIA GPU in GFLOPS (10⁹ floating-point operations
per second).

---

## What it measures

The benchmark saturates the GPU's FP32 arithmetic pipelines by executing a
dense loop of **fused multiply-add (FMA)** instructions:

```c
a = fmaf(a, multiplier, addend);   // 2 FLOPs: 1 multiply + 1 add
```

All operands live in registers — there is no memory traffic beyond a single
4-byte write at the end — so the result reflects raw arithmetic throughput
rather than memory bandwidth.

Four independent accumulators are used per thread so the compiler can
interleave FMA instructions and hide the 4-cycle FP32 pipeline latency,
achieving full issue-rate throughput.

---

## FLOP counting

| Quantity | Value |
|---|---|
| FLOPs per FMA | **2** (1 multiply + 1 add) |
| GFLOPS formula | `(threads × fmas_per_thread × 2) / elapsed_s / 1e9` |

---

## Build

```bash
# From the repository root:
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target fp32_throughput -j$(nproc)
```

To target specific GPU architectures:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="70;80;86;90"
```

---

## Usage

```
./fp32_throughput [fmas_per_thread]
```

| Argument | Default | Description |
|---|---|---|
| `fmas_per_thread` | `4096` | Number of FMA iterations per thread. Increase on very fast GPUs so the kernel runs for at least ~5 ms. |

---

## Example output

```
Device          : NVIDIA A100-SXM4-80GB (id=0)
SMs             : 108
Grid            : 1024 blocks × 256 threads = 262144 total threads
FMAs / thread   : 4096
Kernel repeats  : 10

Elapsed time    : 56.214 ms (5.621 ms / kernel)
Total FLOPs     : 21474836480
FP32 Throughput : 381.97 GFLOPS
```

> **Note:** The benchmark exercises a single SM wave (1024 blocks). True peak
> throughput is achieved when the grid is large enough to keep all SMs busy
> simultaneously.  Adjust `NUM_BLOCKS` in the source if needed for your GPU.

---

## Files

| File | Purpose |
|---|---|
| `fp32_throughput.cu` | CUDA source — kernel + driver code |
| `CMakeLists.txt` | CMake build rules |
| `README.md` | This file |
