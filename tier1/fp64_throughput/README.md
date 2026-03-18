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

---

## Example output

### Datacenter GPU — NVIDIA A100-SXM4-80GB (CUDA/HIP)

```
Device          : NVIDIA A100-SXM4-80GB (id=0)
CUs             : 108
Grid            : 1024 blocks × 256 threads = 262144 total threads
FMAs / thread   : 4096
Kernel repeats  : 10

Elapsed time    : 119.872 ms (11.987 ms / kernel)
Total FLOPs     : 21474836480
FP64 Throughput : 179.16 GFLOPS
```

### Consumer GPU — NVIDIA GeForce RTX 3080 (CUDA/HIP)

```
Device          : NVIDIA GeForce RTX 3080 (id=0)
CUs             : 68
Grid            : 1024 blocks × 256 threads = 262144 total threads
FMAs / thread   : 4096
Kernel repeats  : 10

Elapsed time    : 3241.500 ms (324.150 ms / kernel)
Total FLOPs     : 21474836480
FP64 Throughput : 6.63 GFLOPS
```

> **Note:** On consumer GPUs, FP64 throughput is severely limited by hardware.
> Consider reducing `fmas_per_thread` (e.g. to 256 or 512) to keep the
> benchmark runtime reasonable.

### Apple M-series GPU (Metal — FP32 proxy)

```
Device          : Apple M3 Pro
NOTE            : Apple GPUs lack native FP64 hardware.
                  Measuring FP32 throughput as proxy.

Grid            : 1024 blocks × 256 threads = 262144 total threads
FMAs / thread   : 4096
Kernel repeats  : 10

Elapsed time    : 9.231 ms (0.923 ms / kernel)
Total FLOPs     : 21474836480
FP32 Throughput : 2328.45 GFLOPS  (FP64 proxy — Apple GPUs lack native FP64)
```

---

## Files

| File | Purpose |
|---|---|
| `fp64_throughput.cu` | CUDA source — kernel + driver code |
| `fp64_throughput.hip` | HIP source — AMD ROCm + NVIDIA HIP-on-CUDA |
| `fp64_throughput_metal.mm` | Apple Metal host code (Objective-C++) |
| `fp64_throughput.metal` | Metal compute kernel (FP32 FMA — Apple FP64 proxy) |
| `Makefile` | Build rules for all three platforms |
| `README.md` | This file |
