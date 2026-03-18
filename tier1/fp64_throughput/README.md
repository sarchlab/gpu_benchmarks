# fp64_throughput

Microbenchmark that measures the peak **double-precision (FP64) floating-point
compute throughput** of an NVIDIA GPU in GFLOPS (10⁹ floating-point operations
per second).

---

## What it measures

The benchmark saturates the GPU's FP64 arithmetic pipelines by executing a
dense loop of **fused multiply-add (FMA)** instructions:

```c
a = fma(a, multiplier, addend);   // 2 FLOPs: 1 multiply + 1 add
```

All operands live in registers — there is no memory traffic beyond a single
8-byte write at the end — so the result reflects raw arithmetic throughput
rather than memory bandwidth.

Four independent accumulators are used per thread so the compiler can
interleave FMA instructions and hide the FP64 pipeline latency, achieving
full issue-rate throughput.

---

## FP64 vs FP32 throughput

FP64 throughput varies significantly by GPU class:

| GPU class | FP64 : FP32 ratio |
|---|---|
| Consumer (GeForce RTX) | 1:32 or 1:64 |
| Professional (Quadro / RTX A-series) | 1:16 or 1:32 |
| Datacenter / HPC (V100, A100, H100) | ~1:2 |

Results from this benchmark will reflect the actual FP64 capability of the
device under test.

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
cmake --build build --target fp64_throughput -j$(nproc)
```

To target specific GPU architectures:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="70;80;86;90"
```

---

## Usage

```
./fp64_throughput [fmas_per_thread]
```

| Argument | Default | Description |
|---|---|---|
| `fmas_per_thread` | `4096` | Number of FMA iterations per thread. Increase on HPC GPUs where FP64 is fast; the kernel should run for at least ~5 ms for a stable measurement. |

---

## Example output

### Datacenter GPU (e.g. NVIDIA A100-SXM4-80GB)

```
Device          : NVIDIA A100-SXM4-80GB (id=0)
SMs             : 108
Grid            : 1024 blocks × 256 threads = 262144 total threads
FMAs / thread   : 4096
Kernel repeats  : 10

Elapsed time    : 119.872 ms (11.987 ms / kernel)
Total FLOPs     : 21474836480
FP64 Throughput : 179.16 GFLOPS
```

### Consumer GPU (e.g. NVIDIA GeForce RTX 3080)

```
Device          : NVIDIA GeForce RTX 3080 (id=0)
SMs             : 68
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

---

## Files

| File | Purpose |
|---|---|
| `fp64_throughput.cu` | CUDA source — kernel + driver code |
| `CMakeLists.txt` | CMake build rules |
| `README.md` | This file |
