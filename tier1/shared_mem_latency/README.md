# shared_mem_latency — Tier 1 Microbenchmark

## Overview

`shared_mem_latency` measures GPU shared-memory (SMEM / LDS / threadgroup
memory) **dependent-access latency** under two access patterns. A single
accumulator forms a `read -> add -> write` recurrence, so LDS accesses cannot
overlap and the kernel time reflects per-access round-trip latency, **not**
bandwidth.

> This benchmark was previously named `shared_mem_bandwidth`, but its
> single-accumulator recurrence measured latency, not bandwidth. It was renamed;
> see the `shared_mem_bandwidth` benchmark for the throughput test (independent
> accumulators, many reads in flight).

| Test | Description |
|------|-------------|
| **no_conflict** | Each thread in a warp/SIMD-group accesses a different bank (stride-1, sequential) |
| **conflict** | All threads in a warp/SIMD-group access the same bank (stride-32 between lanes) — 32-way bank conflict |

The inner-loop iteration count scales total work; the dependent chain means the
time grows with the number of serialized LDS round-trips.

## Platform Support

| Platform | Source file | Compiler |
|----------|-------------|---------|
| NVIDIA CUDA | `shared_mem_latency.cu` | `nvcc` |
| AMD ROCm/HIP | `shared_mem_latency.hip` | `hipcc` |
| Apple Metal | `shared_mem_latency_metal.mm` + `shared_mem_latency.metal` | `clang++` |

## What It Measures

- **no_conflict bandwidth** — achievable shared memory throughput when bank
  conflicts are avoided.  Consecutive threads in a warp access consecutive
  addresses, spreading load across all 32 banks.
- **conflict bandwidth** — degraded throughput when every thread in a warp
  maps to the same bank.  The 32-way bank conflict serialises accesses,
  reducing effective bandwidth significantly.

The ratio `no_conflict / conflict` indicates how much performance is lost
due to bank conflicts on the target hardware.

## Build

The directory is **self-contained** — no CMake or external dependencies.

### Auto-detect platform

```bash
cd tier1/shared_mem_latency
make
```

The Makefile auto-detects the platform:
- macOS (Darwin) → Metal
- Linux with `/opt/rocm` → ROCm/HIP
- Otherwise → CUDA

### Explicit platform

```bash
# NVIDIA CUDA
make PLATFORM=cuda

# AMD ROCm/HIP
make PLATFORM=rocm

# Apple Metal
make PLATFORM=metal
```

### Clean

```bash
make clean
```

## Run

```bash
./shared_mem_latency
```

No command-line arguments are needed.  All parameters are compile-time
constants.

## Example Output

### CUDA / HIP

```
Device        : NVIDIA A100-SXM4-80GB (id=0)
Shared mem/blk: 49152 bytes
Block size    : 256 threads
Grid size     : 128 blocks
Inner iters   : 1000
Warmup iters  : 5
Timed iters   : 5

no_conflict    :    5.432 ms  2302.45 GB/s
conflict       :   78.901 ms   158.63 GB/s
shared_mem_bw,no_conflict,5.432,2302.45
shared_mem_bw,conflict,78.901,158.63

Done.
```

### Metal (Apple M3)

```
Device        : Apple M3 Max
Block size    : 256 threads
Grid size     : 128 groups
Shared mem    : 49152 bytes (12288 floats)
Inner iters   : 1000
Warmup iters  : 5
Timed iters   : 5

no_conflict    :    8.123 ms  1540.12 GB/s
conflict       :   95.456 ms   131.03 GB/s
shared_mem_bw,no_conflict,8.123,1540.12
shared_mem_bw,conflict,95.456,131.03

Done.
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BLOCK_SIZE` | 256 | Threads per block / threadgroup |
| `NUM_BLOCKS` | 128 | Number of blocks / threadgroups |
| `INNER_ITERS` | 1000 | Inner-loop iterations per kernel launch |
| `SMEM_FLOATS` | 12288 | Shared memory array size in floats (48 KB) |
| `WARMUP_ITERS` | 5 | Outer warmup kernel launches |
| `TIMED_ITERS` | 5 | Outer timed kernel launches |

## Notes

- Results are in SI gigabytes per second (1 GB = 10⁹ bytes).
- Bandwidth accounts for both reads and writes:
  `GB/s = 2 × SMEM_FLOATS × 4 × INNER_ITERS × NUM_BLOCKS / time_s / 1e9`
- The **no_conflict** kernel uses a simple stride-`blockDim.x` loop where
  consecutive threads access consecutive shared memory addresses.
- The **conflict** kernel remaps indices so that within each warp, all 32
  lanes access an address that maps to the same bank (stride-32 between
  lanes), causing maximum serialisation.
- Both kernels perform the same total number of reads and writes, making
  the bandwidth numbers directly comparable.
- **CUDA/HIP**: CUDA/HIP events are used for high-resolution GPU-side timing.
- **Metal**: `mach_absolute_time()` wall-clock timing is used; command buffers
  are serialised (`waitUntilCompleted`) for accurate measurement.
