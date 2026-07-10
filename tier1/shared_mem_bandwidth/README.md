# shared_mem_bandwidth — Tier 1 Microbenchmark

## Overview

`shared_mem_bandwidth` measures GPU shared memory (SMEM / LDS / threadgroup
memory) bandwidth under two access patterns:

| Test | Description |
|------|-------------|
| **no_conflict** | Each thread in a warp/SIMD-group accesses a different bank (stride-1, sequential) — maximum throughput |
| **conflict** | All threads in a warp/SIMD-group access the same bank (stride-32 between lanes) — 32-way bank conflict |

The benchmark stresses the shared memory subsystem with a high inner-loop
iteration count (1000) to amortize kernel launch overhead and produce stable
bandwidth numbers.

## Platform Support

| Platform | Source file | Compiler |
|----------|-------------|---------|
| NVIDIA CUDA | `shared_mem_bandwidth.hip` | `nvcc` |
| AMD ROCm/HIP | `shared_mem_bandwidth.hip` | `hipcc` |
| Apple Metal | `shared_mem_bandwidth_metal.mm` + `shared_mem_bandwidth.metal` | `clang++` |

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
cd tier1/shared_mem_bandwidth
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
./shared_mem_bandwidth
```

No command-line arguments are needed.  Parameters are set via
`BENCH_PARAM_*` environment variables (see below).

## Example Output

Progress and device info go to stderr; stdout carries JSON lines — one record
per timed iteration of each selected pattern, plus a final summary:

```
{"type":"kernel","name":"smem_bw_no_conflict","iter":0,"time_ms":5.432101,"host_time_ms":5.541234,"params":{"inner_iters":1000,"block_size":256,"num_blocks":128,"access_pattern":"both"}}
...
{"type":"kernel","name":"smem_bw_conflict","iter":0,"time_ms":78.901234,"host_time_ms":79.012345,"params":{...}}
...
{"type":"summary","total_time_ms":590.334012}
```

- Each pattern emits one record per timed iteration — no averaging.
- `time_ms` is the GPU event time for that launch; `host_time_ms` is the
  host wall-clock around launch + sync.
- `total_time_ms` is the sum of all emitted iterations' GPU event times
  across every pattern that ran.
- GB/s is not printed; it is derived downstream from `time_ms` and the
  params in each record.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BENCH_PARAM_block_size` | 256 | Threads per block / threadgroup |
| `BENCH_PARAM_num_blocks` | 128 | Number of blocks / threadgroups |
| `BENCH_PARAM_inner_iters` | 1000 | Inner-loop iterations per kernel launch |
| `BENCH_PARAM_access_pattern` | both | Pattern(s) to run (`no_conflict`, `conflict`, `both`) |
| `BENCH_PARAM_warmup` | 3 | Outer warmup kernel launches (not timed) |
| `BENCH_PARAM_iteration` | 7 | Outer timed kernel launches, each recorded individually |
| `SMEM_FLOATS` (compile-time) | 12288 | Shared memory array size in floats (48 KB) |

## Notes

- Bandwidth in SI gigabytes per second (1 GB = 10⁹ bytes) is derived
  downstream from the recorded times, accounting for both reads and writes:
  `GB/s = 2 × SMEM_FLOATS × 4 × inner_iters × num_blocks / time_s / 1e9`
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
