# atomic_operations — Tier 1 Microbenchmark

## Overview

`atomic_operations` measures the throughput of atomic operations on global
memory by having all GPU threads contend on a single shared counter. Three
atomic operations are benchmarked:

| Operation   | HIP                |
|-------------|--------------------|
| atomicAdd   | `atomicAdd()`      |
| atomicCAS   | `atomicCAS()`      |
| atomicMin   | `atomicMin()`      |

The binary reports the raw kernel time; throughput in Giga-Operations per
second is derived downstream from the recorded `time_ms` and params:

```
Gops = (num_threads × ops_per_thread) / elapsed_seconds / 1e9
```

## Platform Support

| Platform       | Source file(s)                                              | Compiler |
|----------------|-------------------------------------------------------------|----------|
| NVIDIA CUDA    | `atomic_operations.hip`                                     | `nvcc`   |
| AMD ROCm/HIP   | `atomic_operations.hip`                                     | `hipcc`  |

## What It Measures

- **Atomic unit throughput** — maximum contention scenario where all threads
  atomically update a single global memory location. This measures the peak
  throughput of the GPU's atomic hardware units (L2 cache atomics on NVIDIA,
  dedicated atomic units on AMD).
- Each execution benchmarks one operation type, selected by
  `BENCH_PARAM_atomic_op`, with a single timed kernel launch.
- **atomicMin caveat** — the counter is reset to 0 before the launch, and
  every thread calls `atomicMin(counter, tid + i)` with a non-negative value,
  so the minimum never changes: no atomicMin ever succeeds in writing. This
  mode therefore measures the throughput of the compare-fail (no-write) path
  of the atomic unit, not of value-updating atomics.
- **atomicCAS caveat** — each op is a plain load of the counter followed by
  one `atomicCAS(counter, old, old + 1)`; success is not checked and there is
  no retry loop, so under contention most CAS attempts fail.

## Build

The directory is **self-contained** — no CMake or external dependencies.

### Auto-detect platform

```bash
cd tier1/atomic_operations
make
```

The Makefile auto-detects the platform:
- Linux with `/opt/rocm` → ROCm/HIP
- Otherwise → CUDA

### Explicit platform

```bash
# NVIDIA CUDA
make PLATFORM=cuda

# AMD ROCm/HIP
make PLATFORM=rocm
```

### Clean

```bash
make clean
```

## Run

```bash
# Default 1024 ops per thread
./atomic_operations

# Custom ops per thread (increase for faster GPUs)
./atomic_operations 4096
```

Each execution launches and times the selected kernel **exactly once** —
there is no in-program warmup or iteration loop. Unless
`BENCH_PARAM_vector_add_warmup` is set to `false` (default `true`), a
vector-add warmer runs ~100 ms of sustained load immediately before the timed
launch, boosting the GPU clock from idle and evicting the last-level cache,
so the kernel is measured cold-cache at a boosted clock. Repeat measurements
come from fresh process re-executions: `gpubench run --reps N`.

## Output Format

Progress and device info go to stderr; stdout carries JSON lines — one record
for the single timed launch plus a final summary:

```
{"type":"kernel","name":"atomicAdd_kernel","iter":0,"time_ms":25.086432,"host_time_ms":25.201004,"params":{"ops_per_thread":1024,"num_blocks":2048,"threads_per_block":256,"atomic_op":"atomicAdd"}}
{"type":"summary","total_time_ms":25.086432}
```

- `time_ms` is the GPU event time for the single kernel launch;
  `host_time_ms` is the host wall-clock around launch + sync.
- `total_time_ms` equals the single launch's GPU event time.
- No derived metrics are printed; Gops/s is computed downstream from
  `time_ms` and the params in the record.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BENCH_PARAM_ops_per_thread` (or argv[1]) | 1024 | Number of atomic operations each thread performs |
| `BENCH_PARAM_threads_per_block` | 256 | Threads per block / threadgroup |
| `BENCH_PARAM_num_blocks` | 2048 | Number of blocks in the dispatch grid |
| `BENCH_PARAM_atomic_op` | atomicAdd | Atomic operation to benchmark (`atomicAdd`, `atomicCAS`, `atomicMin`) |
| `BENCH_PARAM_vector_add_warmup` | true | Run the vector-add warmer (clock boost + LLC eviction) before the timed launch; set `false` to time the kernel as-is |

## Notes

- Gops/s (10⁹ atomic operations per second) is derived downstream from the
  recorded times; the binary itself only emits raw times.
- The vector-add warmer (on by default) boosts the GPU clock and evicts the
  last-level cache before the single timed launch; there are no untimed
  warm-up launches of the benchmark kernel itself.
- **CUDA/HIP**: HIP events are used for high-resolution GPU-side timing.
- All threads contend on a single global counter — this is the worst-case
  contention scenario and reflects the hardware atomic throughput limit.
- atomicCAS uses a read-then-CAS pattern: each op is a plain load of the
  counter plus one `atomicCAS`, regardless of success (no retry loop).
- atomicMin never succeeds in writing (the counter stays at its 0 minimum),
  so it measures the compare-fail path — see What It Measures.
