# warp_shuffle — Tier 1 Microbenchmark

## Overview

`warp_shuffle` measures warp/wave/SIMD shuffle instruction throughput by
executing a tight chain of shuffle operations on every GPU thread.

Each thread performs `ops_per_thread` shuffle operations in a dependency
chain (each shuffle uses the result of the previous one). The binary emits
raw kernel times; the shuffle rate in Giga-Operations per second is derived
downstream from each record's `time_ms` and params:

```
Gops = (num_threads × ops_per_thread) / elapsed_seconds / 1e9
```

(Optionally, an effective data-movement rate can be derived by multiplying
by `sizeof(int)` = 4 bytes per shuffle.)

Four shuffle variants are benchmarked:

| Variant | HIP Intrinsic | Description |
|---------|--------------|-------------|
| `shfl_xor` | `__shfl_xor` | Butterfly / XOR shuffle |
| `shfl` | `__shfl` | Indexed (direct) shuffle |
| `shfl_up` | `__shfl_up` | Shift up by delta |
| `shfl_down` | `__shfl_down` | Shift down by delta |

## Platform Support

| Platform | Source file | Compiler |
|----------|-------------|---------|
| NVIDIA CUDA | `warp_shuffle.hip` | `nvcc -x cu` |
| AMD ROCm/HIP | `warp_shuffle.hip` | `hipcc` |

## What It Measures

- **Warp/SIMD shuffle throughput** — the rate at which lanes can exchange data
  via hardware shuffle instructions without going through shared memory or
  global memory.
- Shuffles are a fundamental building block for reductions, scans, transposes,
  and other warp-level primitives.
- The benchmark creates a dependency chain to prevent compiler elimination
  while reflecting the hardware's true shuffle throughput.

## Build

The directory is **self-contained** — no CMake or external dependencies.

### Auto-detect platform

```bash
cd tier1/warp_shuffle
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
# Default 4096 shuffle ops per thread
./warp_shuffle

# Custom ops count per thread
./warp_shuffle 8192
```

Each execution launches and times each selected variant's kernel **exactly
once** — there is no in-program warmup or iteration loop. Unless
`BENCH_PARAM_vector_add_warmup` is set to `false` (default `true`), a
vector-add warmer runs ~100 ms of sustained load immediately before each
timed launch, boosting the GPU clock from idle and evicting the last-level
cache, so the kernel is measured cold-cache at a boosted clock. Repeat
measurements come from fresh process re-executions: `gpubench run --reps N`.

## Output Format

Human-readable progress goes to stderr; stdout carries JSON lines — one
record per selected variant (each timed once), plus a final summary:

```
{"type":"kernel","name":"shfl_xor_kernel","iter":0,"time_ms":2.469123,"host_time_ms":2.561234,"params":{"ops_per_thread":4096,"num_blocks":1,"threads_per_block":256}}
{"type":"kernel","name":"shfl_kernel","iter":0,"time_ms":2.513456,"host_time_ms":2.601234,"params":{...}}
{"type":"kernel","name":"shfl_up_kernel","iter":0,"time_ms":2.578901,"host_time_ms":2.671234,"params":{...}}
{"type":"kernel","name":"shfl_down_kernel","iter":0,"time_ms":2.491234,"host_time_ms":2.581234,"params":{...}}
{"type":"summary","total_time_ms":10.052714}
```

- Each variant that runs emits exactly one record.
- `time_ms` is the GPU event time for that launch; `host_time_ms` is the
  host wall-clock around launch + sync.
- `total_time_ms` is the sum of the emitted records' GPU event times across
  every variant that ran.
- Gops/s is not printed; it is derived downstream from `time_ms` and params.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BENCH_PARAM_ops_per_thread` (or argv[1]) | 4096 | Shuffle iterations per thread |
| `BENCH_PARAM_threads_per_block` | 256 | Threads per block / threadgroup |
| `BENCH_PARAM_num_blocks` | 1 | Number of blocks in the dispatch grid (default 1 = single-SM/CU, latency-bound measurement) |
| `BENCH_PARAM_shuffle_variant` | all | Variant(s) to run (`shfl_xor`, `shfl`, `shfl_up`, `shfl_down`, `all`) |
| `BENCH_PARAM_vector_add_warmup` | true | Run the vector-add warmer (clock boost + LLC eviction) before each timed launch; set `false` to time the kernels as-is |

> **Single-SM, latency-bound measurement.** The default grid is one block,
> and each thread's shuffles form a serial dependent chain, so the reported
> rate is the dependent-shuffle issue rate of one SM (latency-bound) — not
> peak shuffle throughput, and multiplying by SM count does not give a
> device peak. Raising `BENCH_PARAM_num_blocks` or `threads_per_block` adds
> warp-level parallelism, but the per-thread chain remains dependent by
> design.

## Notes

- Gops/s (10⁹ shuffle operations per second) is derived downstream from the
  recorded times; the binary itself only emits raw times. An effective
  data-movement rate can optionally be derived by multiplying by 4 bytes
  per shuffle.
- The vector-add warmer (on by default) boosts the GPU clock and evicts the
  last-level cache before each timed launch; there are no untimed warm-up
  launches of the benchmark kernels themselves.
- **CUDA/HIP**: HIP events (`hipEvent_t`) are used for high-resolution
  GPU-side timing. HIP shuffle intrinsics (`__shfl_xor`, etc.) do not require
  the `_sync` suffix — they operate on the full warp/wavefront.
- The warp size is 64 on AMD GCN/CDNA GPUs and 32 on NVIDIA GPUs. The HIP
  code automatically selects the correct size at compile time.
- A dependency chain ensures the compiler cannot optimize away the shuffle
  operations; only thread 0 of block 0 writes the final result to prevent
  memory traffic from dominating the measurement.
