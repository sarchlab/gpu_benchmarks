# cache_latency — GPU Cache/Memory Latency Microbenchmark

## Purpose

Measures the round-trip **load latency** (in nanoseconds per access) for each
level of the GPU memory hierarchy:

| Test        | Array size | Targeted level               |
|-------------|------------|------------------------------|
| L1 cache    | 16 KB      | L1 data cache (~32–128 KB)   |
| L2 cache    | 2 MB       | L2 cache (~4–16 MB)          |
| Global mem  | 64 MB      | DRAM (exceeds L2)            |

## Methodology

### Pointer Chasing

The benchmark allocates an array of `uint32_t` indices and wires them into a
**single Hamiltonian cycle** using a Knuth (Fisher–Yates) random shuffle:

```
arr[perm[0]] = perm[1]
arr[perm[1]] = perm[2]
...
arr[perm[N-1]] = perm[0]
```

A single GPU thread (one warp, one block) then walks the chain:

```c
for (uint32_t i = 0; i < num_accesses; ++i) {
    idx = arr[idx];   // dependent load
}
```

The **true data dependency** (`idx = arr[idx]`) forces the hardware to fully
resolve each load before the address of the next load is known.  This prevents
any memory-level parallelism and exposes the raw latency of the targeted cache
level.

### Avoiding Prefetchers

The traversal order is a random permutation, so there is no predictable stride
pattern that hardware stream-prefetchers could exploit.

### Timing

| Platform | Timing method |
|----------|---------------|
| CUDA     | `cudaEvent_t` — GPU-side hardware timer |
| ROCm/HIP | `hipEvent_t`  — GPU-side hardware timer |
| Metal    | `mach_absolute_time()` — wall-clock timer on Apple Silicon |

`BENCH_PARAM_num_iters` (default 7) consecutive kernel launches are timed
after `BENCH_PARAM_num_warmup` (default 3) untimed warm-up launches (the cold
fill lap is discarded). Every timed launch is emitted as its own record — no
averaging or min-keeping. Latency is not printed by the binary; it is derived
downstream from each record's `time_ms` and `num_accesses` params:

```
latency_ns = time_ms × 1e6 / num_accesses
```

### Cache Level Targeting

| Level   | Array size | Why it hits that level                                              |
|---------|------------|---------------------------------------------------------------------|
| L1      | 16 KB      | Fits inside the per-SM/CU L1 cache (≥ 16 KB on all supported GPUs) |
| L2      | 2 MB       | Larger than L1 but fits inside L2 (typically 4–16 MB)              |
| Global  | 64 MB      | Exceeds L2; every access goes to DRAM                              |

## Files

| File                      | Description                                      |
|---------------------------|--------------------------------------------------|
| `cache_latency.hip`       | HIP implementation (AMD ROCm + NVIDIA HIP-on-CUDA) |
| `cache_latency_metal.mm`  | Apple Metal host code (Objective-C++)            |
| `cache_latency.metal`     | Apple Metal shader (Metal Shading Language)      |
| `Makefile`                | Build system with platform auto-detection        |

## Building

The Makefile auto-detects the platform:
- **Darwin** → Metal
- **Linux with `/opt/rocm`** → ROCm/HIP
- **Linux (default)** → CUDA

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

## Running

```bash
./cache_latency
```

Parameters are set via `BENCH_PARAM_*` environment variables (e.g.
`BENCH_PARAM_array_bytes`, `BENCH_PARAM_cacheline_bytes`,
`BENCH_PARAM_measure_laps`, `BENCH_PARAM_rng_seed`), plus the harness
controls `BENCH_PARAM_num_warmup` (untimed warm-up launches, default 3) and
`BENCH_PARAM_num_iters` (timed launches, each recorded, default 7).

For the Metal build, `cache_latency.metal` must be in the same directory as
the `cache_latency` binary (the host code loads it at runtime).

## Example Output

Progress and device info go to stderr; stdout carries JSON lines — one record
per timed iteration plus a final summary:

```
{"type":"kernel","name":"pointer_chase_kernel","iter":0,"time_ms":12.483200,"host_time_ms":12.561004,"params":{"array_bytes":16384,"num_accesses":1024,"warmup_accesses":256,"measure_laps":4,"cacheline_bytes":64,"rng_seed":42}}
...
{"type":"summary","total_time_ms":87.512300}
```

- `time_ms` is the GPU event time for that launch; `host_time_ms` is the host
  wall-clock around launch + sync.
- `total_time_ms` is the sum of all iterations' GPU event times.
- No latency figure is printed; ns/access is derived downstream from
  `time_ms` and `num_accesses`.

*(Actual values depend on GPU model, clock frequency, and cache configuration.)*

## Notes

- Only a **single thread** participates — no thread-level parallelism that
  could hide latency.
- The output value (`result`) is written to device memory to prevent the
  compiler from eliminating the loop as dead code.
- No external dependencies beyond the GPU runtime (CUDA Toolkit / ROCm / Metal).
- The benchmark is self-contained — no CMake or shared build system required.
