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

`NUM_ITERS` consecutive kernel launches are timed after one warm-up launch.
The reported latency is:

```
latency_ns = total_elapsed_ns / (num_accesses × num_iters)
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

For the Metal build, `cache_latency.metal` must be in the same directory as
the `cache_latency` binary (the host code loads it at runtime).

## Example Output

### CUDA / ROCm

```
=== cache_latency microbenchmark ===
Device : NVIDIA A100-SXM4-40GB (id=0)
Method : single-thread pointer chasing (random permutation)
Each test: one warm-up + 5 timed kernel launch(es)

Running tests...
  [L1 cache  ] array = 16 KB, accesses = 10000000, iters = 5
  [L2 cache  ] array = 2048 KB, accesses = 2000000, iters = 5
  [Global mem] array = 65536 KB, accesses = 300000, iters = 5

Cache level     Array size    Latency
──────────      ──────────    ──────────
L1 cache          16 KB        28.14 ns
L2 cache        2048 KB       126.83 ns
Global mem     65536 KB       482.37 ns

Done.
```

### Metal (Apple Silicon)

```
=== cache_latency microbenchmark ===
Device : Apple M3 Pro
Method : single-thread pointer chasing (random permutation)
Each test: one warm-up + 5 timed kernel launch(es)

Running tests...
  [L1 cache  ] array = 16 KB, accesses = 10000000, iters = 5
  [L2 cache  ] array = 2048 KB, accesses = 2000000, iters = 5
  [Global mem] array = 65536 KB, accesses = 300000, iters = 5

Cache level     Array size    Latency
──────────      ──────────    ──────────
L1 cache          16 KB        XX.XX ns
L2 cache        2048 KB       XXX.XX ns
Global mem     65536 KB       XXX.XX ns

Done.
```

*(Actual values depend on GPU model, clock frequency, and cache configuration.)*

## Notes

- Only a **single thread** participates — no thread-level parallelism that
  could hide latency.
- The output value (`result`) is written to device memory to prevent the
  compiler from eliminating the loop as dead code.
- No external dependencies beyond the GPU runtime (CUDA Toolkit / ROCm / Metal).
- The benchmark is self-contained — no CMake or shared build system required.
