# shoc_sort — SHOC Sort Benchmark

GPU bitonic sort benchmark derived from the SHOC benchmark suite.

## Algorithm

Bitonic sort: a comparison-based parallel sorting network with O(n log² n)
compare-and-swap operations, fully parallelizable on the GPU. Each GPU
kernel pass performs one compare-and-swap step for all n elements
simultaneously.

## Default Parameters

| Parameter   | Default     | Description                         |
|-------------|-------------|-------------------------------------|
| `--size N`  | 4194304     | Number of elements (must be 2^k)    |
| `--block_size B` | 256    | GPU thread-block size               |
| `--precision P` | uint    | Element type (uint/int/float)       |

Environment variables (`BENCH_PARAM_size`, `BENCH_PARAM_block_size`,
`BENCH_PARAM_precision`) override the CLI flags, plus:

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_num_warmup` | 3 | Untimed warmup passes |
| `BENCH_PARAM_num_iters` | 7 | Timed passes, each recorded individually |

## Output

- **stdout** (JSON lines): per timed pass, one
  `{"type":"kernel","name":"bitonic_step",...}` record with that pass's
  accumulated GPU event time (`time_ms`, summed over all compare-and-swap
  steps) and host wall-clock time (`host_time_ms`), followed by a
  `{"type":"summary","total_time_ms":...}` line whose value is the sum of all
  recorded GPU event times. Melements/s is derived downstream.
- **stderr**: device info, per-pass timing, correctness check

## Build

### Auto-detect Platform
```bash
make
```

### NVIDIA CUDA
```bash
make PLATFORM=cuda
# Requires: nvcc (CUDA toolkit)
```

### AMD ROCm/HIP
```bash
make PLATFORM=rocm
# Requires: hipcc (ROCm installation at /opt/rocm)
```

### Apple Metal
```bash
make PLATFORM=metal
# Requires: Xcode command-line tools, macOS 10.14+
```

## Run

```bash
./shoc_sort
BENCH_PARAM_num_iters=10 ./shoc_sort --size 8388608
```

### Example Output (stderr)

```
Device: NVIDIA A100-SXM4-80GB (id=0)
Array size: 4194304 elements (16.0 MB)

pass 0: bitonic_step gpu = 45.231 ms, host = 47.102 ms
...
Correctness: PASS
Total GPU time: 315.812 ms over 7 passes
```

## Files

| File | Description |
|------|-------------|
| `shoc_sort.hip` | HIP source — bitonic sort kernel + host code (CUDA & ROCm) |
| `shoc_sort.metal` | Metal compute shader — bitonic sort kernel |
| `shoc_sort_metal.mm` | Objective-C++ Metal host code (Apple) |
| `Makefile` | Platform-aware build system |
| `README.md` | This file |

## Notes

- **Self-contained**: all utilities are inlined; no shared headers required.
- The Metal implementation dispatches one command buffer per (j, k) pass.
  For very large arrays this creates many small dispatches; this is the
  straightforward Metal bitonic sort approach.
- N must be a power of 2. Non-power-of-2 values are rounded up automatically.
