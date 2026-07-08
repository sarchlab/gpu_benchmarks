# tier2/parboil_histogram — Atomic Histogram Benchmark

Computes a 256-bin histogram of random 32-bit integers using GPU atomic operations.
Derived from the Parboil benchmark suite histogram workload.

## Algorithm

1. Generate N random `uint32_t` values on the host (seeded at 42).
2. Upload the data array to the GPU (once, reused across iterations).
3. Launch a GPU kernel: each thread reads one element, computes `bin = val & 0xFF`, and performs `atomicAdd(&hist[bin], 1)`.
4. Run `num_warmup` (default 3) untimed warmup passes, then `num_iters` (default 7) timed passes, each recorded individually — the histogram buffer is zeroed (untimed) before every pass.
5. Copy the result histogram back and verify against a CPU reference (exact integer comparison of all 256 bins).
6. Emit per-iteration kernel times; bandwidth (GB/s) and throughput (Melements/s) are derived downstream.

**Kernel configuration**: 256 threads per block, ⌈N/256⌉ blocks.

## Files

| File | Description |
|------|-------------|
| `parboil_histogram.hip` | HIP/CUDA kernel + host code (NVIDIA & AMD) |
| `parboil_histogram.metal` | Metal compute shader (Apple GPU) |
| `parboil_histogram_metal.mm` | Objective-C++ Metal host (Apple GPU) |
| `Makefile` | Build system with auto-platform detection |
| `README.md` | This file |

## Build

Platform is auto-detected but can be overridden:

```bash
# Auto-detect (Darwin → metal, /opt/rocm present → rocm, else → cuda)
make

# Explicit platform
make PLATFORM=cuda    # NVIDIA CUDA
make PLATFORM=rocm    # AMD ROCm/HIP
make PLATFORM=metal   # Apple Metal

make clean
```

## Run

```bash
./parboil_histogram                          # default: N=16M
./parboil_histogram --n 33554432             # custom N (32M elements)
BENCH_PARAM_num_iters=10 ./parboil_histogram # custom timed-pass count
```

Environment variables (override argv):

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_n` | 16777216 | Number of input elements |
| `BENCH_PARAM_num_bins` | 256 | Histogram bins (echoed in params) |
| `BENCH_PARAM_block_size` | 256 | Threads per block |
| `BENCH_PARAM_num_warmup` | 3 | Untimed warmup passes |
| `BENCH_PARAM_num_iters` | 7 | Timed passes, each recorded individually |

## Output Format

**stderr** (human-readable):
```
Device: Apple M2 Pro
N = 16777216 elements  |  Bins = 256  |  Iterations: 3 warmup + 7 timed

histogram_kernel iter 0: gpu = 3.215 ms, host = 3.301 ms
...
Correctness check (all 256 bins): PASS

Total GPU time: 22.501 ms over 7 iterations
```

**stdout** (JSON lines): one `{"type":"kernel","name":"histogram_kernel",...}`
record per timed pass with that pass's GPU event time (`time_ms`) and host
wall-clock time (`host_time_ms`), followed by a
`{"type":"summary","total_time_ms":...}` line whose value is the sum of all
recorded GPU event times.

## Performance Metric (derived downstream)

```
GB/s = N × sizeof(uint32_t) / time_s / 1e9
     = N × 4 / time_s / 1e9

Melements/s = N / time_s / 1e6
```

These are computed downstream from each record's `time_ms` and params; the
benchmark itself emits raw times only.

## Correctness Verification

A CPU reference histogram is computed using the same input data and random seed.
All 256 bins are compared exactly (integer equality). Prints `PASS` or `FAIL` to stderr.

## Default Parameters

| Parameter | Value |
|-----------|-------|
| N (elements) | 16,777,216 (16 × 1024 × 1024) |
| Bins | 256 |
| Block size | 256 threads |
| Warmup / timed passes | 3 / 7 |
| Bin mapping | `val & 0xFF` (lower 8 bits) |
