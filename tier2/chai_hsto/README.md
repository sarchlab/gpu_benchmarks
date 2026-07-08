# tier2/chai_hsto — Histogram (Chai)

Benchmarks a 256-bin histogram computation on the GPU using shared memory privatization and global atomic reduction. Emits raw per-iteration kernel times; throughput (GB/s) is derived downstream.

Derived from the [Chai benchmark suite](https://github.com/chai-benchmarks/chai) HSTO (histogram) collaborative computing pattern.

## Algorithm

- **N** random bytes as input (default: N=16,777,216 = 16M bytes = 16 MB)
- 256 histogram bins (one per byte value 0–255)
- GPU implementation:
  1. Each threadgroup initializes a private histogram in shared/threadgroup memory (256 bins)
  2. Threads iterate over input data (grid-stride loop), atomically incrementing shared bins
  3. After barrier, each threadgroup atomically adds its private bins to the global histogram
- This "privatization + reduction" pattern reduces global atomic contention
- GB/s = N × sizeof(uint8\_t) / time\_sec / 10⁹ (derived downstream from the emitted times and params)

## Files

| File | Description |
|------|-------------|
| `chai_hsto.hip` | HIP source — histogram kernel, CPU reference, verification |
| `chai_hsto.metal` | Metal compute shader — histogram kernel |
| `chai_hsto_metal.mm` | ObjC++ Metal host code |
| `Makefile` | Auto-detecting build (Darwin→metal, ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
make PLATFORM=metal
```

## Run

```bash
./chai_hsto                          # defaults: N=16777216 (16M)
./chai_hsto --size 1048576           # smaller input (1M)
BENCH_PARAM_size=1048576 ./chai_hsto # env vars override argv
```

Environment variables: `BENCH_PARAM_size`, `BENCH_PARAM_num_bins`,
`BENCH_PARAM_block_size`, plus `BENCH_PARAM_num_warmup` (warmup passes,
default 3) and `BENCH_PARAM_num_iters` (timed passes, each recorded,
default 7).

## Output

**stdout** — JSON lines, one kernel record per timed pass with the GPU event
time (`time_ms`) and host wall-clock time (`host_time_ms`), then a summary
whose `total_time_ms` is the sum of all recorded GPU event times (the
per-pass bin reset is untimed):
```
{"type":"kernel","name":"histogram_kernel","iter":0,"time_ms":1.234500,"host_time_ms":1.301200,"params":{"size":16777216,"num_bins":256,"block_size":256}}
...
{"type":"summary","total_time_ms":8.641500}
```

Derived metrics such as GB/s are computed downstream from `time_ms` and the
params.

**stderr** — human-readable results:
```
Device: Apple M2
Histogram  |  N: 16777216 (16.00 MB)  |  Bins: 256  |  Iterations: 3 warmup + 7 timed

histogram_kernel iter 0: gpu = 1.234 ms, host = 1.301 ms
...
Total GPU time: 8.642 ms over 7 iterations
PASS
```

## Verification

- All 256 bins are compared exactly against a CPU reference histogram
- Reports PASS/FAIL with per-bin mismatch details

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `chai_hsto.hip`; uses `__shared__` memory and `atomicAdd`
- **Metal**: compiled with `clang++` from `chai_hsto_metal.mm`; the Metal shader (`chai_hsto.metal`) is loaded and compiled at runtime from the same directory as the binary; uses `threadgroup atomic_uint` and `atomic_fetch_add_explicit`
