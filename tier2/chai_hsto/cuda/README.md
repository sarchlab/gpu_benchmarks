# tier2/chai_hsto/cuda — Histogram (Chai) (native CUDA)

Native CUDA version of this benchmark. Kernels, driver, verification and
output are identical to [`../hip/`](../hip/); only runtime API names differ.
It builds with `nvcc` only, uses no architecture-specific features, and targets
A100 (sm_80) and H100 (sm_90). Shared timing/warmup headers come from
`common_cuda/`.

Benchmarks a 256-bin histogram computation on the GPU using shared memory privatization and global atomic reduction. The kernel runs exactly once per execution; its raw time is emitted and throughput (GB/s) is derived downstream.

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
| `chai_hsto.cu` | CUDA source — histogram kernel, CPU reference, verification |
| `Makefile` | nvcc build (`CUDA_ARCH`, default `native`) |

## Build

```bash
make                     # -arch=native (builds for the GPU in this machine)
make CUDA_ARCH=sm_80     # A100
make CUDA_ARCH=sm_90     # H100
```

## Run

```bash
./chai_hsto                          # defaults: N=16777216 (16M)
./chai_hsto --size 1048576           # smaller input (1M)
BENCH_PARAM_size=1048576 ./chai_hsto # env vars override argv
```

Environment variables: `BENCH_PARAM_size` (default 16777216),
`BENCH_PARAM_block_size` (default 256), and `BENCH_PARAM_vector_add_warmup`
(default true — cold-cache calibration: boosts the clock and evicts the LLC
before the timed launch). The histogram is a fixed 256-bin workload (one bin
per byte value); the bin count is not a parameter.

## Output

**stdout** — JSON lines, one kernel record for the single timed launch with
the GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`),
then a summary whose `total_time_ms` equals that launch's GPU event time (the
bin reset is untimed):
```
{"type":"kernel","name":"histogram_kernel","iter":0,"time_ms":1.234500,"host_time_ms":1.301200,"params":{"size":16777216,"block_size":256}}
{"type":"summary","total_time_ms":1.234500}
```

Derived metrics such as GB/s are computed downstream from `time_ms` and the
params.

**stderr** — human-readable results:
```
Device: NVIDIA H100 PCIe (id=0)
Histogram  |  N: 16777216 (16.00 MB)  |  Bins: 256

histogram_kernel: gpu = 1.234 ms, host = 1.301 ms
Total GPU time: 1.234 ms
PASS
```

## Verification

- All 256 bins are compared exactly against a CPU reference histogram
- Reports PASS/FAIL with per-bin mismatch details; on FAIL the binary exits with a non-zero status

## Platform Notes

- **CUDA**: compiled with `nvcc` from `chai_hsto.cu`; uses `__shared__` memory and `atomicAdd`
