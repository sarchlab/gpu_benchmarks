# tier2/npb_is — Integer Sort (NAS IS)

Benchmarks a **parallel bucket sort** of N random integers in range [0, 2¹⁹). Emits raw per-iteration kernel times; throughput (Mkeys/sec) is derived downstream.

Derived from the [NAS Parallel Benchmarks](https://www.nas.nasa.gov/software/npb.html) IS kernel.

## Algorithm

- **N** random integer keys (default: N=8,388,608 = 2²³) in range [0, 524,288 = 2¹⁹)
- **NUM_BUCKETS** = 1,024 (2¹⁰), each covering a range of 512 key values
- Three phases:
  1. **Histogram** — count keys per bucket (atomic increments)
  2. **Prefix sum** — exclusive scan on the histogram (Blelloch algorithm, single threadgroup)
  3. **Scatter** — place each key at its sorted position (atomic offset increment)
- Result: keys sorted by bucket (bucket-level ordering)

## Files

| File | Description |
|------|-------------|
| `npb_is.hip` | HIP source — histogram, prefix sum, and scatter kernels |
| `npb_is.metal` | Metal compute shaders — histogram, prefix sum, and scatter |
| `npb_is_metal.mm` | ObjC++ Metal host code |
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
./npb_is                              # defaults: N=8388608
./npb_is --size 1048576               # smaller workload
BENCH_PARAM_size=1048576 ./npb_is     # env vars override argv
```

Environment variables: `BENCH_PARAM_size`, `BENCH_PARAM_block_size`,
`BENCH_PARAM_verify`, plus `BENCH_PARAM_num_warmup` (warmup sort passes,
default 3) and `BENCH_PARAM_num_iters` (timed sort passes, each recorded,
default 7).

## Output

**stdout** — JSON lines: per timed sort pass, one kernel record per phase
(histogram, prefix sum, scatter) with that phase's GPU event time
(`time_ms`) and host wall-clock time (`host_time_ms`), then a summary whose
`total_time_ms` is the sum of all recorded GPU event times. The per-pass
histogram reset and device-to-device offset copy are untimed:
```
{"type":"kernel","name":"histogram_kernel","iter":0,"time_ms":1.523400,"host_time_ms":1.580000,"params":{"size":8388608,"block_size":256,"verify":"true"}}
{"type":"kernel","name":"prefix_sum_kernel","iter":0,"time_ms":0.041200,"host_time_ms":0.058000,"params":{...}}
{"type":"kernel","name":"scatter_kernel","iter":0,"time_ms":3.558800,"host_time_ms":3.610000,"params":{...}}
...
{"type":"summary","total_time_ms":35.864800}
```

Derived metrics such as Mkeys/sec are computed downstream from `time_ms` and
the params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
IS keys: 8388608 (2^23)  |  Buckets: 1024  |  Iterations: 3 warmup + 7 timed

pass 0: histogram gpu = 1.523 ms, prefix_sum gpu = 0.041 ms, scatter gpu = 3.559 ms
...
Total GPU time: 35.865 ms over 7 passes
PASS
```

## Verification

- Checks that the sorted output is non-decreasing at bucket boundaries (bucket-ordered)
- Each key's bucket index = key / (MAX_KEY / NUM_BUCKETS); consecutive elements must have non-decreasing bucket indices

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `npb_is.hip`; three kernel launches per sort (histogram + prefix sum + scatter)
- **Metal**: compiled with `clang++` from `npb_is_metal.mm`; the Metal shader (`npb_is.metal`) is loaded and compiled at runtime from the same directory as the binary
