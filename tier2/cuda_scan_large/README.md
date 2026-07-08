# tier2/cuda_scan_large — Work-Efficient Parallel Prefix Sum (Blelloch Scan)

Benchmarks a **work-efficient parallel exclusive prefix sum** (Blelloch scan) on a large array of unsigned integers. Emits per-iteration kernel times; GB/s is derived downstream.

Derived from the [CUDA SDK scanLargeArray](https://github.com/NVIDIA/cuda-samples) sample.

## Algorithm

- **N** unsigned integers (default: N=16,777,216 = 16M)
- Three-phase hierarchical Blelloch scan:
  1. **Block-level scan** — each block of 512 elements performs an in-place exclusive scan using shared memory (up-sweep + down-sweep), stores block total
  2. **Scan of block sums** — recursively scan the array of block totals
  3. **Add block sums back** — add each scanned block total to all elements in the corresponding block
- Handles arbitrarily large arrays via recursive decomposition
- GB/s = 2 · N · sizeof(uint) / time\_sec / 10⁹ (derived downstream from the emitted times and params)

## Files

| File | Description |
|------|-------------|
| `cuda_scan_large.hip` | HIP source — block scan + add-back kernels with recursive dispatch |
| `cuda_scan_large.metal` | Metal compute shaders — block scan + add-back kernels |
| `cuda_scan_large_metal.mm` | ObjC++ Metal host code with recursive scan |
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
./cuda_scan_large                              # defaults: N=16777216
./cuda_scan_large --size 1048576               # smaller array (1M)
BENCH_PARAM_num_iters=10 ./cuda_scan_large     # more timed passes
```

Environment variables (override argv):

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_size` | 16777216 | Number of uint elements to scan |
| `BENCH_PARAM_block_size` | 256 | Threads per block |
| `BENCH_PARAM_verify` | true | CPU verification (true/false/skip) |
| `BENCH_PARAM_num_warmup` | 3 | Untimed warmup passes |
| `BENCH_PARAM_num_iters` | 7 | Timed passes, each recorded individually |

## Output

**stdout** — JSON lines: per timed pass, one `{"type":"kernel",...}` record per
kernel (`scan_block_kernel`, `add_block_sums_kernel`) with that pass's
accumulated GPU event time (`time_ms`, summed over all recursion levels) and
host wall-clock time (`host_time_ms`), followed by a
`{"type":"summary","total_time_ms":...}` line whose value is the sum of all
recorded GPU event times. Derived metrics (GB/s) are computed downstream from
`time_ms` and the params.

**stderr** — human-readable results:
```
Device: NVIDIA GeForce RTX 4090 (id=0)
Scan size: 16777216  |  Iterations: 3 warmup + 7 timed

pass 0: scan_block gpu = 2.345 ms, add_block_sums gpu = 1.234 ms
...
Total GPU time: 25.123 ms over 7 passes
PASS
```

## Verification

- Full element-wise comparison against CPU exclusive prefix sum
- Exact match required (integer arithmetic, no floating-point tolerance)

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `cuda_scan_large.hip`; uses shared memory for block-level scan, recursive kernel launches for hierarchical decomposition
- **Metal**: compiled with `clang++` from `cuda_scan_large_metal.mm`; the Metal shader (`cuda_scan_large.metal`) is loaded and compiled at runtime from the same directory as the binary; recursive scan uses separate command buffer submissions per level
