# tier2/rodinia_nw — Rodinia Needleman-Wunsch Benchmark

Dynamic programming sequence alignment using the Needleman-Wunsch algorithm.
The scoring matrix is filled with block-based anti-diagonal wavefront parallelism
and threadgroup/shared-memory tiling, based on the Rodinia NW benchmark.

## Algorithm

- Sequences of length **N** over an alphabet of size 10 (synthetic, reproducible).
- DP matrix is **(N+1) × (N+1)**; borders initialised on CPU.
- Scoring: **match = +1**, **mismatch = −1**, **gap = −penalty** (default 2).
- Two GPU kernel passes process the block-dependency graph's upper-left and
  lower-right triangles respectively; within each block a shared-memory
  anti-diagonal sweep fills a `block_size × block_size` tile.
- Emits per-iteration kernel times; **GCUPS** (giga cell-updates per second)
  and effective **GB/s** are derived downstream.

## Files

| File | Description |
|------|-------------|
| `rodinia_nw.hip` | HIP/CUDA source (self-contained, no external headers) |
| `rodinia_nw.metal` | Metal compute shader (two kernels) |
| `rodinia_nw_metal.mm` | ObjC++ Metal host |
| `Makefile` | Platform auto-detection + build rules |
| `README.md` | This file |

## Build

```bash
# Auto-detect (Darwin → metal, /opt/rocm present → rocm, else → cuda)
make

# Explicit platform
make PLATFORM=cuda
make PLATFORM=rocm
make PLATFORM=metal
```

## Run

```bash
./rodinia_nw                              # defaults: N=2048, block=16, penalty=2
./rodinia_nw --sequence_length 1024
./rodinia_nw --sequence_length 4096 --block_size 32 --penalty 10
```

Environment variables (override argv):

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_sequence_length` | 2048 | Sequence length N |
| `BENCH_PARAM_block_size` | 16 | Threads per block / tile edge |
| `BENCH_PARAM_penalty` | 2 | Gap penalty |
| `BENCH_PARAM_warmup` | 3 | Untimed warmup passes |
| `BENCH_PARAM_iteration` | 7 | Timed passes, each recorded individually |

## Output

**stdout** — JSON lines: per timed pass, one `{"type":"kernel",...}` record per
kernel (`nw_kernel1`, `nw_kernel2`) with that pass's accumulated GPU event time
(`time_ms`, summed over the wavefront diagonals) and host wall-clock time
(`host_time_ms`), followed by a `{"type":"summary","total_time_ms":...}` line
whose value is the sum of all recorded GPU event times. Derived metrics
(GCUPS, GB/s) are computed downstream from `time_ms` and the params.

Standard error shows per-pass timing lines.

## Platforms

| Platform | Compiler | Source file |
|----------|----------|-------------|
| CUDA     | `nvcc`   | `rodinia_nw.hip` (compiled as CUDA via `-x cu`) |
| ROCm     | `hipcc`  | `rodinia_nw.hip` |
| Metal    | `clang++`| `rodinia_nw_metal.mm` + `rodinia_nw.metal` |
