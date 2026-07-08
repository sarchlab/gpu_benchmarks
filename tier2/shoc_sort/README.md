# shoc_sort — SHOC Sort Benchmark (faithful radix-sort port)

Faithful port of the SHOC level1 **Sort** benchmark. The device kernels are the
**original SHOC/CUDPP 4-bit radix sort**, taken verbatim from
`shoc/src/cuda/level1/sort/sort_kernel.cu`, and driven with the original
pipeline from `Sort.cu` (`radixSortStep` + `scanArrayRecursive`). Only the host
driver is restructured for the per-iteration timing harness and JSON output.

## Algorithm

A full sort of 32-bit unsigned keys is performed as **8 passes of 4 radix
bits** (`SORT_BITS = 32`). Each radix pass runs four kernels:

1. `radixSortBlocks` — block-local 4-bit sort in shared memory (CTA = 128, each
   thread handles a `uint4`).
2. `findRadixOffsets` — per-block radix counts and starting offsets (CTA = 256).
3. `scan` + `vectorAddUniform4` — recursive block-sum prefix scan of the radix
   counters (CTA = 256).
4. `reorderData` — global scatter of keys/values into sorted order (CTA = 256).

One recorded pass = one full sort (all 8 radix passes). Each kernel's GPU event
time is accumulated across all of its launches within the pass and emitted once.

## Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--size N` | 4194304 | Number of key/value pairs (multiple of 1024) |
| `--precision P` | uint | Key data-type label (uint/int/float; keys always sorted as 32-bit unsigned) |

Fixed (compile-time) CTA sizes are `SORT_BLOCK_SIZE = 128` and
`SCAN_BLOCK_SIZE = 256`, echoed into the params as `sort_block_size` /
`scan_block_size`.

Environment variables (`BENCH_PARAM_size`, `BENCH_PARAM_precision`) override the
CLI flags, plus:

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_num_warmup` | 3 | Untimed warmup passes |
| `BENCH_PARAM_num_iters` | 7 | Timed passes, each recorded individually |

## Output

- **stdout** (JSON lines): per timed pass, one `{"type":"kernel",...}` record
  for each of `radixSortBlocks`, `findRadixOffsets`, `scan`,
  `vectorAddUniform4`, `reorderData`, carrying that kernel's accumulated GPU
  event time (`time_ms`) and host wall-clock time (`host_time_ms`), followed by
  a `{"type":"summary","total_time_ms":...}` line whose value is the sum of all
  recorded GPU event times. Sort rate is derived downstream.
- **stderr**: device info, per-pass per-kernel timing, correctness check.

## Data reset

The radix sort is **destructive** and returns its result in `dKeys`/`dVals`
(each `radixSortStep` ping-pongs keys→tempKeys→keys). The pristine keys and
values are therefore re-uploaded (untimed) at the start of every pass so each
pass performs an identical sort of random data.

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda # nvcc
make PLATFORM=rocm # hipcc
make PLATFORM=metal
```

## Run

```bash
./shoc_sort
BENCH_PARAM_num_iters=10 ./shoc_sort --size 8388608
```

## Files

| File | Description |
|------|-------------|
| `shoc_sort.hip` | HIP/CUDA source — verbatim SHOC radix-sort kernels + timing driver |
| `shoc_sort.metal` | Metal compute shader (legacy bitonic implementation) |
| `shoc_sort_metal.mm` | Objective-C++ Metal host code (Apple) |
| `Makefile` | Platform-aware build system |
| `README.md` | This file |

## Notes

- N must be a multiple of 1024 (`4 * SCAN_BLOCK_SIZE`) so every kernel's grid
  geometry divides evenly; other values are rounded down with a warning.
- Input keys are synthetic random `uint32` (SHOC uses `i % 1024`); radix-sort
  work is data-independent, so timing is unaffected.
