# shoc_sort — SHOC Sort Benchmark (faithful radix-sort port)

Faithful port of the SHOC level1 **Sort** benchmark. The device kernels are the
**original SHOC/CUDPP 4-bit radix sort**, taken verbatim from
`shoc/src/cuda/level1/sort/sort_kernel.cu`, and driven with the original
pipeline from `Sort.cu` (`radixSortStep` + `scanArrayRecursive`). Only the host
driver is restructured for the single-run timing harness and JSON output.

## Algorithm

A full sort of 32-bit unsigned keys is performed as **8 passes of 4 radix
bits** (`SORT_BITS = 32`). Each radix pass runs four kernels:

1. `radixSortBlocks` — block-local 4-bit sort in shared memory (CTA = 128, each
   thread handles a `uint4`).
2. `findRadixOffsets` — per-block radix counts and starting offsets (CTA = 256).
3. `scan` + `vectorAddUniform4` — recursive block-sum prefix scan of the radix
   counters (CTA = 256).
4. `reorderData` — global scatter of keys/values into sorted order (CTA = 256).

The measured run = one full sort (all 8 radix passes). Each kernel's GPU event
time is accumulated across all of its launches within the run and emitted once.

## Measurement protocol

The full sort runs **exactly once** per execution — there is no in-program
warmup or iteration loop. When `BENCH_PARAM_vector_add_warmup` is `true`
(default), the vector-add warmer runs after the (untimed) key/value upload and
immediately before the first timed launch: ~100 ms of sustained load ramps the
GPU clock while its final pass evicts the last-level cache, so the sort is
measured cold-cache at a boosted clock. Repetition is the harness's job:
`gpubench run --reps N` re-executes the binary in fresh processes.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--size N` | 4194304 | Number of key/value pairs (multiple of 1024) |

The `BENCH_PARAM_size` environment variable overrides the CLI flag, plus:

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_vector_add_warmup` | true | Clock-boost + LLC-evict warmer before the timed run |

Fixed (compile-time) CTA sizes are `SORT_BLOCK_SIZE = 128` and
`SCAN_BLOCK_SIZE = 256`, echoed into the params as `sort_block_size` /
`scan_block_size`.

## Output

- **stdout** (JSON lines): one `{"type":"kernel",...}` record for each of
  `radixSortBlocks`, `findRadixOffsets`, `scan`, `vectorAddUniform4`,
  `reorderData`, carrying that kernel's accumulated GPU event time (`time_ms`)
  and host wall-clock time (`host_time_ms`) over all of its launches in the
  run, followed by a `{"type":"summary","total_time_ms":...}` line whose value
  is the sum of the recorded GPU event times. Sort rate is derived downstream.
- **stderr**: device info, per-kernel timing, correctness check.

## Verification

After the measured run, the sorted keys are checked for non-decreasing order.
The JSON records and summary are emitted before the check, so on FAIL the
process exits non-zero without losing the measurements. There is no
verification on/off parameter.

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda # nvcc
make PLATFORM=rocm # hipcc
```

## Run

```bash
./shoc_sort
./shoc_sort --size 8388608
```

## Files

| File | Description |
|------|-------------|
| `shoc_sort.hip` | HIP/CUDA source — verbatim SHOC radix-sort kernels + timing driver |
| `Makefile` | Platform-aware build system |
| `README.md` | This file |

## Notes

- N must be a multiple of 1024 (`4 * SCAN_BLOCK_SIZE`) so every kernel's grid
  geometry divides evenly; other values are rounded down with a warning.
- Input keys are synthetic random `uint32` (SHOC uses `i % 1024`). The radix
  sort's *operation count* is data-independent, but memory locality is not:
  `reorderData`'s scatter pattern depends on the key distribution. Uniform
  random keys give representative (well-spread) scatter traffic, unlike SHOC's
  periodic `i % 1024` keys.
