# shoc_sort — SHOC Sort Benchmark (faithful radix-sort port)

Faithful port of the SHOC level1 **Sort** benchmark. The device kernels are the
**original SHOC/CUDPP 4-bit radix sort**, taken verbatim from
`shoc/src/cuda/level1/sort/sort_kernel.cu`, and driven with the original
pipeline from `Sort.cu` (`radixSortStep` + `scanArrayRecursive`). Only the host
driver is restructured for the single-run timing harness and JSON output.

## Provenance

Derived from the SHOC benchmark suite (https://github.com/vetter/shoc) — the `Sort` workload. Independent reimplementation, spot-verified to reproduce SHOC's algorithm and constants faithfully: the CUDPP 4-bit radix sort is reproduced verbatim (from `shoc/src/cuda/level1/sort/sort_kernel.cu`), including its quirks. The only deviation is the synthetic key distribution (uniform random `uint32` vs SHOC's periodic `i % 1024`), noted under "Notes" below; the sort's operation count is data-independent.

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

The resolved size must be positive; rejection happens before device access.
Buffer byte counts use `size_t`, avoiding the former 32-bit wrap on supported
64-bit hosts. This does not guarantee available memory or qualify arbitrary
manual sizes within the collection time budget. Positive CLI fallback and
environment precedence are preserved.

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

After the measured run, both actual output arrays are downloaded and every
key/value pair is checked. Keys must be non-decreasing; values must be valid
source indices, with each index appearing exactly once; and each key must equal
the pristine input key at its source index. This proves exact conservation of
all input pairs without imposing an ordering on equal keys. The GPU sort is not
rerun for verification.

All invalid output positions are counted (one error per position even when
several checks fail), with at most ten detailed messages. A mismatch or an
unavailable host verification buffer returns nonzero. Kernel JSON records are
emitted before verification, and the summary follows it, including on these
failures, so the harness retains timings while marking the run as failed.
Records include `verification: full_pairs_v1`. There is no verification switch.

Verification takes O(N) host work. It preserves the original keys, reuses the
existing host values buffer for the measured indices, and adds one output-key
buffer plus one byte per source index: 20 MiB extra at the largest declared
4,194,304-pair size. GPU allocations and all eight radix passes are unchanged.
Available-memory/time admission for manual sizes and AMD hardware qualification
remain separate.

From the repository root, run
`python3 tests/shoc_sort_verification.py --host-only --compiler clang++ --work-dir /tmp/sort-host`
for independent pair-multiset/order fixtures, complete error counts, diagnostic
caps, allocation failure and memory guards under fatal host sanitizers. Use
`--platform cuda --sweep --work-dir /tmp/sort-gpu` for actual measured-device key,
value, duplicate-pair and association faults, skipped work, unavailable buffers,
and all 13 declared sizes with both warmer settings. Each GPU process has a
60-second limit; use a new work directory to retain logs, hashes and outcomes.
CI executes host checks and compiles GPU variants; GPU execution needs hardware.

`python3 tests/shoc_sort_input_ranges.py --compiler clang++ --work-dir /tmp/sort-input`
checks the production parser, positivity guard, rounding and byte arithmetic
without allocating large arrays. Add `--binary /path/to/shoc_sort` to require
exact invalid-input errors from the actual binary with GPUs hidden. These host
checks include the former 4 GiB wrap boundary and `INT_MAX`; they do not execute
or qualify those large sorts.

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

- Positive N is rounded down to a multiple of 1024 (`4 * SCAN_BLOCK_SIZE`),
  with a minimum of 1024, so every kernel's grid geometry divides evenly.
  Nonmultiples produce a warning.
- Input keys are synthetic random `uint32` (SHOC uses `i % 1024`). The radix
  sort's *operation count* is data-independent, but memory locality is not:
  `reorderData`'s scatter pattern depends on the key distribution. Uniform
  random keys give representative (well-spread) scatter traffic, unlike SHOC's
  periodic `i % 1024` keys.
