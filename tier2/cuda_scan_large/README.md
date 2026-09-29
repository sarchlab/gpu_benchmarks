# tier2/cuda_scan_large — Work-Efficient Parallel Prefix Sum (Blelloch Scan)

Benchmarks a **work-efficient parallel exclusive prefix sum** (Blelloch scan) on a large array of unsigned integers. Emits one kernel-time record per kernel; GB/s is derived downstream.

## Provenance & fidelity

Derived from the [CUDA SDK samples](https://github.com/NVIDIA/cuda-samples) — the `scanLargeArray` sample. Independent reimplementation for portable measurement, **not** upstream source, and **not** a faithful reproduction. It differs as follows:

- Omits the `CONFLICT_FREE_OFFSET` bank-conflict padding (correct results, but different shared-memory bank-conflict behavior) vs the CUDA SDK scan sample.

Read this as "a large-array scan benchmark in the spirit of the CUDA SDK sample," not the `scanLargeArray` sample itself.

## Algorithm

- **N** unsigned integers (default: N=16,777,216 = 16M)
- Three-phase hierarchical Blelloch scan:
  1. **Block-level scan** — each block of 512 elements performs an in-place exclusive scan using shared memory (up-sweep + down-sweep), stores block total
  2. **Scan of block sums** — recursively scan the array of block totals
  3. **Add block sums back** — add each scanned block total to all elements in the corresponding block
- Handles multiple blocks via recursive decomposition
- GB/s = 2 · N · sizeof(uint) / time\_sec / 10⁹ (derived downstream from the emitted times and params)

## Measurement

The full hierarchical scan runs **exactly once** per process execution (see
the root README "Measurement Methodology"). With
`BENCH_PARAM_vector_add_warmup` (default `true`), a vector-add warmer boosts
the GPU clock and evicts the last-level cache immediately before the timed
run. Repetition = `gpubench run --reps N`.

## Parameters

| Variable (or `--flag`) | Description | Default |
|----------|-------------|---------|
| `BENCH_PARAM_size` / `--size` | Number of uint elements to scan | 16777216 |
| `BENCH_PARAM_vector_add_warmup` | Cold-cache clock warmer on/off | true |

`BENCH_PARAM_verify=false|0|skip` skips CPU verification (undeclared debug
knob, not in `params.json` and never swept).

The resolved size must be positive; rejection happens before allocation or
device access, including when debug verification is disabled. Recursive block
counts use wide host arithmetic, avoiding signed overflow near `INT_MAX`.
Positive sizes representable as `int` retain their existing index geometry;
on supported 64-bit hosts, buffer byte counts also remain representable.
This does not guarantee available memory or qualify arbitrary manual sizes
within the collection time budget. Existing positive CLI fallback and
environment precedence are preserved.

## Files

| File | Description |
|------|-------------|
| `cuda_scan_large.hip` | HIP source — block scan + add-back kernels with recursive dispatch |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./cuda_scan_large                              # defaults: N=16777216
./cuda_scan_large --size 1048576               # smaller array (1M)
BENCH_PARAM_size=1048576 ./cuda_scan_large     # env overrides argv
```

Each kernel record includes the JSON boolean `verification_enabled`, resolved
from the existing debug flag. Verification is enabled by default; only the
case-sensitive values `false`, `0`, and `skip` disable it. An inherited harness
environment can set this unswept control. The boolean reports whether checks
are enabled, not whether they passed: verification failures still exit nonzero.
A successful process with `verification_enabled: false` is unchecked and must
not be treated as verified collection evidence. Older records missing the field
have unknown verification status.

## Output

**stdout** — JSON lines: one `{"type":"kernel",...}` record (`iter:0`) per
kernel (`scan_block_kernel`, `add_block_sums_kernel`) with that kernel's
GPU event time (`time_ms`, summed over all recursion levels of the single
run) and host wall-clock time (`host_time_ms`), followed by a
`{"type":"summary","total_time_ms":...}` line whose value is the sum of all
recorded GPU event times. Derived metrics (GB/s) are computed downstream from
`time_ms` and the params.

**stderr** — human-readable results:
```
Device: NVIDIA GeForce RTX 4090 (id=0)
Scan size: 16777216

scan_block gpu = 2.345 ms, add_block_sums gpu = 1.234 ms
Total GPU time: 3.579 ms
PASS
```

## Verification

- Full element-wise comparison against CPU exclusive prefix sum
- Exact match required (integer arithmetic, no floating-point tolerance)
- On FAIL the process exits non-zero (skipped when `BENCH_PARAM_verify=false`)

`python3 tests/scan_input_ranges.py --compiler clang++ --work-dir /tmp/scan-input`
(from the repository root) checks the actual parser, positivity guard, buffer
arithmetic and recursive block-count expression under fatal host sanitizers.
It covers all declared sizes and integer boundaries without allocating large
arrays. Add `--binary /path/to/cuda_scan_large` to require exact invalid-input
errors from the actual binary with GPUs hidden; CI runs this path too.

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `cuda_scan_large.hip`; uses shared memory for block-level scan, recursive kernel launches for hierarchical decomposition
