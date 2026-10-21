# tier2/shoc_scan/hip — SHOC Exclusive Prefix Scan Benchmark

## Overview

The **Scan** benchmark times an **exclusive prefix sum** (scan) operation and
emits raw per-kernel times; effective GPU memory bandwidth is derived
downstream:

```
output[i] = input[0] + input[1] + … + input[i-1]   (output[0] = 0)
```

This is a classic parallel-computing primitive with read–modify–write memory
access patterns.  The implementation uses a **work-efficient (Blelloch)
two-phase scan** (up-sweep + down-sweep) with shared/threadgroup memory, and
supports partial blocks via recursive multi-block scan.  Derived from
the [SHOC benchmark suite](https://github.com/vetter/shoc).

## Provenance & fidelity

Derived from the SHOC benchmark suite (https://github.com/vetter/shoc) — the `Scan` workload. This is an independent reimplementation for portable measurement, **not** upstream source, and **not** a faithful reproduction of the upstream kernel. It differs from SHOC's version as follows:

- Recursive Blelloch scan vs SHOC's vec4 reduce/top-scan/bottom-scan.

Read this as "an exclusive-prefix-scan benchmark in the spirit of SHOC," not as SHOC's `Scan`.

## Files

| File | Description |
|------|-------------|
| `shoc_scan.hip` | Self-contained HIP kernel + host (NVIDIA/AMD) |
| `Makefile` | Auto-detects platform; supports `cuda`, `rocm` |

## Algorithm

The Blelloch work-efficient scan runs in O(n) work and O(log n) depth:

1. **Up-sweep (reduce)**: tree reduction building partial sums.
2. **Clear + Down-sweep**: zero the last element and propagate the exclusive
   prefix down the tree.

For arrays larger than a single block:
- **Phase 1** — each block performs a local Blelloch scan and saves its block
  total to `block_sums[]`.
- **Phase 2** — recursively scan `block_sums[]` to get `scanned_block_sums[]`.
- **Phase 3** — add `scanned_block_sums[i]` to every element in block `i`.

## Measurement protocol

The full multi-level scan runs **exactly once** per execution — there is no
in-program warmup or iteration loop. When `BENCH_PARAM_vector_add_warmup` is
`true` (default), the vector-add warmer runs immediately before the timed run:
~100 ms of sustained load ramps the GPU clock while its final pass evicts the
last-level cache, so the scan is measured cold-cache at a boosted clock.
Repetition is the harness's job: `gpubench run --reps N` re-executes the binary
in fresh processes.

## Building

```bash
# Auto-detect (/opt/rocm present → rocm, else → cuda)
make

# Explicit platform
make PLATFORM=rocm
make PLATFORM=cuda
```

**Requirements:**
- **CUDA**: `nvcc` (CUDA toolkit ≥ 10)
- **ROCm/HIP**: `hipcc` (ROCm ≥ 4.0)

## Running

```bash
./shoc_scan [--array_size N] [--block_size B]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--array_size N` | `1048576` | Number of `float` elements (1 M ≈ 4 MiB) |
| `--block_size B` | `256` | Threads per block — elements per block = 2× |

Environment variables override the flags:

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_array_size` | `1048576` | Number of elements |
| `BENCH_PARAM_block_size` | `256` | Threads per block |
| `BENCH_PARAM_vector_add_warmup` | `true` | Clock-boost + LLC-evict warmer before the timed run |

Resolved `array_size` and `block_size` must be positive. The existing clamp of
positive block sizes above 512 to 512 is preserved; after that clamp, the block
size must be a power of two for the Blelloch tree. Padding to the next multiple
of `2 * block_size` must fit in `int` (`INT_MAX`), as required by the kernel's
signed indices. Invalid geometry exits with a diagnostic before device access
or allocation. Padding and recursive block counts use wide host arithmetic.
All 32 declared sizes and the default block size remain admitted. These bounds
establish index representability, not available host/device memory or runtime.
Existing CLI fallback and environment precedence are unchanged.

## Output

**stdout** — JSON lines: one `kernel` record per kernel
(`scan_block_kernel`, `add_block_sums_kernel`) with that kernel's GPU event
time accumulated across all recursion levels of the run, then a `summary`
whose `total_time_ms` is the sum of the recorded GPU event times:
```json
{"type":"kernel","name":"scan_block_kernel","iter":0,"time_ms":0.185000,"host_time_ms":0.210000,"params":{"array_size":1048576,"block_size":256,"verification":"full_cpu_fp64accum_v1"}}
{"type":"kernel","name":"add_block_sums_kernel","iter":0,"time_ms":0.030000,"host_time_ms":0.045000,"params":{"array_size":1048576,"block_size":256,"verification":"full_cpu_fp64accum_v1"}}
{"type":"summary","total_time_ms":0.215000}
```

Derived metrics (e.g. GB/s) are computed downstream from `time_ms` and params.

**stderr** — human-readable summary:
```
Device: NVIDIA GeForce RTX 4090 (id=0)
Array size: 1048576 floats (4.0 MiB, padded to 1048576)
Block size: 256

scan gpu = 0.185 ms, add gpu = 0.030 ms
Total GPU time: 0.215 ms
PASS
```

## Verification

After the measured run (all JSON records and the summary already emitted), the
output left by that run is downloaded and compared element-wise against a CPU
exclusive-scan reference. Verification does not launch another GPU scan or
another clock warmer.
On FAIL the process exits non-zero, so the harness marks the run as failed
without losing the measurements. There is no verification on/off parameter.

The CPU reference accumulates the actual input in `double`, then stores each
exclusive prefix as `float`. Generated inputs are integers from 1 to 7, so every
prefix is exact in the double accumulator over the positive `int` size domain.
The previous serial float accumulator drifted and eventually saturated, falsely
rejecting large declared arrays. GPU arithmetic and the acceptance rule remain
unchanged: both compared values must be finite, and the absolute difference
must not exceed `1e-3 * abs(reference) + 1e-5`. Every output is checked; all errors
are counted with at most ten details. `verification: full_cpu_fp64accum_v1`
identifies the corrected reference. No extra arrays or GPU launches are added.

`python3 tests/shoc_scan_reference.py --host-only --compiler clang++ --work-dir /tmp/shoc-scan-host`
(from the repository root) compares the actual CPU reference against independent
integer-prefix formulas, known answers, input immutability and buffer guards,
including sizes where float accumulation loses precision. Use `--platform cuda
--sweep --work-dir /tmp/shoc-scan-gpu` for every declared size with both warmer
settings and actual-device first/middle/tail corruption, nonfinite values,
zeroed outputs and bounded error diagnostics. Each process has a 60-second
budget; raw logs and source/binary hashes remain in the new work directory.
CI runs host checks and compiles GPU variants; GPU execution requires hardware.
`python3 tests/shoc_scan_input_ranges.py --compiler clang++ --work-dir /tmp/shoc-scan-inputs`
checks actual parsing/padding/recursive expressions under fatal host sanitizers
without allocation. Add `--binary /path/to/shoc_scan` to require exact invalid
geometry diagnostics with GPUs hidden. Capacity admission, general parsing
policy and AMD hardware qualification remain separate.

## Bandwidth Formula

Derived downstream from the emitted times:

```
Bandwidth (GB/s) = 2 × N × sizeof(float) / time_seconds / 1e9
```

The factor of 2 accounts for one read (input) and one write (output).
