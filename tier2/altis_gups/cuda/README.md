# Altis GUPS (RandomAccess) (native CUDA)

This benchmark restores the algorithm in [Altis `gups.cu`, commit
`042e2926d1fe9a5ea5b3e2d659b75bf925b152cc`](https://github.com/utcs-scea/altis/blob/042e2926d1fe9a5ea5b3e2d659b75bf925b152cc/src/cuda/level1/gups/gups.cu).
Its defining workload is a seed-1, polynomial-7 LFSR stream with exactly
`4 * table_size` logical XOR updates to a power-of-two table of 64-bit words.
Each table entry starts at its index. The address is `state & (table_size - 1)`.
GPU threads receive contiguous, nonoverlapping intervals of the same stream,
using the upstream square-table jump to find their starting states.

Both upstream atomic implementations are available:

- `cas`: a retrying 64-bit compare-and-swap loop for each logical XOR update.
- `xor`: two 32-bit atomic XOR operations, one for each half of the logical word.

The number of CAS attempts depends on contention. A logical update is not an
atomic instruction count; the XOR mode issues two atomic operations per update.
This is the pinned Altis atomic workload, not a claim of HPCC submission compliance.

The port changes host setup, launch geometry, timing and verification. It uploads
the initial table from the host instead of launching upstream's untimed initializer,
uses configurable full blocks instead of the upstream occupancy-derived grid, and
runs one selected mode per process. The BSD 2-Clause notice is retained in the source.
The previous xorshift/per-thread-update implementation used a different workload;
its measurements must be recollected for comparison with this version.

## Parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `table_size` | 1048576 | Positive power-of-two count of 64-bit entries |
| `block_size` | 256 | Threads per block, checked against device limits |
| `num_threads` | 65536 | Requested threads; rounded up to complete blocks |
| `atomic_mode` | `xor` | `cas` or `xor`, as defined above |
| `vector_add_warmup` | `true` | Run the untimed entry warmer; `false` disables it |

Parameters use `BENCH_PARAM_*` environment variables. The obsolete `num_updates`
variable is explicitly rejected: changing updates per thread would violate the
fixed `4*N` stream. Thread count changes its partition and contention, not the
logical update count. A thread may receive an empty interval for small tables.

The manifest retains the nine power-of-two sizes from 65,536 through 16,777,216,
three thread counts and both modes: **54 configurations**. Nine scaling points are
an intrinsic-domain exception to the usual density guidance. Non-power-of-two
sizes are rejected, rather than rounded into duplicate configurations.

## Timing and verification

Initial table and jump-table uploads precede the entry warmer. The selected update
kernel, including its per-thread LFSR jump, is timed once. There is one `kernel`
JSON record (`gups_cas_kernel` or `gups_xor_kernel`) followed by a `summary` with
the same GPU event time. Setup, the warmer, transfers and CPU verification are
outside that bracket. The warmer expresses the repository's intended entry
policy; actual cache and clock state still require hardware qualification.

Kernel parameters record the requested controls, `num_blocks`, `actual_threads`,
`total_updates`, `atomic_policy` (`cas64_retry` or `xor32_pair`), `stream_seed: 1`,
`lfsr_polynomial: 7`, `algorithm: "altis_lfsr4n_v1"` and
`verification: "full_table_cpu_v1"`. Derive GUPS downstream as
`total_updates / time_ms / 1e6`.

After timing, the actual entire table is downloaded. An independent serial CPU
pass generates all `4*N` LFSR updates from a pristine table; every 64-bit result
must match exactly. This replaces the old separate sampled GPU verification run
and the pinned source's incomplete identity-table check. No GPU rerun, sampling
or tolerance can hide a measured-state error. Diagnostics show at most ten
mismatches but count all of them. Verification or storage failure exits nonzero.

The reference uses one CPU worker. Two host tables require `16*N` bytes; the GPU
table requires `8*N` bytes, in addition to the shared warmer. All host table
allocation occurs before measurement. Combined resource admission remains a
separate suite-level task; allocation failure does not qualify a result.

## Build and run

```bash
make                     # -arch=native (builds for the GPU in this machine)
make CUDA_ARCH=sm_80     # A100
make CUDA_ARCH=sm_90     # H100
./altis_gups
BENCH_PARAM_table_size=2097152 BENCH_PARAM_atomic_mode=cas ./altis_gups
make clean
```

Use the harness for fresh-process repetitions. Trace/replay work remains
pending; this restoration does not establish it.

**[TODO for cuda]** The evidence in this paragraph was collected for [`../hip/`](../hip/); it has not yet been rerun for this CUDA build. The H100 qualification for this restoration covers all 54 declared configurations
and 12 additional healthy controls. The longest complete process, including the
default warmer and full serial verification, took 1.066 seconds against the
60-second collection budget. Fourteen complete tables matched the pinned CUDA
kernels byte for byte, including both atomic modes, rounded launch geometry and
the largest table. Measured-table fault injection, GPU sanitizers and native
CLI/SQLite success/failure checks also passed. These are correctness and budget
checks on H100 PCIe with CUDA 12.9, not publication performance measurements.
