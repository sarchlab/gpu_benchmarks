# cache_bandwidth — Per-Cache-Level Read Bandwidth Microbenchmark

## Purpose

Measures GPU **read bandwidth** at each level of the memory hierarchy. Many
streaming threads repeatedly read a resident window of `working_set_bytes`;
sweeping the window size across executions traces a bandwidth-vs-size curve that
plateaus per level (L1 → L2 → last-level / Infinity Cache → DRAM), with knees at
each cache's capacity. Bandwidth in GB/s is not printed by the binary — it is
derived downstream from the recorded time and the byte-accounting params.

## Methodology

### Streaming reads with re-reads for steady state

The kernel `bw_read_kernel` has each thread block stream **vectorized `float4`
reads** across its window, and repeats that pass `iters` times:

```c
for (uint32_t it = 0; it < iters; ++it)
    for (uint32_t i = threadIdx.x; i < vecs_per_window; i += blockDim.x)
        acc += win[i];
```

Re-reading the same window `iters` times keeps it resident in whichever cache
level holds it, so cache-resident sizes are measured at **steady state** (the
first pass warms the level; the remaining passes read from it) rather than
paying a one-time cold fill. `iters` is auto-tuned so each launch moves roughly
`TARGET_TRAFFIC_BYTES` (8 GB), clamped to `[3, 4000000]`. The accumulator is
written to a sink buffer so the reads are not optimized away.

### Shared vs. private mode

- **`mode = "shared"`** (default): every block reads the **same** single window
  (one shared buffer, `windows = 1`). This measures the bandwidth a resident
  buffer delivers to the whole device (the "global" / device-wide view). Small
  windows are effectively replicated across each unit's private cache; larger
  windows are served by the shared levels. Grid: `SHARED_BLOCKS_PER_CU = 8`
  blocks/CU × `SHARED_THREADS_PER_BLOCK = 256` threads.

- **`mode = "private"`**: blocks read **distinct** sub-buffers to measure
  per-unit bandwidth without cross-unit sharing (report `GB/s / num_blocks` for
  the per-unit figure). Grid: `PRIVATE_BLOCKS_PER_CU = 1` block/CU ×
  `PRIVATE_THREADS_PER_BLOCK = 1024` threads. Block `b` reads window
  `b % windows`, where

  ```
  windows = min(num_blocks, cap / working_set_bytes)
  cap     = min(free_memory/2, PRIVATE_ALLOC_CAP = 4 GB)
  ```

  Because the allocation is capped, `windows` can be **fewer than `num_blocks`**
  for large working sets: once `working_set_bytes` exceeds roughly
  `4 GB / num_blocks`, `windows < num_blocks` and multiple blocks map onto the
  same window (i.e. they share). On a device with a few hundred CUs this
  crossover lands around **16 MB** — so private-mode working sets at or above
  ~16 MB no longer give every block a fully private window. The record's
  `num_blocks` and `iters` (below) always reflect the real work, so downstream
  byte accounting stays correct.

### Byte accounting

Total bytes read by a launch is:

```
bytes = num_blocks * working_set_bytes * iters
```

(every one of the `num_blocks` blocks reads its full `working_set_bytes` window
`iters` times, in both modes). Bandwidth is then `bytes / time_ms` downstream —
all three factors are emitted in the record params.

### Measurement protocol

This benchmark follows the suite-wide protocol (see the root README
"Measurement Methodology"):

- Each execution runs the kernel **exactly once** and times it with GPU events
  (`hipEvent`). There is no in-program timed-iteration or warm-up loop; `iters`
  is the *inner re-read count within the single timed launch*, not a repeated
  measurement.
- Unless `BENCH_PARAM_vector_add_warmup=false` (default `true`), a vector-add
  warmer runs ~100 ms of sustained load immediately before the timed launch,
  ramping the GPU clock from idle and evicting the last-level cache, so the run
  is cold-cache at a boosted clock.
- Repetition is a fresh re-execution of the process: `gpubench run --reps N`.

## Files

| File                  | Description                                        |
|-----------------------|----------------------------------------------------|
| `cache_bandwidth.hip` | HIP implementation (AMD ROCm + NVIDIA HIP-on-CUDA) |
| `Makefile`            | Build system with platform auto-detection          |
| `params.json`         | Parameter sweep definition for the harness         |

## Building

The Makefile auto-detects the platform (Linux with `/opt/rocm` → ROCm/HIP;
otherwise → CUDA):

```bash
make                  # auto-detect
make PLATFORM=cuda
make PLATFORM=rocm
make clean
```

## Running

```bash
./cache_bandwidth                                  # default 1 MB shared
./cache_bandwidth 262144                           # 256 KB working set (argv override)
BENCH_PARAM_mode=private ./cache_bandwidth 262144  # per-unit view
```

Parameters are set via `BENCH_PARAM_*` environment variables (which override
argv):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BENCH_PARAM_working_set_bytes` (or argv[1]) | 1048576 (1 MB) | Resident window size; selects the targeted level. Must be ≥ 16 and a multiple of 16. Swept 1 KB → 1 GB (21 points in `params.json`). |
| `BENCH_PARAM_mode` | shared | `shared` (all blocks read one buffer, global bandwidth) or `private` (each block its own buffer, per-unit bandwidth). |
| `BENCH_PARAM_vector_add_warmup` | true | Cold-cache clock warmer on/off (suite-wide). |

`iters`, `num_blocks`, and `blocks_per_cu` are computed at run time (from the
device CU count and memory budget) and reported in the output record rather than
being primary sweep knobs.

## Example Output

Progress and device info go to stderr; stdout carries JSON lines — one kernel
record for the single timed launch plus a final summary:

```
{"type":"kernel","name":"bw_read_kernel","iter":0,"time_ms":1.204800,"host_time_ms":1.281003,"params":{"working_set_bytes":1048576,"mode":"shared","blocks_per_cu":8,"num_blocks":2432,"iters":3}}
{"type":"summary","total_time_ms":1.204800}
```

- `time_ms` is the GPU event time for the launch; `host_time_ms` is the host
  wall-clock around launch + sync.
- `total_time_ms` is the sum of all emitted records' GPU event times (one
  record here, so they match).
- GB/s is not printed; it is derived downstream as
  `num_blocks * working_set_bytes * iters / time_ms`.

If a working set exceeds the memory budget, the binary emits a
`{"type":"error",...}` line instead of a kernel record and exits non-zero.

*(Actual values depend on GPU model, clock frequency, cache sizes, and CU
count.)*

## Notes

- Reads are vectorized (`float4`) to reach the level's bandwidth ceiling with
  fewer instructions.
- No external dependencies beyond the GPU runtime (CUDA Toolkit / ROCm); the
  directory is self-contained.
