# cache_latency — GPU Cache/Memory Latency Microbenchmark

## Purpose

Measures the round-trip **load latency** of the GPU memory hierarchy using
single-thread pointer chasing. A single working-set size (`array_bytes`) is
measured per execution; sweeping that size across executions traces a
latency-vs-size curve whose plateaus expose each level (L1 → L2 → last-level /
Infinity Cache → DRAM). Latency in nanoseconds per access is not printed by the
binary — it is derived downstream from each record's `time_ms` and
`num_accesses`.

## Methodology

### Pointer chasing at cacheline granularity

The benchmark allocates an array of `uint32_t` indices and wires a subset of
them into a **single random Hamiltonian cycle** (Fisher–Yates shuffle). A
single GPU thread (one block, one thread) then walks the chain:

```c
for (uint64_t i = 0; i < num_accesses; ++i) {
    idx = arr[idx];   // dependent load
}
```

The **true data dependency** (`idx = arr[idx]`) forces the hardware to fully
resolve each load before the next load's address is known. This removes all
memory-level parallelism and exposes raw latency rather than bandwidth.

Chasing is done at **cacheline granularity, not element granularity**. Only the
first word of each cacheline is used as a chase node: nodes live at element
indices `{0, stride, 2*stride, ...}` where `stride = cacheline_bytes / 4`
(16 elements for the default 64-byte line). Consecutive accesses therefore land
on **distinct cachelines** — there are no intra-line spatial hits — while the
touched lines still span the whole array, so the cache footprint equals
`array_bytes`. The number of chase nodes (and touched lines) is
`M = array_bytes / cacheline_bytes`.

Because the traversal order is a random permutation, there is no predictable
stride for hardware stream-prefetchers to exploit.

### Cold fill discarded, steady-state laps timed

An **untimed cold-fill lap** runs first (a separate kernel launch, outside the
timed bracket). It walks the chain once to make the working set resident and to
load the kernel code object, so the timed run measures steady-state hits rather
than the cold fill. This fill is capped at the largest cache (`WARMUP_CAP_BYTES
= 256 MB` worth of accesses) so the DRAM end pays only a partial fill instead of
a full, slow traversal; `warmup_accesses` in the record reports the actual
count used.

The single **timed** launch then chases `num_accesses` accesses:

```
num_accesses = clamp(measure_laps * M, 131072, 16000000)
```

with `measure_laps` steady-state laps of the `M`-node chain, floored at
`131072` (so tiny resident sizes still gather enough samples) and capped at
`16000000` (so the DRAM end stays within the harness timeout). Setting
`BENCH_PARAM_num_accesses` overrides the computed count. Resident sizes thus
measure pure hits (L1/L2/last-level cache); sizes past the caches measure
all-miss DRAM latency.

### Measurement protocol

This benchmark follows the suite-wide protocol (see the root README
"Measurement Methodology"):

- Each execution runs the timed kernel **exactly once** and times it with GPU
  events (`hipEvent`). There is no in-program timed-iteration or warm-up loop;
  the cold-fill lap above is untimed.
- Unless `BENCH_PARAM_vector_add_warmup=false` (default `true`), a vector-add
  warmer runs ~100 ms of sustained load immediately before the timed launch,
  ramping the GPU clock from idle and evicting the last-level cache, so the
  timed run is cold-cache at a boosted clock.
- Repetition is a fresh re-execution of the process: `gpubench run --reps N`.

## Files

| File                | Description                                        |
|---------------------|----------------------------------------------------|
| `cache_latency.hip` | HIP implementation (AMD ROCm + NVIDIA HIP-on-CUDA) |
| `Makefile`          | Build system with platform auto-detection          |
| `params.json`       | Parameter sweep definition for the harness         |

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
./cache_latency                 # default array_bytes = 2 MB
./cache_latency 16384           # 16 KB working set (argv override)
```

Parameters are set via `BENCH_PARAM_*` environment variables (which override
argv):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BENCH_PARAM_array_bytes` (or argv[1]) | 2097152 (2 MB) | Working-set size; selects the targeted level. Swept 4 KB → 4 GB (26 points in `params.json`). |
| `BENCH_PARAM_cacheline_bytes` | 64 | Stride between chase nodes; set to the cacheline size so each access hits a fresh line. |
| `BENCH_PARAM_measure_laps` | 4 | Timed steady-state laps (the cold-fill lap is untimed and excluded). |
| `BENCH_PARAM_num_accesses` | — | Overrides the computed timed-access count. |
| `BENCH_PARAM_rng_seed` | 42 | Seed for the chain permutation (reproducibility). |
| `BENCH_PARAM_vector_add_warmup` | true | Cold-cache clock warmer on/off (suite-wide). |

The `array_bytes` sweep is a 26-point geometric progression from 4096 bytes to
4294967296 bytes (4 GB): small sizes stay resident in L1/L2, mid sizes spill to
the last-level (MALL / Infinity) cache, large sizes reach DRAM.

## Example Output

Progress and device info go to stderr; stdout carries JSON lines — one kernel
record for the single timed run plus a final summary. Example for a 16 KB
working set (where the `131072` floor sets `num_accesses`):

```
{"type":"kernel","name":"pointer_chase_kernel","iter":0,"time_ms":12.483200,"host_time_ms":12.561004,"params":{"array_bytes":16384,"num_accesses":131072,"warmup_accesses":256,"measure_laps":4,"cacheline_bytes":64,"rng_seed":42}}
{"type":"summary","total_time_ms":12.483200}
```

- `time_ms` is the GPU event time for the timed launch; `host_time_ms` is the
  host wall-clock around launch + sync.
- `total_time_ms` is the sum of all emitted records' GPU event times (one
  record here, so they match).
- No latency figure is printed; ns/access is derived downstream as
  `time_ms * 1e6 / num_accesses`.

*(Actual values depend on GPU model, clock frequency, and cache configuration.)*

## Notes

- Only a **single thread** participates — no thread-level parallelism that could
  hide latency.
- The chase result is written to device memory so the compiler cannot eliminate
  the loop as dead code.
- No external dependencies beyond the GPU runtime (CUDA Toolkit / ROCm); the
  directory is self-contained.
