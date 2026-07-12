# altis_gups — GUPS (Giga Updates Per Second)

## Provenance & fidelity

Derived from the [Altis benchmark suite](https://github.com/utcs-scea/altis) — the `gups` (HPCC RandomAccess) workload. Independent reimplementation for portable measurement, **not** upstream source, and **not** a faithful reproduction. It differs as follows:

- Uses `atomicXor` vs HPCC RandomAccess's racy (non-atomic) stores; this measures the atomic pipeline rather than the specified racy-update semantics. (Disclosed.)

Read this as "a random-access benchmark in the spirit of Altis/HPCC," not the HPCC RandomAccess spec.

## Algorithm

Measures random memory access throughput on GPU by performing random
read-modify-write operations on a large table of 64-bit unsigned integers.
Based on the HPCC RandomAccess (GUPS) benchmark from the Altis suite.

Each thread uses an xorshift64 PRNG to generate random table indices and
performs atomic XOR updates:

```
for each update:
    state = xorshift64(state)
    table[state % table_size] ^= state
```

This benchmark stresses the memory subsystem with random, irregular access
patterns that defeat caching and prefetching mechanisms.

## Parameters

| Parameter       | Default   | Description                          |
|----------------|-----------|--------------------------------------|
| `table_size`    | 1048576   | Number of 64-bit entries in table    |
| `block_size`    | 256       | Threads per block                    |
| `num_updates`   | 128       | Random updates per thread            |
| `num_threads`   | 65536     | Total threads launched               |
| `vector_add_warmup` | true  | Cold-cache calibration: boost the clock and evict the LLC before the timed launch; set `false` to time as-is |

Parameters are read from `BENCH_PARAM_*` environment variables.
The kernel runs exactly once per execution; the pristine table is uploaded
before the timed launch (untimed). Repetition across fresh processes is the
harness's job (`gpubench run --reps N`).

## Build & Run

```bash
# Auto-detect platform
make
./altis_gups

# Explicit platform
make PLATFORM=cuda
make PLATFORM=rocm

# Custom parameters
BENCH_PARAM_table_size=2097152 BENCH_PARAM_num_updates=256 ./altis_gups
```

## Output Format

**JSON-lines to stdout** — one `kernel` record for the single timed launch,
then a `summary` whose `total_time_ms` equals that launch's GPU event time:
```json
{"type":"kernel","name":"gups_kernel","iter":0,"time_ms":12.345000,"host_time_ms":12.501000,"params":{"table_size":1048576,...}}
{"type":"summary","total_time_ms":12.345000}
```

**Human-readable to stderr:**
```
Device: AMD Radeon RX 7900 XTX (id=0)
GUPS benchmark  |  Table size: 1048576 (8.39 MB)  |  Updates/thread: 128  |  ...

gups_kernel: gpu = 12.345 ms, host = 12.501 ms
Total GPU time: 12.345 ms
PASS
```

## Metric

**GUPS** (Giga Updates Per Second) = `total_updates / time_seconds / 1e9`

where `total_updates = actual_threads × num_updates`. Derived downstream
from the emitted `time_ms` and params; not reported by the binary.

## Verification

Compares GPU table state against a sequential CPU reference implementation
for the first 1024 entries. Because atomic XOR is used, results are
deterministic and must match exactly. On FAIL the binary exits with a
non-zero status.
