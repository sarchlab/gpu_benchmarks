# empty_kernel — Empty-Dispatch Timing Microbenchmark

## Purpose

Measures an empty kernel over a selected grid. Sweeping `num_blocks` and
`block_size` shows how the observed timing interval changes with grid geometry.

Each timed run records two times:

- **`time_ms`** — GPU event interval around one empty dispatch, including grid
  completion and any delay between the recorded events.
- **`host_time_ms`** — host bracket spanning event recording, launch submission,
  error checking, stop-event recording, and synchronization.

Neither is an isolated host launch-call cost. Short GPU intervals can approach
the event timer's resolution; more printed digits or repetitions do not improve
that resolution. See [measurement interpretation](../../docs/measurement_interpretation.md#empty-dispatch-and-timing-resolution)
for timer qualification and handling zero or near-resolution samples.

## Methodology

This benchmark follows the suite-wide protocol (see the root README
"Measurement Methodology"):

- Each execution launches the empty kernel **exactly once** and times it with
  GPU events (`hipEvent`) plus a host chrono bracket. There is no in-program
  timed-iteration or warm-up loop.
- Unless `BENCH_PARAM_vector_add_warmup=false` (default `true`), a vector-add
  warmer runs ~100 ms of sustained load immediately before the timed launch,
  intended to raise clocks and disturb cache state. Qualify the resulting
  clock state on the target GPU.
- Repetition is a fresh re-execution of the process: `gpubench run --reps N`.

## Files

| File               | Description                                        |
|--------------------|----------------------------------------------------|
| `empty_kernel.hip` | HIP implementation (AMD ROCm + NVIDIA HIP-on-CUDA) |
| `Makefile`         | Build system with platform auto-detection          |
| `params.json`      | Parameter sweep definition for the harness         |

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
./empty_kernel               # default 1024 blocks x 256 threads
./empty_kernel 4096 128      # num_blocks=4096, block_size=128 (argv override)
```

Parameters are set via `BENCH_PARAM_*` environment variables (which override
argv):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BENCH_PARAM_num_blocks` (or argv[1]) | 1024 | Thread blocks in the empty grid; primary overhead axis. Swept 1 → 2097152 in `params.json`. |
| `BENCH_PARAM_block_size` (or argv[2]) | 256 | Threads per block. Swept 32 / 64 / 256 / 1024 in `params.json`. |
| `BENCH_PARAM_vector_add_warmup` | true | Cold-cache clock warmer on/off (suite-wide). |

## Example Output

Progress and device info go to stderr; stdout carries JSON lines — one kernel
record for the single timed launch plus a final summary. Illustrative timings:

```
{"type":"kernel","name":"empty_kernel","iter":0,"time_ms":0.004320,"host_time_ms":0.021500,"params":{"num_blocks":1024,"block_size":256}}
{"type":"summary","total_time_ms":0.004320}
```

- `time_ms` is the GPU event time for the launch; `host_time_ms` is the host
  bracket described above.
- `total_time_ms` is the sum of all emitted records' GPU event times (one
  record here, so they match).

*(Actual values depend on GPU model, driver, and grid geometry.)*

## Notes

- The empty kernel isolates a grid with no body work; event timing still
  includes scheduling effects and timer uncertainty.
- No external dependencies beyond the GPU runtime (CUDA Toolkit / ROCm); the
  directory is self-contained.
