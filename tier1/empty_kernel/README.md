# empty_kernel — GPU Launch-Overhead Microbenchmark

## Purpose

Measures the cost of launching a single **empty kernel** — a kernel that does
no work — so the measured time is pure launch / dispatch / grid-scheduling
overhead with no compute or memory traffic mixed in. Sweeping `num_blocks` (and
`block_size` as a non-scaling axis) shows how that overhead grows with grid
geometry.

Each timed run records two times:

- **`host_time_ms`** — host wall-clock around launch + synchronize: the full
  launch + sync API round trip as the host sees it. This is the closer proxy
  for real launch overhead, since it includes the driver/runtime submission
  path.
- **`time_ms`** — GPU event time bracketing the launch: the device-side
  dispatch / grid-scheduling / retire cost measured on the GPU timeline.

## Methodology

This benchmark follows the suite-wide protocol (see the root README
"Measurement Methodology"):

- Each execution launches the empty kernel **exactly once** and times it with
  GPU events (`hipEvent`) plus a host chrono bracket. There is no in-program
  timed-iteration or warm-up loop.
- Unless `BENCH_PARAM_vector_add_warmup=false` (default `true`), a vector-add
  warmer runs ~100 ms of sustained load immediately before the timed launch,
  ramping the GPU clock from idle (and evicting the last-level cache), so the
  launch is timed at a boosted clock.
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
record for the single timed launch plus a final summary:

```
{"type":"kernel","name":"empty_kernel","iter":0,"time_ms":0.004320,"host_time_ms":0.021500,"params":{"num_blocks":1024,"block_size":256}}
{"type":"summary","total_time_ms":0.004320}
```

- `time_ms` is the GPU event time for the launch; `host_time_ms` is the host
  wall-clock around launch + sync (the closer proxy for launch overhead).
- `total_time_ms` is the sum of all emitted records' GPU event times (one
  record here, so they match).

*(Actual values depend on GPU model, driver, and grid geometry.)*

## Notes

- The kernel body is empty, so any nonzero `time_ms` is scheduling/dispatch
  overhead rather than compute.
- No external dependencies beyond the GPU runtime (CUDA Toolkit / ROCm); the
  directory is self-contained.
