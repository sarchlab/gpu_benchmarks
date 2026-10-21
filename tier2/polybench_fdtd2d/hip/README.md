# polybench_fdtd2d — PolyBench FDTD-2D Benchmark

2D Finite Difference Time Domain (FDTD) electromagnetic simulation implementing the canonical PolyBench FDTD-2D recurrence. Measures effective memory bandwidth of iterative stencil updates on three 2D field arrays.

## Provenance & fidelity

Derived from the [PolyBench/GPU benchmark suite](https://sourceforge.net/projects/polybench/) — the `fdtd-2d` workload. Independent reimplementation for portable measurement, **not** upstream source, but it reproduces the canonical PolyBench FDTD-2D recurrence: `ey` carries the `i`-difference plus the `_fict_[t] = t` time-source row, `ex` carries the `j`-difference, both E-field updates subtract, and `hz -= 0.7·(∂ex + ∂ey)`. (An earlier version had the `ex`/`ey` difference axes swapped, added instead of subtracted, and dropped the source; that has been corrected.)

## Algorithm

Three field arrays `ex[NX][NY]`, `ey[NX][NY]`, `hz[NX][NY]` are updated each time step. The equations actually computed are:

```
ex[i][j] -= 0.5*(hz[i][j] - hz[i][j-1])       for all i, j >= 1
ex[i][0] is unchanged

ey[0][j]  = t                                for all j
ey[i][j] -= 0.5*(hz[i][j] - hz[i-1][j])       for i >= 1, all j

hz[i][j] -= 0.7*(ex[i][j+1] - ex[i][j] +
                 ey[i+1][j] - ey[i][j])       for i < NX-1, j < NY-1
hz's final row and final column are unchanged
```

Each step uses the current `hz` to update `ex` and `ey`, then uses both
updated electric fields to update `hz`. The unchanged boundaries retain their
initial input values; they are not forced to zero. The source row uses
`_fict_[t] = t` for `t = 0, ..., TMAX-1`.

The benchmark runs the `TMAX`-step simulation exactly once per execution and
records each kernel's GPU event time accumulated over the steps. A logical
access-count model for downstream bandwidth calculations is given below.

## Files

| File | Description |
|------|-------------|
| `polybench_fdtd2d.hip` | HIP/CUDA source — 3 GPU kernels, self-contained |
| `Makefile` | Build system with platform auto-detection |
| `README.md` | This file |

## Build

Platform is auto-detected. Override with `PLATFORM=cuda|rocm`.

```bash
# Auto-detect
make

# Force CUDA
make PLATFORM=cuda

# Force ROCm/HIP
make PLATFORM=rocm

# Clean
make clean
```

### Requirements

- **CUDA**: NVIDIA GPU + CUDA toolkit (`nvcc`)
- **ROCm**: AMD GPU + ROCm/HIP toolkit (`hipcc`)

## Run

```bash
# Default: 512×512 grid, 50 time steps
./polybench_fdtd2d

# Custom grid size, time steps
./polybench_fdtd2d --size 1024 --tmax 100
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--size N` | 512 | Grid dimension (N×N) |
| `--tmax T` | 50 | FDTD time steps per execution |

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_size` | 512 | Grid dimension (N×N) |
| `BENCH_PARAM_tmax` | 50 | FDTD time steps per execution |
| `BENCH_PARAM_vector_add_warmup` | true | Vector-add warmer (clock boost + LLC eviction) before the timed run |

The TMAX-step simulation runs exactly once per execution; repetition is the harness's job (`gpubench run --reps N`).

Resolved size and time-step count must be positive, with `size*size <= INT_MAX`
for the signed initialization bounds and GPU field indices. Invalid requests fail
before allocation or GPU access and emit no timing records. Size 1 remains valid:
the time-varying `ey` source row is still updated. These guards preserve all 60
declared configurations, existing positive-only CLI fallback and environment
`atoi` overrides. GPU work, input generation and full numerical verification
are unchanged; memory capacity and runtime admission remain separate.

## Output

**stdout** — JSON lines: one `{"type":"kernel",...}` record per kernel (`fdtd_update_ex`, `fdtd_update_ey`, `fdtd_update_hz`) with that kernel's GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`) accumulated over the TMAX steps, then a summary whose `total_time_ms` is the sum of all recorded GPU event times:
```
{"type":"kernel","name":"fdtd_update_ex","iter":0,"time_ms":14.2310,"host_time_ms":15.0121,"params":{"size":512,"tmax":50,"verification":"full_cpu_v1"}}
{"type":"kernel","name":"fdtd_update_ey","iter":0,"time_ms":14.8120,"host_time_ms":15.4020,"params":{"size":512,"tmax":50,"verification":"full_cpu_v1"}}
{"type":"kernel","name":"fdtd_update_hz","iter":0,"time_ms":16.1900,"host_time_ms":16.8010,"params":{"size":512,"tmax":50,"verification":"full_cpu_v1"}}
{"type":"summary","total_time_ms":45.2330}
```

The field arrays are uploaded (untimed) at the start of the run, so every execution performs identical work.

**stderr**:
```
Device: NVIDIA GeForce RTX 3090 (id=0)
Grid: 512×512  |  TMAX: 50

ex=14.231 ey=14.812 hz=16.190 ms
Total GPU time: 45.233 ms (50 steps/pass)
```

## Performance Notes

Each time step performs three stencil passes. The following model counts one
float access for each source-level array read or write in the current kernels:

| Kernel | Float accesses per step | Counted work |
|---|---:|---|
| `fdtd_update_ex` | `4 * NX * (NY-1)` | Two `hz` reads, one `ex` read and one `ex` write per updated cell |
| `fdtd_update_ey` | `NY + 4 * (NX-1) * NY` | One source-row write per column; elsewhere two `hz` reads, one `ey` read and one `ey` write |
| `fdtd_update_hz` | `6 * (NX-1) * (NY-1)` | Two `ex` reads, two `ey` reads, one `hz` read and one `hz` write per updated cell |

For the square driver (`NX = NY = N`), the total is
`14*N*N - 19*N + 6` float accesses per step. A logical effective-bandwidth
calculation for the complete simulation is therefore:

```
logical_bytes = (14*N*N - 19*N + 6) * sizeof(float) * TMAX
logical_GB_per_second = logical_bytes / (total_time_ms * 1e-3) / 1e9
```

For a single kernel, use its row's count and its accumulated `time_ms`, with
the same `sizeof(float) * TMAX` factor. Initial field uploads and the warmer
are outside these timed regions and this access model. The benchmark emits
raw times and workload parameters; it does not compute this derived rate.

This is a source-level logical traffic model, not measured DRAM traffic.
Caching, reuse, compiler transformations and memory transactions can change
physical traffic; hardware counters are needed to qualify that quantity.

## Measured-result verification

Every run checks every value in the measured final ex, ey and hz fields against
a complete independent double CPU recurrence from the pristine initial fields.
The reference uses the exact float `0.7f` coefficient promoted to double and
the same source sequence and boundaries. Each timestep completes the electric
field phase before updating hz. It never consumes GPU results as reference
inputs and launches no separate GPU verification pass.

Actual, reference and error must be finite. Updated values require
`abs(actual-reference) <= 1e-3 * max(1,abs(reference))`. This is an empirical
tolerance qualified for the declared inputs and tested controls, not a
universal roundoff bound; smaller errors are intentionally admitted. The
retained ex first column, retained hz last row/column and final ey source row
must match exactly. All mismatches are counted, at most ten details go to
stderr, and any mismatch causes a nonzero exit. Every kernel record includes
`verification="full_cpu_v1"`.

The CPU phases use at most sixteen workers on disjoint row ranges, with a join
between phases. Each value keeps the same operation order at every worker
count. Automatic mode uses one worker below 512×512 cells, otherwise caps by
hardware concurrency and row count. Thread creation failures, including
allocation failures, finish missing partitions on the caller and join prior
workers. There are no concurrent phase pools.

All reference work and three full result downloads follow GPU timing. The
downloads reuse the consumed input buffers. GPU kernels, RNG/inputs, geometry,
source/timestep sequence, initial uploads, warmer/events and declared sweep
are unchanged. At N=8192 all 201,326,592 final values are checked; three double
references add 1.5 GiB to the original 768 MiB host fields. The three 256 MiB
downloads allocate no additional host result arrays. CPU work is outside GPU
event time but inside the process timeout. Other-host admission/budgets,
AMD/tolerance qualification and full upstream qualification remain open.
