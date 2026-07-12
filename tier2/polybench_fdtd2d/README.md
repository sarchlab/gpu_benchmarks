# polybench_fdtd2d — PolyBench FDTD-2D Benchmark

2D Finite Difference Time Domain (FDTD) electromagnetic simulation implementing the canonical PolyBench FDTD-2D recurrence. Measures effective memory bandwidth of iterative stencil updates on three 2D field arrays.

## Provenance & fidelity

Derived from the [PolyBench/GPU benchmark suite](https://sourceforge.net/projects/polybench/) — the `fdtd-2d` workload. Independent reimplementation for portable measurement, **not** upstream source, but it reproduces the canonical PolyBench FDTD-2D recurrence: `ey` carries the `i`-difference plus the `_fict_[t] = t` time-source row, `ex` carries the `j`-difference, both E-field updates subtract, and `hz -= 0.7·(∂ex + ∂ey)`. (An earlier version had the `ex`/`ey` difference axes swapped, added instead of subtracted, and dropped the source; that has been corrected.)

## Algorithm

Three field arrays `ex[NX][NY]`, `ey[NX][NY]`, `hz[NX][NY]` are updated each time step. The equations actually computed are:

```
ex[0][j]  = 0                                          (boundary)
ex[i][j] += 0.5*(hz[i][j] - hz[i-1][j])  for i >= 1

ey[i][0]  = 0                                          (boundary)
ey[i][j] += 0.5*(hz[i][j] - hz[i][j-1])  for j >= 1

hz[i][j] -= 0.7*(ex[i][j+1] - ex[i][j] +
                 ey[i+1][j]  - ey[i][j])  for i < NX-1, j < NY-1
```

**Fidelity to canonical PolyBench:** this implements the upstream FDTD-2D recurrence — `ey[0][j] = _fict_[t]` (with `_fict_[t] = t`), `ey[i][j] -= 0.5*(hz[i][j]-hz[i-1][j])` for `i ≥ 1`, `ex[i][j] -= 0.5*(hz[i][j]-hz[i][j-1])` for `j ≥ 1`, and `hz[i][j] -= 0.7*(ex[i][j+1]-ex[i][j] + ey[i+1][j]-ey[i][j])`.

The benchmark runs the `TMAX`-step simulation exactly once per execution and records each kernel's GPU time accumulated over the steps. Effective bandwidth (`7 * NX * NY * sizeof(float)` bytes per step) is derived downstream.

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

## Output

**stdout** — JSON lines: one `{"type":"kernel",...}` record per kernel (`fdtd_update_ex`, `fdtd_update_ey`, `fdtd_update_hz`) with that kernel's GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`) accumulated over the TMAX steps, then a summary whose `total_time_ms` is the sum of all recorded GPU event times:
```
{"type":"kernel","name":"fdtd_update_ex","iter":0,"time_ms":14.2310,"host_time_ms":15.0121,"params":{"size":512,"tmax":50}}
{"type":"kernel","name":"fdtd_update_ey","iter":0,"time_ms":14.8120,"host_time_ms":15.4020,"params":{"size":512,"tmax":50}}
{"type":"kernel","name":"fdtd_update_hz","iter":0,"time_ms":16.1900,"host_time_ms":16.8010,"params":{"size":512,"tmax":50}}
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

- Each time step performs three stencil passes over NX×NY float arrays
- Memory-bound benchmark; performance scales with GPU memory bandwidth
- Effective bandwidth formula (derived downstream from `time_ms` and params): `7 * NX * NY * 4 bytes * TMAX / time_seconds`
  - The factor 7 accounts for reads and writes across all three kernels
