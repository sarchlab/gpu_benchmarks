# polybench_fdtd2d — PolyBench FDTD-2D Benchmark

2D Finite Difference Time Domain (FDTD) electromagnetic simulation, adapted from the PolyBench benchmark suite. Measures effective memory bandwidth of iterative stencil updates on three 2D field arrays.

## Algorithm

Three field arrays `ex[NX][NY]`, `ey[NX][NY]`, `hz[NX][NY]` are updated each time step:

```
ex[0][j]  = 0                                          (boundary)
ex[i][j] += 0.5*(hz[i][j] - hz[i-1][j])  for i >= 1

ey[i][0]  = 0                                          (boundary)
ey[i][j] += 0.5*(hz[i][j] - hz[i][j-1])  for j >= 1

hz[i][j] -= 0.7*(ex[i][j+1] - ex[i][j] +
                 ey[i+1][j]  - ey[i][j])  for i < NX-1, j < NY-1
```

The benchmark runs `TMAX` time steps per timed pass and records each kernel's accumulated GPU time per pass. Effective bandwidth (`7 * NX * NY * sizeof(float)` bytes per step) is derived downstream.

## Files

| File | Description |
|------|-------------|
| `polybench_fdtd2d.hip` | HIP/CUDA source — 3 GPU kernels, self-contained |
| `polybench_fdtd2d.metal` | Metal compute shader — 3 kernels (`update_ex`, `update_ey`, `update_hz`) |
| `polybench_fdtd2d_metal.mm` | Objective-C++ Metal host program |
| `Makefile` | Build system with platform auto-detection |
| `README.md` | This file |

## Build

Platform is auto-detected. Override with `PLATFORM=cuda|rocm|metal`.

```bash
# Auto-detect
make

# Force CUDA
make PLATFORM=cuda

# Force ROCm/HIP
make PLATFORM=rocm

# Force Metal (macOS only)
make PLATFORM=metal

# Clean
make clean
```

### Requirements

- **CUDA**: NVIDIA GPU + CUDA toolkit (`nvcc`)
- **ROCm**: AMD GPU + ROCm/HIP toolkit (`hipcc`)
- **Metal**: macOS with Apple GPU (`clang++`, Xcode command-line tools)

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
| `--tmax T` | 50 | FDTD time steps per benchmark pass |

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_size` | 512 | Grid dimension (N×N) |
| `BENCH_PARAM_tmax` | 50 | FDTD time steps per benchmark pass |
| `BENCH_PARAM_block_size` | 256 | Threads per block (recorded in params) |
| `BENCH_PARAM_num_warmup` | 3 | Warmup passes (untimed) |
| `BENCH_PARAM_num_iters` | 7 | Timed passes, each recorded |

## Output

**stdout** — JSON lines: per timed pass, one `{"type":"kernel",...}` record per kernel (`fdtd_update_ex`, `fdtd_update_ey`, `fdtd_update_hz`) with that kernel's accumulated GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`) over the pass's TMAX steps, then a summary whose `total_time_ms` is the sum of all recorded GPU event times:
```
{"type":"kernel","name":"fdtd_update_ex","iter":0,"time_ms":14.2310,"host_time_ms":15.0121,"params":{"size":512,"tmax":50,"block_size":256}}
...
{"type":"summary","total_time_ms":316.6170}
```

The field arrays are re-uploaded (untimed) at the start of every pass, so each pass performs identical work.

**stderr**:
```
Device: NVIDIA GeForce RTX 3090 (id=0)
Grid: 512×512  |  TMAX: 50  |  Passes: 3 warmup + 7 timed

pass 0: ex=14.231 ey=14.812 hz=16.190 ms
...
Total GPU time: 316.617 ms over 7 passes (50 steps/pass)
```

## Performance Notes

- Each time step performs three stencil passes over NX×NY float arrays
- Memory-bound benchmark; performance scales with GPU memory bandwidth
- Effective bandwidth formula (derived downstream from `time_ms` and params): `7 * NX * NY * 4 bytes * TMAX / time_seconds`
  - The factor 7 accounts for reads and writes across all three kernels
