# tier2/altis_particlefilter — Particle Filter (Altis)

Benchmarks a Sequential Importance Resampling (SIR) particle filter for Monte Carlo localization on a 2D grid. Tracks a moving target over multiple time steps using particles with position updates, weight computation, and resampling. Emits raw per-pass kernel times; throughput (particles/sec) is derived downstream.

Derived from the [Altis benchmark suite](https://github.com/utcs-scea/altis) Particle Filter workload.

## Algorithm

- **N** particles (default: N=100,000) on a 128×128 grid, 10 time steps
- Three GPU kernels per time step:
  1. **Update positions** — random walk: each particle moves by Gaussian noise (σ=2.0)
  2. **Compute weights** — Gaussian likelihood: w = exp(−d² / 2σ²) where d is distance to observation
  3. **Resample** — systematic resampling: parallel prefix sum of weights, then binary search in CDF
- Synthetic observations: target moves in a circle (radius 20) around grid center
- particles/sec = N × timesteps / time\_sec (derived downstream from the emitted times and params)

## Parameters (environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_size` | 100000 | Number of particles |
| `BENCH_PARAM_num_particles_ratio` | 0.5 | Resampling threshold ratio (0.25, 0.5, 0.75, 1.0) |
| `BENCH_PARAM_seed` | 42 | RNG seed |
| `BENCH_PARAM_warmup` | 3 | Warmup filter passes (not recorded) |
| `BENCH_PARAM_iteration` | 7 | Timed filter passes, each recorded |

## Files

| File | Description |
|------|-------------|
| `altis_particlefilter.hip` | HIP source — particle filter kernels (update, weights, scan, resample) |
| `altis_particlefilter.metal` | Metal compute shaders — all particle filter kernels |
| `altis_particlefilter_metal.mm` | ObjC++ Metal host code |
| `Makefile` | Auto-detecting build (Darwin→metal, ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
make PLATFORM=metal
```

## Run

```bash
./altis_particlefilter                              # defaults: N=100000
./altis_particlefilter --size 50000                 # fewer particles
BENCH_PARAM_iteration=10 ./altis_particlefilter --size 100000
```

## Output

**stdout** — JSON lines. Each timed pass emits one `{"type":"kernel",...}`
record per kernel (`update_particles_kernel`, `compute_weights_kernel`,
`prefix_sum_kernel`, `add_block_sums_kernel`, `resample_kernel`) with that
kernel's accumulated GPU event time (`time_ms`) and host wall-clock time
(`host_time_ms`) across the pass's time steps, plus the run params. The final
summary's `total_time_ms` is the sum of all recorded GPU event times:
```
{"type":"kernel","name":"update_particles_kernel","iter":0,"time_ms":1.234,"host_time_ms":1.301,"params":{"size":100000,"num_particles_ratio":"0.5","seed":42}}
...
{"type":"summary","total_time_ms":123.456}
```
Derived metrics (particles/sec) are computed downstream from `time_ms` and params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
Particle Filter  |  Particles: 100000  |  Grid: 128x128  |  Steps/pass: 10  |  Passes: 3 warmup + 7 timed

pass 0: update=1.234 weights=0.456 prefix=0.789 addblock=0.123 resample=2.345 ms (gpu)
...
Total GPU time: 123.456 ms over 7 passes
GPU mean: (64.1234, 63.8765)  CPU mean: (64.2345, 63.7654)
PASS
```

## Verification

- Runs both GPU and CPU particle filter on same initial data (up to 10,000 particles)
- Compares mean particle positions — both should converge to the observation region
- Reports PASS if distance between GPU and CPU mean < 20 grid units (generous tolerance for stochastic algorithm)

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `altis_particlefilter.hip`; uses xorshift32 PRNG per particle; Blelloch scan for prefix sum with host-side block sum fixup
- **Metal**: compiled with `clang++` from `altis_particlefilter_metal.mm`; the Metal shader (`altis_particlefilter.metal`) is loaded and compiled at runtime from the same directory as the binary
