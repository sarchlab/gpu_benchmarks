# tier2/altis_particlefilter — Particle Filter (Altis)

Benchmarks a Sequential Importance Resampling (SIR) particle filter for Monte Carlo localization on a 2D grid. Tracks a moving target over multiple time steps using particles with position updates, weight computation, and resampling. The full 10-step filter runs exactly once per execution; raw per-kernel times are emitted and throughput (particles/sec) is derived downstream.

## Provenance & fidelity

Derived from the [Altis benchmark suite](https://github.com/utcs-scea/altis) — the `particlefilter` workload. Independent reimplementation for portable measurement, **not** upstream source, and **not** a faithful reproduction. It differs as follows:

- Closed-form Gaussian likelihood vs Altis's video-frame gather; the inter-block scan runs on the host mid-algorithm. (A missing `__syncthreads` in the prefix-sum kernel has been fixed.)

Read this as "a particle-filter benchmark in the spirit of Altis," not Altis's `particlefilter`.

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
| `BENCH_PARAM_vector_add_warmup` | true | Cold-cache calibration: boost the clock and evict the LLC before the timed run; set `false` to time as-is |

The initialization seed is fixed (12345) and echoed in the emitted params; it
is not configurable.

## Files

| File | Description |
|------|-------------|
| `altis_particlefilter.hip` | HIP source — particle filter kernels (update, weights, scan, resample) |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |
| `params.json` | Parameter specification |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./altis_particlefilter                              # defaults: N=100000
./altis_particlefilter --size 50000                 # fewer particles
BENCH_PARAM_size=50000 ./altis_particlefilter       # env vars override argv
```

## Output

**stdout** — JSON lines. The single timed run emits one `{"type":"kernel",...}`
record per kernel (`update_particles_kernel`, `compute_weights_kernel`,
`prefix_sum_kernel`, `add_block_sums_kernel`, `resample_kernel`) with that
kernel's accumulated GPU event time (`time_ms`) and host wall-clock time
(`host_time_ms`) across the run's 10 time steps, plus the run params. The final
summary's `total_time_ms` is the sum of all recorded GPU event times:
```
{"type":"kernel","name":"update_particles_kernel","iter":0,"time_ms":1.234,"host_time_ms":1.301,"params":{"size":100000,"seed":12345}}
...
{"type":"summary","total_time_ms":5.234}
```
Derived metrics (particles/sec) are computed downstream from `time_ms` and params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
Particle Filter  |  Particles: 100000  |  Grid: 128x128  |  Steps/pass: 10

update=1.234 weights=0.456 prefix=0.789 addblock=0.123 resample=2.345 ms (gpu)
Total GPU time: 5.234 ms
GPU mean: (64.1234, 63.8765)  CPU mean: (64.2345, 63.7654)
PASS
```

## Verification

- Runs both GPU and CPU particle filter on same initial data (up to 10,000 particles)
- Compares mean particle positions — both should converge to the observation region
- Reports PASS if distance between GPU and CPU mean < 20 grid units (generous tolerance for stochastic algorithm); on FAIL the binary exits with a non-zero status

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `altis_particlefilter.hip`; uses xorshift32 PRNG per particle; Blelloch scan for prefix sum with host-side block sum fixup
