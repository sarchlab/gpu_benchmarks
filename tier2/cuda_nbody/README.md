# tier2/cuda_nbody — All-Pairs N-Body Simulation (CUDA N-Body)

Benchmarks a gravitational N-body simulation using the **classic tile-based shared memory optimization**. Each thread computes the force on one body from all others. Emits the run's kernel time; throughput in GFLOPS is derived downstream.

Derived from the [CUDA N-Body sample](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda).

## Algorithm

- **N** bodies (default: 16,384), 10 timesteps per run
- All-pairs gravitational force computation with O(N²) interactions
- **Tile-based shared memory optimization**: bodies loaded in tiles of 256 (compile-time `TILE_SIZE`, also the block size) into shared memory for data reuse
- Each interaction: distance computation, softening (ε=1e-5), inverse cube law → ~20 FLOPs
- After force accumulation: velocity and position integration (leapfrog, dt=0.01)
- GFLOPS = 20 · N² · timesteps / time\_sec / 10⁹ (derived downstream from the emitted times)
- Particle positions/velocities are uploaded pristine before the run (untimed); the run then integrates all `timesteps`

## Measurement

The simulation (all `timesteps` kernel launches) runs **exactly once** per
process execution (see the root README "Measurement Methodology"). With
`BENCH_PARAM_vector_add_warmup` (default `true`), a vector-add warmer boosts
the GPU clock and evicts the last-level cache immediately before the timed
run. Repetition = `gpubench run --reps N`.

## Parameters

| Env var (or `--flag`) | Description | Default |
|---------|-------------|---------|
| `BENCH_PARAM_bodies` / `--bodies` | Number of bodies | 16384 |
| `BENCH_PARAM_timesteps` / `--timesteps` | Simulation timesteps per run | 10 |
| `BENCH_PARAM_vector_add_warmup` | Cold-cache clock warmer on/off | true |

## Files

| File | Description |
|------|-------------|
| `cuda_nbody.hip` | HIP source — tile-based N-body kernel with shared memory |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./cuda_nbody                                   # defaults: N=16384, 10 timesteps
./cuda_nbody --bodies 8192                     # fewer bodies
./cuda_nbody --bodies 16384 --timesteps 20
```

## Output

**stdout** — JSON lines: one `kernel` record (`iter:0`) with the GPU event
time accumulated over all timesteps of the single run, followed by a `summary`
whose `total_time_ms` is the sum of all recorded GPU event times:
```json
{"type":"kernel","name":"nbody_kernel","iter":0,"time_ms":234.567800,"host_time_ms":235.100000,"params":{"bodies":16384,"timesteps":10}}
{"type":"summary","total_time_ms":234.567800}
```

Derived metrics (e.g. GFLOPS) are computed downstream from `time_ms` and params.

**stderr** — human-readable results:
```
Device: NVIDIA A100 (id=0)
Bodies: 16384  |  Timesteps: 10

nbody_kernel gpu = 234.568 ms, host = 235.100 ms
Total GPU time: 234.568 ms
PASS (verified 4096 bodies, 1 timestep)
```

## Verification

Runs 1 timestep on a subset of bodies (up to 4096) and compares GPU results against a CPU reference implementation with relative tolerance of 1e-2. On FAIL the process exits non-zero.

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `cuda_nbody.hip`; tile size = 256 threads with `__shared__` memory
