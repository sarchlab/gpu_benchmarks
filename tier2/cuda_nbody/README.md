# tier2/cuda_nbody — All-Pairs N-Body Simulation (CUDA N-Body)

Benchmarks a gravitational N-body simulation using the **classic tile-based shared memory optimization**. Each thread computes the force on one body from all others. Emits per-iteration kernel times; throughput in GFLOPS is derived downstream.

Derived from the [CUDA N-Body sample](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda).

## Algorithm

- **N** bodies (default: 16,384), 10 timesteps per run
- All-pairs gravitational force computation with O(N²) interactions
- **Tile-based shared memory optimization**: bodies loaded in tiles of 256 into threadgroup/shared memory for data reuse
- Each interaction: distance computation, softening (ε=1e-5), inverse cube law → ~20 FLOPs
- After force accumulation: velocity and position integration (leapfrog, dt=0.01)
- GFLOPS = 20 · N² · timesteps / time\_sec / 10⁹ (derived downstream from the emitted times)
- Each timed pass resets particle positions/velocities to the pristine initial state (untimed) before running all timesteps, so every pass simulates the same trajectory

## Parameters

| Env var | Description | Default |
|---------|-------------|---------|
| `BENCH_PARAM_bodies` | Number of bodies | 16384 |
| `BENCH_PARAM_timesteps` | Simulation timesteps per pass | 10 |
| `BENCH_PARAM_iterations` | Legacy count, echoed into records only | 5 |
| `BENCH_PARAM_block_size` | Threads per block | 256 |
| `BENCH_PARAM_num_warmup` | Warmup passes (untimed) | 3 |
| `BENCH_PARAM_num_iters` | Timed passes, each recorded | 7 |

## Files

| File | Description |
|------|-------------|
| `cuda_nbody.hip` | HIP source — tile-based N-body kernel with shared memory |
| `cuda_nbody.metal` | Metal compute shader — tile-based N-body kernel with threadgroup memory |
| `cuda_nbody_metal.mm` | ObjC++ Metal host code |
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
./cuda_nbody                                   # defaults: N=16384, 10 timesteps
./cuda_nbody --bodies 8192                     # fewer bodies
./cuda_nbody --bodies 16384 --timesteps 20
```

## Output

**stdout** — JSON lines, one `kernel` record per timed pass (the accumulated
GPU event time of all timesteps within the pass) followed by a `summary`
whose `total_time_ms` is the sum of all recorded GPU event times:
```json
{"type":"kernel","name":"nbody_kernel","iter":0,"time_ms":234.567800,"host_time_ms":235.100000,"params":{"bodies":16384,"timesteps":10,"iterations":5,"block_size":256}}
...
{"type":"summary","total_time_ms":1641.974600}
```

Derived metrics (e.g. GFLOPS) are computed downstream from `time_ms` and params.

**stderr** — human-readable results:
```
Device: NVIDIA A100 (id=0)
Bodies: 16384  |  Timesteps: 10  |  Iterations: 3 warmup + 7 timed

pass 0: nbody_kernel gpu = 234.568 ms, host = 235.100 ms
...
Total GPU time: 1641.975 ms over 7 passes
PASS (verified 4096 bodies, 1 timestep)
```

## Verification

Runs 1 timestep on a subset of bodies (up to 4096) and compares GPU results against a CPU reference implementation with relative tolerance of 1e-2.

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `cuda_nbody.hip`; tile size = 256 threads with `__shared__` memory
- **Metal**: compiled with `clang++` from `cuda_nbody_metal.mm`; tile size = 256 threads with `threadgroup` memory; the Metal shader (`cuda_nbody.metal`) is loaded and compiled at runtime from the same directory as the binary
