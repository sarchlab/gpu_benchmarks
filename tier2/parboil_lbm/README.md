# tier2/parboil_lbm — Lattice Boltzmann Method D3Q19 (Parboil)

Benchmarks a Lattice Boltzmann Method (LBM) fluid simulation using the D3Q19 velocity set on a regular 3D grid. Each GPU thread processes one lattice node, performing a fused collide-stream operation with the BGK collision operator and bounce-back boundary conditions. Emits per-pass kernel times; throughput in MLUPS (million lattice updates per second) is derived downstream.

Derived from the [Parboil benchmark suite](http://impact.crhc.illinois.edu/parboil/parboil.aspx) LBM workload.

## Algorithm

- **NxNxN** 3D lattice (default: N=64, i.e. 262,144 nodes)
- D3Q19 velocity set with 19 discrete velocities per node
- BGK collision operator: f_post = f + ω(f_eq − f), where ω = 1/τ
- Equilibrium distribution: f_eq = w_q × ρ × (1 + 3(e·u) + 4.5(e·u)² − 1.5|u|²)
- Fused collide-stream kernel with bounce-back at domain boundaries
- Double-buffered distribution arrays, swapped each timestep
- MLUPS = N³ × num_timesteps / time_sec / 10⁶ (derived downstream from the emitted times)
- The equilibrium initial state is re-uploaded before every measurement pass (untimed), so each recorded pass simulates the same `num_timesteps` from the same starting point

## Parameters

| Env var | Description | Default |
|---------|-------------|---------|
| `BENCH_PARAM_grid_dim` | Grid dimension N (N×N×N lattice) | 64 |
| `BENCH_PARAM_block_size` | Threads per block | 128 |
| `BENCH_PARAM_num_timesteps` | LBM timesteps per pass | 100 |
| `BENCH_PARAM_tau` | BGK relaxation time | 0.7 |
| `BENCH_PARAM_num_warmup` | Warmup passes (untimed) | 3 |
| `BENCH_PARAM_num_iters` | Timed passes, each recorded | 7 |

`BENCH_PARAM_iterations` is superseded by `BENCH_PARAM_num_iters` and is ignored.

## Files

| File | Description |
|------|-------------|
| `parboil_lbm.hip` | HIP source — portable ROCm/CUDA implementation |
| `parboil_lbm.metal` | Metal compute shader — LBM kernel |
| `parboil_lbm_metal.mm` | ObjC++ Metal host code |
| `Makefile` | Auto-detecting build (Darwin→metal, ROCm→rocm, else→cuda) |
| `params.json` | Parameter specification |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
make PLATFORM=metal
```

## Run

```bash
./parboil_lbm                                       # defaults: 64^3, 100 timesteps
BENCH_PARAM_grid_dim=128 ./parboil_lbm              # larger grid
BENCH_PARAM_num_timesteps=50 BENCH_PARAM_tau=1.0 ./parboil_lbm
```

## Output

**stdout** — JSON lines: one `kernel` record per timed pass (GPU event time
and host time accumulated over the pass's `num_timesteps` kernel launches),
then a `summary` whose `total_time_ms` is the sum of all recorded GPU event
times:
```json
{"type":"kernel","name":"lbm_collide_stream_kernel","iter":0,"time_ms":45.678900,"host_time_ms":46.102000,"params":{"grid_dim":64,"block_size":128,"num_timesteps":100,"tau":0.70}}
...
{"type":"summary","total_time_ms":319.752300}
```

Derived metrics (MLUPS) are computed downstream from `time_ms` and params.

**stderr** — human-readable:
```
Device: Apple M1 Max
LBM D3Q19  |  Grid: 64x64x64 = 262144  |  Timesteps: 100  |  Tau: 0.70  |  Iterations: 3 warmup + 7 timed

pass 0: gpu = 45.679 ms, host = 46.102 ms
...
Total GPU time: 319.752 ms over 7 passes
PASS
```

## Verification

- 128 randomly-selected interior nodes are checked for reasonable density (0.5 < ρ < 2.0)
- Near-equilibrium initialization ensures density should stay close to 1.0
- Reports PASS/FAIL with details of suspicious nodes

## Platform Notes

- **CUDA/ROCm**: Uses `__constant__` memory for D3Q19 velocity vectors and weights; SoA layout for distribution functions
- **Metal**: Uses `constant` array literals in the shader; loads `parboil_lbm.metal` at runtime from the same directory as the binary
