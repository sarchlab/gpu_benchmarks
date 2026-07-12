# tier2/parboil_lbm — Lattice Boltzmann Method D3Q19 (Parboil)

Benchmarks a Lattice Boltzmann Method (LBM) fluid simulation using the D3Q19 velocity set on a regular 3D grid. Each GPU thread processes one lattice node, performing a fused collide-stream operation with the BGK collision operator and bounce-back boundary conditions. Emits the run's kernel time; throughput in MLUPS (million lattice updates per second) is derived downstream.

## Provenance & fidelity

Derived from the [Parboil benchmark suite](http://impact.crhc.illinois.edu/parboil/) — the `lbm` workload. Independent reimplementation for portable measurement, **not** upstream source, and **not** a faithful reproduction. It differs as follows:

- Generic D3Q19 lid/channel LBM with link bounce-back; no obstacles and not Parboil's memory layout. (A prior write-write race at domain faces has been fixed.) It uses an SoA (structure-of-arrays) distribution layout with double-buffered ping-pong streaming, and does *not* reproduce the upstream Parboil cell layout (margin cells, per-cell flags, obstacle geometry).

Read this as "an LBM benchmark in the spirit of Parboil," not Parboil's `lbm`.

## Algorithm

- **NxNxN** 3D lattice (default: N=64, i.e. 262,144 nodes)
- D3Q19 velocity set with 19 discrete velocities per node
- BGK collision operator: f_post = f + ω(f_eq − f), where ω = 1/τ
- Equilibrium distribution: f_eq = w_q × ρ × (1 + 3(e·u) + 4.5(e·u)² − 1.5|u|²)
- Fused collide-stream kernel with bounce-back at domain boundaries
- SoA distribution arrays, double-buffered and swapped each timestep
- MLUPS = N³ × num_timesteps / time_sec / 10⁶ (derived downstream from the emitted times)
- The near-equilibrium initial state is uploaded before the run (untimed); the run then advances the full `num_timesteps`

## Measurement

The simulation (all `num_timesteps` timesteps) runs **exactly once** per
process execution (see the root README "Measurement Methodology"). With
`BENCH_PARAM_vector_add_warmup` (default `true`), a vector-add warmer boosts
the GPU clock and evicts the last-level cache immediately before the timed
run. Repetition = `gpubench run --reps N`.

## Parameters

| Env var | Description | Default |
|---------|-------------|---------|
| `BENCH_PARAM_grid_dim` | Grid dimension N (N×N×N lattice) | 64 |
| `BENCH_PARAM_block_size` | Threads per block | 128 |
| `BENCH_PARAM_num_timesteps` | LBM timesteps per run | 100 |
| `BENCH_PARAM_tau` | BGK relaxation time | 0.7 |
| `BENCH_PARAM_vector_add_warmup` | Cold-cache clock warmer on/off | true |

## Files

| File | Description |
|------|-------------|
| `parboil_lbm.hip` | HIP source — portable ROCm/CUDA implementation |
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
./parboil_lbm                                       # defaults: 64^3, 100 timesteps
BENCH_PARAM_grid_dim=128 ./parboil_lbm              # larger grid
BENCH_PARAM_num_timesteps=50 BENCH_PARAM_tau=1.0 ./parboil_lbm
```

## Output

**stdout** — JSON lines: one `kernel` record (`iter:0`) with the GPU event and
host times accumulated over the run's `num_timesteps` kernel launches, then a
`summary` whose `total_time_ms` is the sum of all recorded GPU event times:
```json
{"type":"kernel","name":"lbm_collide_stream_kernel","iter":0,"time_ms":45.678900,"host_time_ms":46.102000,"params":{"grid_dim":64,"block_size":128,"num_timesteps":100,"tau":0.70}}
{"type":"summary","total_time_ms":45.678900}
```

Derived metrics (MLUPS) are computed downstream from `time_ms` and params.

**stderr** — human-readable:
```
Device: AMD Radeon RX 7900 XTX (id=0)
LBM D3Q19  |  Grid: 64x64x64 = 262144  |  Timesteps: 100  |  Tau: 0.70

gpu = 45.679 ms, host = 46.102 ms
Total GPU time: 45.679 ms
PASS
```

## Verification

- 128 randomly-selected interior nodes are checked for reasonable density (0.5 < ρ < 2.0)
- Near-equilibrium initialization ensures density should stay close to 1.0
- Reports PASS/FAIL with details of suspicious nodes; on FAIL the process exits non-zero

## Platform Notes

- **CUDA/ROCm**: Uses `__constant__` memory for D3Q19 velocity vectors and weights; SoA layout for distribution functions
