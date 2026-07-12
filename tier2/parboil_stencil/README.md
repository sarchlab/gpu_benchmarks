# tier2/parboil_stencil — 3D Jacobi Stencil (Parboil)

7-point 3D Jacobi stencil computation over a cubic grid using ping-pong buffers. Each interior cell is updated from its six face-adjacent neighbors. Boundary cells remain fixed. A classic memory-bandwidth-bound benchmark.

## Provenance & fidelity

Derived from the [Parboil benchmark suite](http://impact.crhc.illinois.edu/parboil/) — the `stencil` workload. Independent reimplementation for portable measurement, **not** upstream source, and **not** a faithful reproduction. It differs as follows:

- Different stencil formula than Parboil's.

Read this as "a 3D Jacobi stencil benchmark in the spirit of Parboil," not Parboil's `stencil`.

## Algorithm

For each interior cell `(ix, iy, iz)` of an `N×N×N` grid:
```
out[i] = c0 * in[i]
       + c1 * (in[i-1] + in[i+1]          // x neighbors
             + in[i-nx] + in[i+nx]         // y neighbors
             + in[i-nx*ny] + in[i+nx*ny])  // z neighbors
```
where `c0 = 0.6`, `c1 = (1 - c0) / 6 ≈ 0.0667`.

Boundary cells (1-cell border) are skipped. The run performs `num_timesteps`
time-steps using ping-pong buffers.

## Measurement

The stencil (all `num_timesteps` launches) runs **exactly once** per process
execution (see the root README "Measurement Methodology"). With
`BENCH_PARAM_vector_add_warmup` (default `true`), a vector-add warmer boosts
the GPU clock and evicts the last-level cache immediately before the timed
run. Repetition = `gpubench run --reps N`. There is **no result verification**
in this benchmark.

## Parameters

| Env var (or `--flag`) | Description | Default |
|----------|-------------|---------|
| `BENCH_PARAM_grid_dim` / `--grid_dim` | Grid dimension N (N×N×N) | 64 |
| `BENCH_PARAM_num_timesteps` / `--num_timesteps` | Stencil time-steps per run | 10 |
| `BENCH_PARAM_vector_add_warmup` | Cold-cache clock warmer on/off | true |

## Files

| File | Description |
|------|-------------|
| `parboil_stencil.hip` | HIP source — 3D stencil kernel + self-contained host utilities |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./parboil_stencil                                          # defaults: 64×64×64, 10 timesteps
./parboil_stencil --grid_dim 128
./parboil_stencil --grid_dim 64 --num_timesteps 20
```

## Output

**stdout** — JSON lines: one `{"type":"kernel",...}` record (`iter:0`) carrying the stencil kernel's GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`) accumulated over the run's `num_timesteps` launches, then a summary whose `total_time_ms` is the sum of all recorded GPU event times:
```
{"type":"kernel","name":"stencil3d","iter":0,"time_ms":1.234500,"host_time_ms":1.301100,"params":{"grid_dim":64,"num_timesteps":10}}
{"type":"summary","total_time_ms":1.234500}
```

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
Grid: 64×64×64  |  Timesteps/pass: 10

stencil3d gpu = 1.234 ms, host = 1.301 ms
Total GPU time: 1.234 ms (10 timesteps/pass)
```

## Performance Metrics (derived downstream from time_ms and params)

- **GFLOPS** = 8 × interior_cells × num_timesteps / time_s / 1e9
- **GB/s** = 8 × sizeof(float) × interior_cells × num_timesteps / time_s / 1e9
  (7 reads + 1 write per interior cell per time-step)

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `parboil_stencil.hip`
- Input data is generated synthetically — no external input files required
