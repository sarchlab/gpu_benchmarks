# tier2/parboil_stencil — 3D Jacobi Stencil (Parboil)

7-point 3D Jacobi stencil computation over a cubic grid using ping-pong buffers. Each interior cell is updated from its six face-adjacent neighbors. Boundary cells remain fixed. A classic memory-bandwidth-bound benchmark.

Derived from the [Parboil benchmark suite](http://impact.crhc.illinois.edu/parboil/parboil.aspx) stencil workload.

## Algorithm

For each interior cell `(ix, iy, iz)` of an `N×N×N` grid:
```
out[i] = c0 * in[i]
       + c1 * (in[i-1] + in[i+1]          // x neighbors
             + in[i-nx] + in[i+nx]         // y neighbors
             + in[i-nx*ny] + in[i+nx*ny])  // z neighbors
```
where `c0 = 0.6`, `c1 = (1 - c0) / 6 ≈ 0.0667`.

Boundary cells (1-cell border) are skipped. Multiple time-steps are performed per benchmark iteration using ping-pong buffers.

## Files

| File | Description |
|------|-------------|
| `parboil_stencil.hip` | HIP source — 3D stencil kernel + self-contained host utilities |
| `parboil_stencil.metal` | Metal compute shader — 3D stencil kernel |
| `parboil_stencil_metal.mm` | ObjC++ Metal host code |
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
./parboil_stencil                                          # defaults: 64×64×64, 10 timesteps
./parboil_stencil --grid_dim 128
./parboil_stencil --grid_dim 64 --num_timesteps 20
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_grid_dim` | 64 | Grid dimension N (N×N×N) |
| `BENCH_PARAM_num_timesteps` | 10 | Stencil time-steps per pass |
| `BENCH_PARAM_block_size` | 256 | Threads per block (recorded in params) |
| `BENCH_PARAM_num_warmup` | 3 | Warmup passes (untimed) |
| `BENCH_PARAM_num_iters` | 7 | Timed passes, each recorded |

## Output

**stdout** — JSON lines: per timed pass, one `{"type":"kernel",...}` record carrying the stencil kernel's accumulated GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`) over the pass's `num_timesteps` launches, then a summary whose `total_time_ms` is the sum of all recorded GPU event times:
```
{"type":"kernel","name":"stencil3d","iter":0,"time_ms":1.2345,"host_time_ms":1.3011,"params":{"grid_dim":64,...}}
...
{"type":"summary","total_time_ms":8.6415}
```

**stderr** — human-readable results:
```
Device: Apple M4 Pro
Grid: 64×64×64  |  Timesteps/pass: 10  |  Passes: 3 warmup + 7 timed

pass 0: stencil3d gpu = 1.234 ms, host = 1.301 ms
...
Total GPU time: 8.642 ms over 7 passes (10 timesteps/pass)
```

## Performance Metrics (derived downstream from time_ms and params)

- **GFLOPS** = 8 × interior_cells × num_timesteps / time_s / 1e9
- **GB/s** = 8 × sizeof(float) × interior_cells × num_timesteps / time_s / 1e9
  (7 reads + 1 write per interior cell per time-step)

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `parboil_stencil.hip`
- **Metal**: compiled with `clang++` from `parboil_stencil_metal.mm`; the Metal shader is loaded at runtime from the same directory as the binary
- Input data is generated synthetically — no external input files required
