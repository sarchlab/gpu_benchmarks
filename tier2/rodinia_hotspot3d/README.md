# tier2/rodinia_hotspot3d — 3D Thermal Simulation (Rodinia HotSpot3D)

3D stencil thermal simulation extending the classic 2D HotSpot benchmark to three dimensions. Each cell's temperature is updated based on its six neighbors (±x, ±y, ±z), local power density, and thermal resistances in an N×N×N grid.

## Provenance & fidelity

Derived from the [Rodinia benchmark suite](http://lava.cs.virginia.edu/Rodinia/) — the `hotspot3D` workload. Independent reimplementation for portable measurement, **not** upstream source, and **not** a faithful reproduction. It differs as follows:

- Naive one-thread-per-cell kernel vs Rodinia's z-marching register-reuse kernel. It reads all six neighbors from global memory (no tiling), computing the same stencil.

Read this as "a 3D thermal stencil benchmark in the spirit of Rodinia," not Rodinia's `hotspot3D`.

## Algorithm

For each cell `(x, y, z)` in an `N×N×N` grid:
```
T_new[z][y][x] = T[z][y][x] + step_div_cap * (
    power[z][y][x]
    + (T[z][y][x-1] + T[z][y][x+1] - 2*T[z][y][x]) * Rx_1   // ±x neighbors
    + (T[z][y-1][x] + T[z][y+1][x] - 2*T[z][y][x]) * Ry_1   // ±y neighbors
    + (T[z-1][y][x] + T[z+1][y][x] - 2*T[z][y][x]) * Rz_1   // ±z neighbors
    + (AMB_TEMP - T[z][y][x]) * Ra_1                           // ambient heat loss
)
```
Boundary cells use clamped (no-flux) boundary conditions.

The simulation runs `num_iterations` time-steps using ping-pong buffers. The
benchmark executes the simulation **exactly once**; repetition is the
harness's job (`gpubench run --reps N`).

## Files

| File | Description |
|------|-------------|
| `rodinia_hotspot3d.hip` | HIP source — 3D stencil kernel + self-contained utilities |
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
./rodinia_hotspot3d                                         # defaults: 64×64×64, 10 steps
BENCH_PARAM_grid_size=128 ./rodinia_hotspot3d
BENCH_PARAM_grid_size=32 BENCH_PARAM_num_iterations=20 ./rodinia_hotspot3d
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_grid_size` | 64 | Grid dimension per axis (N×N×N) |
| `BENCH_PARAM_block_size` | 8 | Thread-block dimension (B×B×B) |
| `BENCH_PARAM_num_iterations` | 10 | Stencil time-steps per run |
| `BENCH_PARAM_amb_temp` | 80.0 | Ambient temperature |
| `BENCH_PARAM_vector_add_warmup` | `true` | Cold-cache calibration before the timed run |

Each variable also has a command-line fallback (`--grid_size`, `--block_size`,
`--num_iterations`, `--amb_temp`); the environment variables win.

## Output

**stdout** — JSON lines: one `{"type":"kernel",...}` record with the kernel's accumulated GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`) over the `num_iterations` steps, then a summary whose `total_time_ms` is the sum of all recorded GPU event times:
```
{"type":"kernel","name":"hotspot3d_kernel","iter":0,"time_ms":1.234,"host_time_ms":1.301,"params":{"grid_size":64,"block_size":8,"num_iterations":10,"amb_temp":80.0}}
{"type":"summary","total_time_ms":1.234}
```

Derived metrics (effective bandwidth) are computed downstream from `time_ms` and the params.

**stderr** — human-readable results:
```
Device: NVIDIA GeForce RTX 4090 (id=0)
Grid: 64×64×64  |  Block: 8³  |  Steps/pass: 10  |  single timed pass

hotspot3d_kernel gpu = 1.234 ms, host = 1.301 ms
Total GPU time: 1.234 ms over 1 pass (10 steps/pass)
```

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc`
- Input data is generated synthetically — no external input files required
