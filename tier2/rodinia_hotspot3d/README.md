# tier2/rodinia_hotspot3d — 3D Thermal Simulation (Rodinia HotSpot3D)

3D stencil thermal simulation extending the classic 2D HotSpot benchmark to three dimensions. Each cell's temperature is updated based on its six neighbors (±x, ±y, ±z), local power density, and thermal resistances in an N×N×N grid.

Derived from the [Rodinia benchmark suite](https://rodinia.cs.virginia.edu/) HotSpot3D workload.

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

Multiple time-steps are run per benchmark pass using ping-pong buffers. The pristine temperature grid is re-uploaded (untimed) at the start of every pass, so each pass performs identical work.

## Files

| File | Description |
|------|-------------|
| `rodinia_hotspot3d.hip` | HIP source — 3D stencil kernel + self-contained utilities |
| `rodinia_hotspot3d.metal` | Metal compute shader — 3D stencil kernel |
| `rodinia_hotspot3d_metal.mm` | ObjC++ Metal host code |
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
./rodinia_hotspot3d                                         # defaults: 64×64×64, 10 steps
BENCH_PARAM_grid_size=128 ./rodinia_hotspot3d
BENCH_PARAM_grid_size=32 BENCH_PARAM_num_iterations=20 ./rodinia_hotspot3d
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_grid_size` | 64 | Grid dimension per axis (N×N×N) |
| `BENCH_PARAM_block_size` | 8 | Thread-block dimension (B×B×B) |
| `BENCH_PARAM_num_iterations` | 10 | Stencil time-steps per pass |
| `BENCH_PARAM_amb_temp` | 80.0 | Ambient temperature |
| `BENCH_PARAM_num_warmup` | 3 | Warmup passes (untimed) |
| `BENCH_PARAM_num_iters` | 7 | Timed passes, each recorded |

## Output

**stdout** — JSON lines: per timed pass, one `{"type":"kernel",...}` record with the kernel's accumulated GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`) over the pass's `num_iterations` steps, then a summary whose `total_time_ms` is the sum of all recorded GPU event times:
```
{"type":"kernel","name":"hotspot3d_kernel","iter":0,"time_ms":1.234,"host_time_ms":1.301,"params":{"grid_size":64,...}}
...
{"type":"summary","total_time_ms":8.638}
```

Derived metrics (effective bandwidth) are computed downstream from `time_ms` and the params.

**stderr** — human-readable results:
```
Device: Apple M4 Pro
Grid: 64×64×64  |  Block: 8³  |  Steps/pass: 10  |  Passes: 3 warmup + 7 timed
pass 0: hotspot3d_kernel gpu = 1.234 ms, host = 1.301 ms
...
Total GPU time: 8.638 ms over 7 passes (10 steps/pass)
```

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc`
- **Metal**: compiled with `clang++`; the Metal shader is loaded at runtime from the same directory as the binary
- Input data is generated synthetically — no external input files required
