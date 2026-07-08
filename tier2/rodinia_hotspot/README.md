# tier2/rodinia_hotspot — 2D Thermal Simulation (Rodinia Hotspot)

Iterative 2D stencil thermal simulation for chip temperature estimation. Each cell's temperature is updated based on its four neighbors (N/S/E/W), local power density, and thermal resistances. Classic structured-grid benchmark.

Derived from the [Rodinia benchmark suite](https://rodinia.cs.virginia.edu/) Hotspot workload.

## Algorithm

For each cell `(row, col)` in an `N×N` grid:
```
T_new[i][j] = T[i][j] + step_div_cap * (
    power[i][j]
    + (T[i-1][j] + T[i+1][j] - 2*T[i][j]) * Ry_1    // N-S neighbors
    + (T[i][j-1] + T[i][j+1] - 2*T[i][j]) * Rx_1    // E-W neighbors
    + (AMB_TEMP  - T[i][j]) * Rz_1                    // heat to substrate
)
```
Boundary cells use clamped (no-flux) boundary conditions.

Multiple time-steps are run per timed pass using ping-pong buffers; the
initial temperature grid is re-uploaded (untimed) at the start of every pass
so all passes simulate the same trajectory.

## Parameters

| Env var | Default | Description |
|---------|---------|-------------|
| `BENCH_PARAM_grid_size` | 512 | Grid dimension (N×N) |
| `BENCH_PARAM_block_size` | 16 | Thread-block dimension (B×B) |
| `BENCH_PARAM_num_iterations` | 10 | Stencil time-steps per pass |
| `BENCH_PARAM_num_warmup` | 3 | Warmup passes (untimed) |
| `BENCH_PARAM_num_iters` | 7 | Timed passes, each recorded |

## Files

| File | Description |
|------|-------------|
| `rodinia_hotspot.hip` | HIP source — stencil kernel + self-contained utilities |
| `rodinia_hotspot.metal` | Metal compute shader — stencil kernel |
| `rodinia_hotspot_metal.mm` | ObjC++ Metal host code |
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
./rodinia_hotspot                                         # defaults: 512×512, 10 steps/pass
./rodinia_hotspot --grid_size 1024
./rodinia_hotspot --grid_size 256 --num_iterations 20
```

## Output

**stdout** — JSON lines, one `kernel` record per timed pass carrying the
accumulated GPU event time (`time_ms`) of all stencil steps in that pass plus
host wall-clock time (`host_time_ms`), then a `summary` whose `total_time_ms`
is the sum of all recorded GPU event times. Effective bandwidth
(3 × cells × 4 bytes × steps / time) is derived downstream:
```
{"type":"kernel","name":"hotspot_kernel","iter":0,"time_ms":8.765400,"host_time_ms":8.990000,"params":{"grid_size":512,"block_size":16,"num_iterations":10}}
...
{"type":"summary","total_time_ms":61.357800}
```

**stderr** — human-readable results:
```
Device: Apple M4 Pro
Grid: 512×512  |  Block: 16×16  |  Steps/launch: 10

pass 0: hotspot_kernel gpu = 8.765 ms, host = 8.990 ms
...
Total GPU time: 61.358 ms over 7 passes (10 steps/pass)
```

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `rodinia_hotspot.hip`
- **Metal**: compiled with `clang++` from `rodinia_hotspot_metal.mm`; the Metal shader is loaded at runtime from the same directory as the binary
- Input data is generated synthetically — no external input files required
