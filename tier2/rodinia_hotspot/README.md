# tier2/rodinia_hotspot — 2D Thermal Simulation (Rodinia Hotspot)

Transient thermal simulation of a chip: an iterative 2D structured-grid stencil
where each cell's temperature is updated from its four neighbors (N/S/E/W),
the local power density, and thermal resistances. Classic structured-grid
benchmark.

Faithful port of the [Rodinia benchmark suite](https://rodinia.cs.virginia.edu/)
3.1 CUDA Hotspot: the device kernel (`calculate_temp`) and its **pyramidal /
ghost-zone temporal-blocking** scheme are the original; only the host driver is
restructured for single-pass timing and JSON output.

## Algorithm

The per-cell update is the classic hotspot stencil:
```
T_new[i][j] = T[i][j] + step_div_cap * (
    power[i][j]
    + (T[i-1][j] + T[i+1][j] - 2*T[i][j]) * Ry_1    // N-S neighbors
    + (T[i][j-1] + T[i][j+1] - 2*T[i][j]) * Rx_1    // E-W neighbors
    + (AMB_TEMP  - T[i][j]) * Rz_1                  // heat to substrate
)
```
with clamped (no-flux) boundary conditions.

Rather than one kernel launch per time step, the original **pyramidal**
scheme advances `pyramid_height` time steps per launch: each 16×16 block
loads an expanded tile (the pyramid base, grown by `EXPAND_RATE = 2` cells
per border per step) into shared memory once, iterates the stencil
`pyramid_height` times in shared memory, and writes back only the valid
inner region. Ping-pong buffers alternate between launches; the whole
simulation runs `ceil(total_iterations / pyramid_height)` launches of
`calculate_temp`.

The benchmark runs **one full simulation of `total_iterations` time steps,
exactly once**; repetition is the harness's job (`gpubench run --reps N`).

## Parameters

| Env var | Default | Description |
|---------|---------|-------------|
| `BENCH_PARAM_grid_size` (or `--grid_size N`) | 1024 | Grid dimension (N×N), scaling axis |
| `BENCH_PARAM_total_iterations` (or `--total_iterations T`) | 360 | Total thermal time steps simulated |
| `BENCH_PARAM_pyramid_height` (or `--pyramid_height P`) | 2 | Time steps advanced per pyramidal launch (must be < BLOCK_SIZE/2 = 8) |
| `BENCH_PARAM_vector_add_warmup` | `true` | Cold-cache calibration before the timed simulation |

The thread-block size is **fixed** at `BLOCK_SIZE = 16` (16×16), as in the
original — not a knob.

## Files

| File | Description |
|------|-------------|
| `rodinia_hotspot.hip` | HIP source — original pyramidal kernel + self-contained host driver |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./rodinia_hotspot                                         # defaults: 1024×1024, 360 steps, pyramid 2
./rodinia_hotspot --grid_size 2048
./rodinia_hotspot --grid_size 512 --total_iterations 360 --pyramid_height 4
```

## Output

**stdout** — JSON lines: one `kernel` record for `calculate_temp` carrying the
accumulated GPU event time (`time_ms`) of all pyramidal launches in the
simulation plus host wall-clock time (`host_time_ms`), then a `summary` whose
`total_time_ms` is the sum of the recorded GPU event times. Effective
bandwidth is derived downstream:
```
{"type":"kernel","name":"calculate_temp","iter":0,"time_ms":8.765400,"host_time_ms":8.990000,"params":{"grid_size":1024,"total_iterations":360,"pyramid_height":2,"block_size":16}}
{"type":"summary","total_time_ms":8.765400}
```

**stderr** — human-readable results:
```
Device: NVIDIA GeForce RTX 4090 (id=0)
Grid: 1024×1024  |  pyramid_height: 2  |  total_iterations: 360
Grid of blocks: 86×86 (BLOCK_SIZE=16)

calculate_temp gpu = 8.765 ms, host = 8.990 ms
Total GPU time: 8.765 ms over 1 pass (360 steps/pass)
```

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `rodinia_hotspot.hip`
- Input data is generated synthetically — no external input files required
