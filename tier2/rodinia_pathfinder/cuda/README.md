# rodinia_pathfinder — Minimum-Cost Path Through a 2D Grid (native CUDA)

Faithful port of the Rodinia PathFinder benchmark: the device kernel
(`dynproc_kernel`) and the host launch geometry are the **original** Rodinia 3.1
CUDA source; only the outer driver loop is restructured for single-pass timing.

## Provenance & fidelity

Derived from the [Rodinia benchmark suite](http://lava.cs.virginia.edu/Rodinia/) — the `pathfinder` workload. Independent reimplementation, spot-verified to reproduce Rodinia's algorithm faithfully.

## Algorithm

Dynamic-programming shortest descent through a `ROWS × COLS` integer cost grid.
Starting from row 0, each subsequent row's accumulated minimum cost is derived
from the three cells above it:

```
next[n] = wall[t][n] + min(cur[n-1], cur[n], cur[n+1])
```

The original kernel uses **pyramidal / ghost-zone temporal blocking** (identical
in spirit to Rodinia Hotspot): a 1-D block of `BLOCK_SIZE = 256` threads loads
one row into shared memory and advances `pyramid_height` rows locally before
writing back, using a border (halo) of `pyramid_height` columns so neighbouring
blocks stay consistent. A block computes a non-overlapping "small block" of
`BLOCK_SIZE − 2·pyramid_height` columns per launch.

The benchmark runs one full top-to-bottom traversal of all `ROWS` — **exactly
once** — driven by the original `calc_path` loop
(`for t = 0; t < rows-1; t += pyramid_height`). Every pyramidal launch is
timed and accumulated into a single `dynproc_kernel` record. With the defaults
(`rows=100`, `pyramid_height=20`) there are `ceil((100-1)/20) = 5` launches;
the last advances `min(20, rows-t-1)` rows. Repetition is the harness's job
(`gpubench run --reps N`).

## Default problem size

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--cols` / `BENCH_PARAM_cols` | 100000 | matches the original standard run |
| `--rows` / `BENCH_PARAM_rows` | 100 | wide/short grid, as in the original run |
| `--pyramid_height` / `BENCH_PARAM_pyramid_height` | 20 | as in the original run |
| `BENCH_PARAM_vector_add_warmup` | `true` | cold-cache calibration before the timed traversal |

`BLOCK_SIZE` is a fixed compile-time `256` in the original kernel; the emitted
`block_size` descriptor reflects that fixed value. Resolved `cols` and `rows`
must be positive, and `pyramid_height` must be in `1..127`. These checks run
before device access, including for environment overrides.

Admission also requires `9*rows <= INT_MAX`, so the generated costs in `0..9`
cannot overflow the integer path sums. The widest padded column coordinate,
including the final partial-height launch, must fit `int`. The existing kernel
and CPU recurrence are unchanged. A direct one-row input retains its existing
behavior without a traversal-kernel launch; the declared performance sweep uses at least 100 rows.

**[TODO for cuda]** The evidence in this paragraph was collected for [`../hip/`](../hip/); it has not yet been rerun for this CUDA build. These bounds establish arithmetic validity. The existing device-total-memory
and host-physical-memory checks still apply afterward; passing them does not
guarantee that a configuration fits currently available memory or finishes
within the collection budget. General capacity admission and supported H100
profiles remain under [R-15/R-22](https://github.com/sarchlab/gpu_benchmarks/issues/71).

## Files

| File | Description |
|------|-------------|
| `rodinia_pathfinder.cu` | CUDA source — (faithful kernel + host driver) |
| `Makefile` | nvcc build (`CUDA_ARCH`, default `native`) |
| `README.md` | This file |

## Build

```bash
make                     # -arch=native (builds for the GPU in this machine)
make CUDA_ARCH=sm_80     # A100
make CUDA_ARCH=sm_90     # H100
make clean
```

## Run

```bash
./rodinia_pathfinder [--cols C] [--rows R] [--pyramid_height P]
```

## Output

**stdout** — JSON lines: one `kernel` record (the GPU event time
is the sum of all pyramidal launches in the traversal), then a `summary` whose
`total_time_ms` is the recorded GPU event time:

```
{"type":"kernel","name":"dynproc_kernel","iter":0,"time_ms":...,"host_time_ms":...,"params":{"cols":100000,"rows":100,"pyramid_height":20,"block_size":256}}
{"type":"summary","total_time_ms":...}
```

**stderr** — device name, pyramid geometry, timing, and verification against a
CPU dynamic-programming reference over the full grid. On verification FAIL the
benchmark exits with a non-zero status.

## Data reset

`gpuWall` (the cost grid) is immutable and uploaded once. The row-0 cost vector
`gpuResult[0]` is seeded (untimed) before the timed traversal.

## References

- [Rodinia Benchmark Suite](https://rodinia.cs.virginia.edu/)
- S. Che et al., "Rodinia: A benchmark suite for heterogeneous computing," IISWC 2009
