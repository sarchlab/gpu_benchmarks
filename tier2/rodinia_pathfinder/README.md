# rodinia_pathfinder — Minimum-Cost Path Through a 2D Grid

Rodinia PathFinder benchmark: computes the minimum-cost path through a 2D grid
from top to bottom using dynamic programming with double-buffered row sweeps on GPU.

## Algorithm

- Grid of `ROWS × COLS` integer costs (randomly initialized to `[0, 10)`)
- Initialize `src[col] = wall[0][col]` (first row)
- For each row `t` from `1` to `ROWS-1`:
  - Each GPU thread (one per column) computes:
    ```
    dst[col] = wall[t][col] + min(src[col-1], src[col], src[col+1])
    ```
  - Swap `src` ↔ `dst` (double buffering)
- The minimum value in the final `src` array is the overall minimum path cost

## Files

| File | Description |
|------|-------------|
| `rodinia_pathfinder.hip` | HIP/CUDA implementation (NVIDIA + AMD) |
| `rodinia_pathfinder.metal` | Metal compute shader (Apple) |
| `rodinia_pathfinder_metal.mm` | ObjC++ Metal host (Apple) |
| `Makefile` | Platform auto-detect build system |
| `README.md` | This file |

## Build

```bash
# Auto-detect platform
make

# Explicit platform
make PLATFORM=cuda    # NVIDIA CUDA
make PLATFORM=rocm    # AMD ROCm/HIP
make PLATFORM=metal   # Apple Metal

# Clean
make clean
```

**Requirements:**
- CUDA: NVIDIA CUDA toolkit + `nvcc`
- ROCm: AMD ROCm 5.x+ + `hipcc`
- Metal: macOS 12+ + Xcode command-line tools

## Run

```bash
./rodinia_pathfinder [--rows R] [--cols C]
```

| Option / Env var | Default | Description |
|--------|---------|-------------|
| `--rows R` / `BENCH_PARAM_rows` | 100000 | Number of grid rows |
| `--cols C` / `BENCH_PARAM_cols` | 100 | Number of grid columns |
| `BENCH_PARAM_block_size` | 256 | Threads per block (descriptor) |
| `BENCH_PARAM_num_warmup` | 3 | Warmup grid sweeps (untimed) |
| `BENCH_PARAM_num_iters` | 7 | Timed grid sweeps, each recorded |

## Output

**stdout** — JSON lines: one `kernel` record per timed sweep (the GPU event
time covers the whole rows−1 kernel-launch chain of that sweep), then a
`summary` whose `total_time_ms` is the sum of all recorded GPU event times:

```
{"type":"kernel","name":"dynproc_kernel","iter":0,"time_ms":45.231800,"host_time_ms":45.610000,"params":{"cols":100,"rows":100000,"block_size":256}}
...
{"type":"summary","total_time_ms":316.622600}
```

Derived metrics (GB/s) are computed downstream from `time_ms` and params.

Informational output (stderr):
- GPU device name
- Grid dimensions
- Verification result (against small CPU reference)
- Per-sweep timing lines
- Minimum cost in the final row

## Performance Notes

- Memory access pattern: each row sweep reads `COLS` ints (src) + `COLS` ints (wall),
  writes `COLS` ints (dst) → `3 × ROWS × COLS × 4` bytes per sweep
- The `src` row is re-uploaded before every sweep (untimed), so each recorded
  sweep performs an identical traversal
- The rows−1 dependent kernel launches are timed as a single event bracket per
  sweep (no interleaved transfers), preserving launch pipelining
- The kernel is bandwidth-bound with minimal compute per element
- Metal implementation batches `1000` row dispatches per command buffer to reduce
  host-side overhead
- For very large grids (e.g., 1M rows), reduce `--rows` or increase batch size in code

## References

- [Rodinia Benchmark Suite](https://rodinia.cs.virginia.edu/)
- S. Che et al., "Rodinia: A benchmark suite for heterogeneous computing," IISWC 2009
