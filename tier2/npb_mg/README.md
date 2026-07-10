# tier2/npb_mg — NPB Multi-Grid

Benchmarks a V-cycle multigrid solver from the NAS Parallel Benchmarks. Solves a 3D Poisson-like equation on a synthetic grid using Jacobi smoothing with restriction and prolongation operators. Emits per-pass GPU/host kernel times; throughput (Mcells/sec) is derived downstream.

## Algorithm

- **N×N×N** 3D grid (default: 64³) with Dirichlet boundary conditions
- V-cycle multigrid hierarchy: N → N/2 → N/4 → ... → 4
- Each V-cycle consists of:
  1. **Pre-smooth** — Jacobi iterations on current level
  2. **Residual** — Compute r = rhs − A·u (discrete Laplacian)
  3. **Restrict** — Downsample residual to coarser grid (2:1 averaging)
  4. **Coarse solve** — Extra smoothing on coarsest level
  5. **Prolong** — Interpolate correction back to fine grid (injection)
  6. **Post-smooth** — Jacobi iterations after correction
- Default: 5 V-cycles per solve, 3 smooth steps per level, 3 warmup + 7 timed passes (each pass = one full multigrid solve from a zeroed solution)

## Files

| File | Description |
|------|-------------|
| `npb_mg.hip` | HIP/ROCm implementation |
| `npb_mg_metal.mm` | Metal host code (Objective-C++) |
| `npb_mg.metal` | Metal compute shaders |
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
./npb_mg                                        # defaults: N=64
BENCH_PARAM_grid_size=128 ./npb_mg              # override via env
./npb_mg --size 128                              # override via CLI
```

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `BENCH_PARAM_grid_size` | Grid dimension N (rounded to power of 2) | 64 |
| `BENCH_PARAM_block_size` | Threads per block | 256 |
| `BENCH_PARAM_num_vcycles` | V-cycles per solve | 5 |
| `BENCH_PARAM_num_smooth_steps` | Smooth steps per level per direction | 3 |
| `BENCH_PARAM_warmup` | Warmup passes (untimed) | 3 |
| `BENCH_PARAM_iteration` | Timed passes, each recorded | 7 |

## Output

**stdout** — JSON-lines: per timed pass, one `{"type":"kernel",...}` record per kernel (`smooth_kernel`, `restrict_kernel`, `prolong_kernel`) carrying that kernel's GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`) accumulated over the pass, then a summary whose `total_time_ms` is the sum of all recorded GPU event times:
```json
{"type":"kernel","name":"smooth_kernel","iter":0,"time_ms":5.234,"host_time_ms":5.401,"params":{"grid_size":64,...}}
{"type":"kernel","name":"restrict_kernel","iter":0,"time_ms":0.412,"host_time_ms":0.498,"params":{...}}
{"type":"kernel","name":"prolong_kernel","iter":0,"time_ms":0.377,"host_time_ms":0.451,"params":{...}}
{"type":"summary","total_time_ms":42.161}
```

Derived metrics such as Mcells/sec are computed downstream from `time_ms` and params. The auxiliary `residual_kernel` runs inside each V-cycle but (as in the original benchmark) is not timed or emitted.

**stderr** — human-readable timing and verification

## Verification

- Computes residual norm `||rhs − A·u|| / ||rhs||` after multigrid solve
- Reports PASS if relative residual < 10.0 (multigrid with few V-cycles provides approximate solution)

## Platform Notes

- **CUDA/ROCm**: Uses `cudaEvent` / `hipEvent` for GPU timing; shared memory reduction for norm computation
- **Metal**: Loads `npb_mg.metal` shader at runtime; uses `mach_absolute_time` for timing
- Grid size is automatically rounded up to the nearest power of 2 for proper multigrid hierarchy
