# tier2/npb_mg — NPB Multi-Grid

Benchmarks a V-cycle multigrid solver from the NAS Parallel Benchmarks. Solves a 3D Poisson-like equation on a synthetic grid using Jacobi smoothing with restriction and prolongation operators. The solve runs exactly once per execution and emits per-kernel GPU/host times; throughput (Mcells/sec) is derived downstream.

## Provenance & fidelity

Derived from the [NAS Parallel Benchmarks](https://www.nas.nasa.gov/software/npb.html) — the `MG` (Multi-Grid) kernel. Independent reimplementation for portable measurement, **not** upstream source, and **not** a faithful reproduction. It differs as follows:

- Simple 7-point geometric multigrid vs NPB MG's 4-coefficient stencil V-cycle. (A residual-sign bug that prevented convergence has been fixed.)

Read this as "a multigrid benchmark in the spirit of NPB," not NPB's `MG`.

## Algorithm

- **N×N×N** 3D grid (default: 64³) with Dirichlet boundary conditions
- V-cycle multigrid hierarchy: N → N/2 → N/4 → ... → 4
- Each V-cycle consists of:
  1. **Pre-smooth** — Jacobi iterations on current level
  2. **Residual** — Compute r = rhs − A·u (discrete Laplacian)
  3. **Restrict** — Downsample residual to coarser grid (2:1 averaging)
  4. **Reset coarse correction** — Explicit `coarse_zero` kernel before descending
  5. **Coarse solve** — Extra smoothing on coarsest level
  6. **Prolong** — Interpolate correction back to fine grid (injection)
  7. **Post-smooth** — Jacobi iterations after correction
- Default: 5 V-cycles per solve, 3 smooth steps per level. One full multigrid solve from a zeroed solution is timed per execution; repetition across fresh processes is the harness's job (`gpubench run --reps N`)

## Files

| File | Description |
|------|-------------|
| `npb_mg.hip` | HIP/ROCm implementation |
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
| `BENCH_PARAM_vector_add_warmup` | Cold-cache calibration: boost the clock and evict the LLC before the timed solve; set `false` to time as-is | true |

## Output

**stdout** — JSON-lines: for the single timed solve, one `{"type":"kernel",...}` record per kernel (`smooth_kernel`, `restrict_kernel`, `prolong_kernel`, `residual_kernel`, `coarse_zero`) carrying that kernel's GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`) accumulated over the solve, then a summary whose `total_time_ms` is the sum of all recorded GPU event times:
```json
{"type":"kernel","name":"smooth_kernel","iter":0,"time_ms":5.234,"host_time_ms":5.401,"params":{"grid_size":64,...}}
{"type":"kernel","name":"restrict_kernel","iter":0,"time_ms":0.412,"host_time_ms":0.498,"params":{...}}
{"type":"kernel","name":"prolong_kernel","iter":0,"time_ms":0.377,"host_time_ms":0.451,"params":{...}}
{"type":"kernel","name":"residual_kernel","iter":0,"time_ms":1.902,"host_time_ms":1.977,"params":{...}}
{"type":"kernel","name":"coarse_zero","iter":0,"time_ms":0.061,"host_time_ms":0.104,"params":{...}}
{"type":"summary","total_time_ms":42.161}
```

Derived metrics such as Mcells/sec are computed downstream from `time_ms` and params. Every piece of device work inside the timed V-cycle is bracketed and emitted so `total_time_ms` reflects the full algorithm: `residual_kernel` (forms the residual that drives restriction; matching upstream NPB MG, which times it), and `coarse_zero` (the per-V-cycle zeroing of the coarse-grid correction before coarse smoothing).

**stderr** — human-readable timing and verification

## Verification

- Computes residual norm `||rhs − A·u|| / ||rhs||` after multigrid solve
- Reports PASS if relative residual < 10.0 (multigrid with few V-cycles provides approximate solution); on FAIL the binary exits with a non-zero status

## Platform Notes

- **CUDA/ROCm**: Uses `cudaEvent` / `hipEvent` for GPU timing; shared memory reduction for norm computation
- Grid size is automatically rounded up to the nearest power of 2 for proper multigrid hierarchy

## Algorithmic reset timing

`coarse_zero` is an explicit GPU kernel that writes every correction cell before
the next level is smoothed. It replaces a runtime memset previously reported
under this kernel name. For L levels (`N, N/2, ..., 4`), each solve launches
`num_vcycles * (L - 1)` resets. Every reset has its own event bracket and its
time contributes to the existing `coarse_zero` accumulator and summary.

Initial solution/RHS setup zeros remain outside timing and before the optional
entry warmer. No warmer runs between levels, kernels or V-cycles. Recollect
old measurements because the implementation of the reset has changed. This
repair preserves the current numerical operator and sweep; it does not resolve
the NAS identity differences under R-09 or the weak residual gate under R-13.
The three separate verification launches remain outside reported time, and
general trace-region/artifact qualification remains R-23/R-24.

`make test PLATFORM=cuda` (or `rocm`) checks the actual reset kernel against
poisoned inputs and untouched guards, from partial blocks through the largest
declared coarse grid. CI compiles this test; execution requires a GPU. The
shared [reset ordering test](../../tests/algorithmic_zero_order.py) verifies
actual NVIDIA Nsight launch/event order and output agreement; see the
[test instructions](../../tests/README.md). AMD runtime qualification remains
R-19. Profiled checks are functional evidence, not performance measurements.
