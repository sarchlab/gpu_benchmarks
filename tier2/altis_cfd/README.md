# tier2/altis_cfd — Computational Fluid Dynamics (Altis)

Benchmarks a compressible Euler equations solver on an unstructured mesh using a finite-volume method. Each GPU thread processes one mesh cell, computing fluxes from neighbor data and performing Runge-Kutta time integration. Emits per-iteration kernel times; Mcells/sec is derived downstream.

Derived from the [Altis benchmark suite](https://github.com/utcs-scea/altis) CFD workload.

## Algorithm

- **N** mesh cells (default: N=97,000) with 4 neighbors each
- Conserved variables per cell: density (ρ), momentum (mx, my, mz), energy (e)
- Finite-volume method with Rusanov (local Lax-Friedrichs) numerical flux:
  1. **Flux computation** — each thread reads its cell and 4 neighbors, computes interface fluxes using Rusanov dissipation
  2. **Runge-Kutta update** — 2-stage scheme: U_new = rk_coeff × U_old + (1 − rk_coeff) × (U − dt/vol × flux)
- Synthetic unstructured mesh with random connectivity and face geometry
- Mcells/sec = N / time\_sec / 10⁶ (derived downstream from the emitted times and params)

## Files

| File | Description |
|------|-------------|
| `altis_cfd.hip` | HIP source — Euler solver kernels (flux + RK), verification |
| `altis_cfd.metal` | Metal compute shaders — flux and RK update kernels |
| `altis_cfd_metal.mm` | ObjC++ Metal host code |
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
./altis_cfd                              # defaults: N=97000
./altis_cfd --size 50000                 # fewer cells
BENCH_PARAM_num_iters=10 ./altis_cfd     # more timed passes
```

Environment variables (override argv):

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_size` | 97000 | Number of mesh cells |
| `BENCH_PARAM_block_size` | 256 | Threads per block |
| `BENCH_PARAM_precision` | float | Floating-point precision (half/float/double) |
| `BENCH_PARAM_num_warmup` | 3 | Untimed warmup passes |
| `BENCH_PARAM_num_iters` | 7 | Timed passes, each recorded individually |

## Output

**stdout** — JSON lines: per timed pass, one `{"type":"kernel",...}` record per
kernel (`compute_flux_kernel`, `rk_update_kernel`) with that pass's accumulated
GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`), followed
by a `{"type":"summary","total_time_ms":...}` line whose value is the sum of
all recorded GPU event times. Derived metrics (Mcells/sec) are computed
downstream from `time_ms` and the params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
CFD Euler solver  |  Cells: 97000  |  Neighbors: 4  |  Iterations: 3 warmup + 7 timed

pass 0: compute_flux gpu = 1.234 ms, rk_update gpu = 0.456 ms
...
Total GPU time: 12.345 ms over 7 passes
PASS
```

## Verification

- Flux computation for first 256 cells is compared against a CPU reference
- Reports PASS/FAIL with mismatch details (relative tolerance 0.1%)

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `altis_cfd.hip`; uses Structure-of-Arrays layout for coalesced memory access
- **Metal**: compiled with `clang++` from `altis_cfd_metal.mm`; the Metal shader (`altis_cfd.metal`) is loaded and compiled at runtime from the same directory as the binary
