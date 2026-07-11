# tier2/altis_cfd — Computational Fluid Dynamics (Altis)

Benchmarks a compressible Euler equations solver on an unstructured mesh using a finite-volume method. Each GPU thread processes one mesh cell, computing fluxes from neighbor data and performing Runge-Kutta time integration. One RK time step (2 stages) runs exactly once per execution; per-kernel times are emitted and Mcells/sec is derived downstream.

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
./altis_cfd                              # defaults: N=97000
./altis_cfd --size 50000                 # fewer cells
BENCH_PARAM_size=50000 ./altis_cfd       # env vars override argv
```

Environment variables (override argv):

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_size` | 97000 | Number of mesh cells |
| `BENCH_PARAM_block_size` | 256 | Threads per block |
| `BENCH_PARAM_vector_add_warmup` | true | Cold-cache calibration: boost the clock and evict the LLC before the timed step; set `false` to time as-is |

## Output

**stdout** — JSON lines: for the single timed RK step, one `{"type":"kernel",...}`
record per kernel (`compute_flux_kernel`, `rk_update_kernel`) with that kernel's
GPU event time accumulated over the two RK stages (`time_ms`) and host
wall-clock time (`host_time_ms`), followed by a
`{"type":"summary","total_time_ms":...}` line whose value is the sum of the
recorded GPU event times. Derived metrics (Mcells/sec) are computed
downstream from `time_ms` and the params.

**stderr** — human-readable results:
```
Device: AMD Radeon RX 7900 XTX (id=0)
CFD Euler solver  |  Cells: 97000  |  Neighbors: 4

compute_flux gpu = 1.234 ms, rk_update gpu = 0.456 ms
Total GPU time: 1.690 ms
PASS
```

## Verification

- Flux computation for first 256 cells is compared against a CPU reference
- Reports PASS/FAIL with mismatch details (relative tolerance 0.1%); on FAIL the binary exits with a non-zero status

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `altis_cfd.hip`; uses Structure-of-Arrays layout for coalesced memory access
