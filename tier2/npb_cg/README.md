# tier2/npb_cg — NPB Conjugate Gradient

Benchmarks a sparse iterative Conjugate Gradient (CG) solver from the NAS Parallel Benchmarks. Solves Ax = b where A is a synthetic sparse matrix in CSR format. Emits per-solve, per-kernel GPU times; SpMV bandwidth and solve throughput are derived downstream.

## Algorithm

- **N×N** sparse matrix with configurable nonzeros per row (default: 7)
- Diagonally-dominant matrix ensures convergence
- CG iteration consists of:
  1. **SpMV** — sparse matrix-vector multiply `Ap = A * p` (CSR format)
  2. **Dot product** — `pAp = dot(p, Ap)` with block-level reduction
  3. **AXPY** — `x = x + alpha * p`, `r = r - alpha * Ap`
  4. **Dot product** — `rr_new = dot(r, r)` for convergence check
  5. **Scale + AXPY** — direction update `p = r + beta * p`
- One full CG solve (default: 25 iterations) is timed per execution;
  repetition across fresh processes is the harness's job (`gpubench run --reps N`)

## Parameters

| Env var | Default | Description |
|---------|---------|-------------|
| `BENCH_PARAM_matrix_size` | 50000 | Matrix rows/columns |
| `BENCH_PARAM_block_size` | 256 | Threads per block |
| `BENCH_PARAM_max_cg_iterations` | 25 | CG iterations per solve |
| `BENCH_PARAM_nonzeros_per_row` | 7 | Nonzeros per matrix row |
| `BENCH_PARAM_vector_add_warmup` | true | Cold-cache calibration: boost the clock and evict the LLC before the timed solve; set `false` to time as-is |

## Files

| File | Description |
|------|-------------|
| `npb_cg.hip` | HIP/ROCm implementation |
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
./npb_cg                                        # defaults: N=50000
BENCH_PARAM_matrix_size=100000 ./npb_cg         # override via env
./npb_cg --size 100000                           # override via CLI
```

## Output

**stdout** — JSON-lines. The single timed CG solve emits one record per kernel
carrying that kernel's accumulated GPU event time (`time_ms`) and host
wall-clock time (`host_time_ms`) within the solve; the summary's
`total_time_ms` is the sum of all recorded GPU event times. Derived metrics
(SpMV bandwidth, avg iteration time) are computed downstream:
```json
{"type":"kernel","name":"spmv_kernel","iter":0,"time_ms":1.234000,"host_time_ms":1.301000,"params":{"matrix_size":50000,...}}
{"type":"kernel","name":"dot_product_kernel","iter":0,"time_ms":1.234000,"host_time_ms":1.310000,"params":{...}}
{"type":"kernel","name":"axpy_kernel","iter":0,"time_ms":1.234000,"host_time_ms":1.298000,"params":{...}}
{"type":"kernel","name":"scale_kernel","iter":0,"time_ms":0.412000,"host_time_ms":0.450000,"params":{...}}
{"type":"summary","total_time_ms":29.081000}
```

**stderr** — human-readable timing and verification

## Verification

- Computes residual `||b - Ax|| / ||b||` after CG solve
- Reports PASS if relative residual < 1.0 (coarse check — CG with few iterations may not fully converge); on FAIL the binary exits with a non-zero status

## Platform Notes

- **CUDA/ROCm**: Uses `cudaEvent` / `hipEvent` for GPU timing; shared memory reduction for dot products
