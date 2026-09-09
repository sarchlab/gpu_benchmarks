# tier2/rodinia_gaussian — Gaussian Elimination (Rodinia)

Gaussian elimination with back-substitution to solve `Ax = b` for an `N×N` dense matrix.

## Provenance & fidelity

Derived from the [Rodinia benchmark suite](http://lava.cs.virginia.edu/Rodinia/) — the `gaussian` workload. Independent reimplementation for portable measurement, **not** upstream source, and **not** a faithful reproduction. It differs as follows:

- Correct Gaussian elimination, but the Fan2 thread→data mapping is coalesced, whereas Rodinia's signature is a famously uncoalesced Fan2 access pattern — the benchmark's characteristic memory behavior is optimized away. Math is correct; the memory-access signature differs.

Read this as "a Gaussian elimination benchmark in the spirit of Rodinia," not Rodinia's `gaussian`.

## Algorithm

Forward elimination using two GPU kernels per pivot step `k = 0..N-2`:

1. **fan1** — Compute multipliers for pivot column `k`:
   ```
   m[i][k] = a[i][k] / a[k][k]   for i = k+1..N-1
   ```
2. **fan2** — Eliminate pivot column from submatrix:
   ```
   a[i][j] -= m[i][k] * a[k][j]  for i,j > k
   b[i]    -= m[i][k] * b[k]      for i > k
   ```

After forward elimination, **back-substitution** is performed on the CPU:
```
x[i] = (b[i] - sum_{j>i} a[i][j]*x[j]) / a[i][i]
```

The matrix `A` is initialized as diagonally dominant (`A_ii = N`, `A_ij ~ Uniform[0, 0.9]`) to ensure numerical stability without partial pivoting.

## Files

| File | Description |
|------|-------------|
| `rodinia_gaussian.hip` | HIP/CUDA source — fan1 + fan2 kernels, self-contained |
| `Makefile` | Auto-detecting build (ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
./rodinia_gaussian                         # defaults: N=512
./rodinia_gaussian --size 256
BENCH_PARAM_size=1024 ./rodinia_gaussian
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_size` | 512 | Matrix dimension N |
| `BENCH_PARAM_vector_add_warmup` | `true` | Cold-cache calibration before the timed pass |

`BENCH_PARAM_verify=false` skips CPU verification — an undeclared debug knob
(not in `params.json`, not a sweep parameter). When verification runs and
FAILs, the benchmark exits with a non-zero status.

The forward elimination runs **exactly once**; repetition is the harness's
job (`gpubench run --reps N`).

Each kernel record includes the JSON boolean `verification_enabled`, resolved
from the existing debug flag. Verification is enabled by default; only the
case-sensitive values `false`, `0`, and `skip` disable it. An inherited harness
environment can set this unswept control. The boolean reports whether checks
are enabled, not whether they passed: verification failures still exit nonzero.
A successful process with `verification_enabled: false` is unchecked and must
not be treated as verified collection evidence. Older records missing the field
have unknown verification status.

## Output

**stdout** — JSON lines: one `kernel` record per kernel
(`fan1`, `fan2`) with that kernel's accumulated GPU event time (`time_ms`)
and host time (`host_time_ms`) across all `N-1` pivot steps, then a summary
whose `total_time_ms` is the sum of all recorded GPU event times:
```json
{"type":"kernel","name":"fan1","iter":0,"time_ms":10.123,"host_time_ms":12.456,"params":{"size":512,"verification_enabled":true}}
{"type":"kernel","name":"fan2","iter":0,"time_ms":35.000,"host_time_ms":38.100,"params":{"size":512,"verification_enabled":true}}
{"type":"summary","total_time_ms":45.123}
```

**stderr** — human-readable details:
```
Device: NVIDIA GeForce RTX 4090 (id=0)
Matrix size: 512×512

fan1 gpu = 10.123 ms, fan2 gpu = 35.000 ms
Total GPU time: 45.123 ms over 1 pass
Verification: rel_err = 1.23e-06 — PASS
```

## Metrics

- **Time**: GPU event time for forward elimination (fan1 + fan2 kernels across all `N-1` pivot steps)
- **GFLOPS**: `(2/3 × N³) / time_s / 1e9` — derived downstream from `time_ms` and params
- **Verification**: relative residual `‖Ax − b‖ / ‖b‖ < 1e-4`

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `rodinia_gaussian.hip`; uses `hipEvent_t` for timing
- Input data is generated synthetically — no external input files required
- Back-substitution runs on the CPU after the GPU forward elimination phase
