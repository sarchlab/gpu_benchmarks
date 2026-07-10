# tier2/rodinia_gaussian — Gaussian Elimination (Rodinia)

Gaussian elimination with back-substitution to solve `Ax = b` for an `N×N` dense matrix. Derived from the [Rodinia benchmark suite](https://rodinia.cs.virginia.edu/) Gaussian workload.

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
| `rodinia_gaussian.metal` | Metal compute shaders — fan1_kernel + fan2_kernel |
| `rodinia_gaussian_metal.mm` | ObjC++ Metal host code |
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
./rodinia_gaussian                         # defaults: N=512
./rodinia_gaussian --size 256
BENCH_PARAM_iteration=5 ./rodinia_gaussian --size 1024
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_PARAM_size` | 512 | Matrix dimension N |
| `BENCH_PARAM_block_size` | 256 | Threads per block (echoed in records) |
| `BENCH_PARAM_verify` | true | Verification mode (`true`, `false`, `skip`) |
| `BENCH_PARAM_warmup` | 3 | Warmup elimination passes (not recorded) |
| `BENCH_PARAM_iteration` | 7 | Timed elimination passes, each recorded |

## Output

**stdout** — JSON lines: per timed pass, one `kernel` record per kernel
(`fan1`, `fan2`) with that kernel's accumulated GPU event time (`time_ms`)
and host time (`host_time_ms`) across all `N-1` pivot steps, then a summary
whose `total_time_ms` is the sum of all recorded GPU event times:
```json
{"type":"kernel","name":"fan1","iter":0,"time_ms":10.123,"host_time_ms":12.456,"params":{"size":512,"block_size":256,"verify":"true"}}
{"type":"kernel","name":"fan2","iter":0,"time_ms":35.000,"host_time_ms":38.100,"params":{"size":512,"block_size":256,"verify":"true"}}
{"type":"summary","total_time_ms":315.861}
```

**stderr** — human-readable details:
```
Device: Apple M4 Pro
Matrix size: 512×512

pass 0: fan1 gpu = 10.123 ms, fan2 gpu = 35.000 ms
...
Total GPU time: 315.861 ms over 7 passes
Verification: rel_err = 1.23e-06 — PASS
```

## Metrics

- **Time**: GPU event time for forward elimination (fan1 + fan2 kernels across all `N-1` pivot steps), recorded per pass
- **GFLOPS**: `(2/3 × N³) / time_s / 1e9` — derived downstream from `time_ms` and params
- **Verification**: relative residual `‖Ax − b‖ / ‖b‖ < 1e-4`

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `rodinia_gaussian.hip`; uses `hipEvent_t` for timing
- **Metal**: compiled with `clang++` from `rodinia_gaussian_metal.mm`; the Metal shader (`rodinia_gaussian.metal`) is loaded at runtime from the same directory as the binary; uses `mach_absolute_time()` for timing
- Input data is generated synthetically — no external input files required
- Back-substitution runs on the CPU after the GPU forward elimination phase
