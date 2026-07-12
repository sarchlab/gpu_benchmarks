# tier2/polybench_correlation — Correlation Matrix (PolyBench)

Computes the **correlation matrix** from a data matrix of **M samples × N features** (M = N). The computation involves four GPU kernels executed in sequence: column means, column standard deviations, data normalization, and the correlation matrix itself (a transposed matrix multiply on the normalized data).

## Provenance & fidelity

Derived from the [PolyBench/GPU benchmark suite](https://sourceforge.net/projects/polybench/) — the `correlation` workload. Independent reimplementation for portable measurement, **not** upstream source, and **not** a faithful reproduction. It differs as follows:

- Tiled full-matrix correlation vs PolyBench's triangular naive form.

It keeps the same four-stage structure (mean, stddev, normalize, correlation), but the correlation stage is a generic tiled shared-memory matrix multiply rather than the upstream kernel. Read this as "a correlation benchmark in the spirit of PolyBench," not PolyBench's `correlation`.

## Algorithm

1. **mean_kernel** — compute the mean of each column: `mean[j] = Σ_i data[i][j] / M`
2. **stddev_kernel** — compute the standard deviation of each column: `stddev[j] = sqrt(Σ_i (data[i][j] - mean[j])² / M)`
3. **normalize_kernel** — center and scale each element: `data[i][j] = (data[i][j] - mean[j]) / (sqrt(M) * stddev[j])`
4. **correlation_kernel** — compute `corr = data^T * data` (NxN), with diagonal forced to 1.0

The correlation kernel uses tiled shared-memory multiplication (16×16 tiles).

## Files

| File | Description |
|------|-------------|
| `polybench_correlation.hip` | HIP/ROCm implementation |
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
./polybench_correlation                                  # defaults: 1024×1024
BENCH_PARAM_size=2048 ./polybench_correlation            # 2048×2048
BENCH_PARAM_block_size=128 ./polybench_correlation       # custom block size
```

Additional environment variables:

| Env var | Default | Description |
|---------|---------|-------------|
| `BENCH_PARAM_vector_add_warmup` | true | Vector-add warmer (clock boost + LLC eviction) before the timed pass |

The four-kernel pass runs exactly once per execution; repetition is the
harness's job (`gpubench run --reps N`). Because `normalize_kernel` mutates
the data matrix in place, the pristine input is re-uploaded (untimed) at the
start of the pass.

## Output

**stdout** — JSON-lines. One record per kernel carrying the
GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`), then a
summary whose `total_time_ms` is the sum of the four recorded GPU event times
(no metrics array; derived metrics are computed downstream):
```json
{"type":"kernel","name":"mean_kernel","iter":0,"time_ms":0.123456,"host_time_ms":0.140000,"params":{"size":1024,"block_size":256}}
{"type":"kernel","name":"stddev_kernel","iter":0,"time_ms":0.234567,"host_time_ms":0.250000,"params":{"size":1024,"block_size":256}}
{"type":"kernel","name":"normalize_kernel","iter":0,"time_ms":0.345678,"host_time_ms":0.360000,"params":{"size":1024,"block_size":256}}
{"type":"kernel","name":"correlation_kernel","iter":0,"time_ms":1.456789,"host_time_ms":1.480000,"params":{"size":1024,"block_size":256}}
{"type":"summary","total_time_ms":2.160490}
```

**stderr** — human-readable:
```
Device: NVIDIA GeForce RTX 4090 (id=0)
Data matrix: 1024x1024

mean=0.1235  stddev=0.2346  normalize=0.3457  correlation=1.4568 (gpu ms)
Total GPU time: 2.160 ms
```

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc`; 4 separate kernel launches per execution
- The correlation kernel uses 16×16 tiled shared-memory to compute `data^T * data`
