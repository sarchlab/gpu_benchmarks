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
{"type":"kernel","name":"mean_kernel","iter":0,"time_ms":0.123456,"host_time_ms":0.140000,"params":{"size":1024,"block_size":256,"verification":"full_cpu_v1"}}
{"type":"kernel","name":"stddev_kernel","iter":0,"time_ms":0.234567,"host_time_ms":0.250000,"params":{"size":1024,"block_size":256,"verification":"full_cpu_v1"}}
{"type":"kernel","name":"normalize_kernel","iter":0,"time_ms":0.345678,"host_time_ms":0.360000,"params":{"size":1024,"block_size":256,"verification":"full_cpu_v1"}}
{"type":"kernel","name":"correlation_kernel","iter":0,"time_ms":1.456789,"host_time_ms":1.480000,"params":{"size":1024,"block_size":256,"verification":"full_cpu_v1"}}
{"type":"summary","total_time_ms":2.160490}
```

**stderr** — human-readable:
```
Device: NVIDIA GeForce RTX 4090 (id=0)
Data matrix: 1024x1024

mean=0.1235  stddev=0.2346  normalize=0.3457  correlation=1.4568 (gpu ms)
Total GPU time: 2.160 ms
```

## Measured-result verification

Every ordinary run checks all values in all four retained measured outputs:
column means, standard deviations, normalized data and the complete correlation
matrix. An independent double CPU reference starts from the original input and
computes all four stages without using GPU intermediates. It preserves the
standard-deviation guard's exact float threshold (`s < 1e-12f` becomes 1) and
the forced correlation diagonal, including for constant columns.

A value passes only when actual, reference and error are finite and
`abs(actual - reference) <= 1e-5 * max(1, abs(reference))`. Every correlation
diagonal value must also equal exactly 1.0: that is a literal GPU output,
so even a one-ULP diagonal corruption fails. Other errors inside the tolerance
are intentionally accepted. Qualification includes independent long-double
references, rectangular inputs, zero/constant/mixed columns, threshold controls
and faults in every retained GPU output. This empirically qualified tolerance
is not a universal error bound; AMD and further edge-input qualification remain
separate requirements.

All checking occurs after the four timed stages. Once the reference has
consumed the original input, its host buffer is reused for four result copies
(two vectors and two matrices). The checker counts every mismatch, prints at
most ten details to stderr and returns nonzero on failure. Every kernel record
includes `verification="full_cpu_v1"`; there is no bypass or separate GPU
verification launch. Inputs, GPU kernels, geometry, initial upload, warmer and
timed events are unchanged.

At N=4096, the independent double reference adds 256 MiB for the two matrices
and 64 KiB for the two vectors, alongside the existing 64 MiB input buffer.
Result copies total 128 MiB plus 32 KiB and reuse that input buffer sequentially
(the driver fixes M=N). Complete CPU product work adds process time outside GPU
timing. Host admission, other-host budgets, AMD and full upstream-identity
qualification remain open.

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc`; 4 separate kernel launches per execution
- The correlation kernel uses 16×16 tiled shared-memory to compute `data^T * data`
