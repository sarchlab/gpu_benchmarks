# polybench_mvt

PolyBench MVT benchmark: matrix-vector product and transpose.

Computes:
- **x1 = A · y1** — matrix-vector product
- **x2 = Aᵀ · y2** — transpose matrix-vector product

where A is an N×N matrix and x1, x2, y1, y2 are N-vectors.

## Provenance & fidelity

Derived from the [PolyBench/GPU](https://sourceforge.net/projects/polybench/) benchmark suite — the `mvt` workload. Independent reimplementation, spot-verified to reproduce PolyBench's algorithm faithfully.

## Algorithm

The MVT pass runs exactly once per execution, launching two GPU kernels:

1. **mvt_kernel1** — `x1 = A · y1`  
   Each thread computes one row: `x1[i] = Σⱼ A[i,j] * y1[j]`

2. **mvt_kernel2** — `x2 = Aᵀ · y2`  
   Each thread computes one column: `x2[j] = Σᵢ A[i,j] * y2[i]`

## Metric

**GB/s** (memory bandwidth) — matrix A is read twice (once per kernel).
The formula is applied downstream to the emitted times:

```
GB/s = (2 * N * N * 4 bytes) / time_s / 1e9   (derived downstream)
```

## Output

**stdout** — JSON lines: one `kernel` record per kernel
(`mvt_kernel1`, `mvt_kernel2`) with that kernel's GPU event time and host
wall-clock time, then a `summary` whose `total_time_ms` is the sum of the two
recorded GPU event times:

```json
{"type":"kernel","name":"mvt_kernel1","iter":0,"time_ms":6.123400,"host_time_ms":6.150000,"params":{"n":4096}}
{"type":"kernel","name":"mvt_kernel2","iter":0,"time_ms":6.234500,"host_time_ms":6.260000,"params":{"n":4096}}
{"type":"summary","total_time_ms":12.357900}
```

Derived metrics (e.g. GB/s) are computed downstream from `time_ms` and params.

**stderr** — device info, per-kernel timing, and a correctness check: the
first 8 elements of `x1` and `x2` are compared against a CPU reference. On
mismatch the check prints `FAIL` and the process exits with a non-zero
status.

## Build

```bash
# Auto-detect platform (/opt/rocm exists → rocm, else → cuda)
make

# Explicit platform
make PLATFORM=cuda
make PLATFORM=rocm
```

## Run

```bash
# Default: N=4096
./polybench_mvt

# Custom size
./polybench_mvt --n 2048
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--n N` | 4096 | Matrix dimension (N×N) |

| Env var | Description | Default |
|---------|-------------|---------|
| `BENCH_PARAM_n` | Matrix dimension (N×N) | 4096 |
| `BENCH_PARAM_vector_add_warmup` | Vector-add warmer (clock boost + LLC eviction) before the timed pass | true |

The pass runs exactly once per execution; repetition is the harness's job
(`gpubench run --reps N`).

## Platform Notes

### CUDA / ROCm (HIP)

- Source: `polybench_mvt.hip`  
- Block size: 256 threads  
- Grid: `ceil(N/256)` for both kernels  
- Requires: `nvcc` (CUDA) or `hipcc` (ROCm)

## Reference

Derived from the [PolyBench/GPU](https://sourceforge.net/projects/polybench/) benchmark suite, MVT kernel.
