# tier2/rodinia_lud — Blocked LU Decomposition (Rodinia)

Implements the Rodinia LUD benchmark: blocked LU decomposition (no pivoting)
of a dense NxN single-precision matrix.

## Algorithm

Three GPU kernels are launched in sequence for each diagonal block step `k`:

| Kernel | Description |
|--------|-------------|
| `lud_diagonal` | In-place LU factor of the 16×16 diagonal block at `(k,k)` |
| `lud_perimeter` | Forward/back-solve for row-panel (right) and column-panel (below) blocks |
| `lud_internal` | Schur-complement update for all remaining interior blocks |

The matrix is stored in-place:
- Lower-triangular part (i>j) → **L** (unit diagonal implicit)
- Upper-triangular part (i≤j) → **U**

## Verification

After the warmup passes the benchmark reconstructs **A = L·U** on the CPU and
checks: `||A - LU||_F / ||A||_F < 1e-4`.

The factorization is in place, so every pass (warmup and timed) re-uploads the
pristine matrix before factoring; the re-upload is outside the timed brackets.

## Performance Metric

Derived downstream from the emitted per-iteration times:

```
GFLOPS = (2/3 · N³) / time_s / 1e9   (derived downstream)
```

## Build

```bash
# Auto-detect platform (Darwin → metal, /opt/rocm present → rocm, else → cuda)
make

# Explicit platform
make PLATFORM=cuda
make PLATFORM=rocm
make PLATFORM=metal
```

Requirements:
- **CUDA**: `nvcc` (CUDA toolkit)
- **ROCm**: `hipcc` (ROCm ≥ 5.0)
- **Metal**: macOS 12+, Xcode Command Line Tools; `rodinia_lud.metal` must be
  in the same directory as the binary at runtime.

## Run

```bash
./rodinia_lud                        # default: N=512
./rodinia_lud --size 1024            # larger matrix
```

`N` must be divisible by the block size (16).

## Output

Progress and verification are printed to **stderr**. **stdout** carries JSON
lines: per timed pass, one `kernel` record per kernel (`lud_diagonal`,
`lud_perimeter`, `lud_internal`) with that kernel's accumulated GPU event time
across all block steps in the pass, then a `summary` whose `total_time_ms` is
the sum of all recorded GPU event times:

```json
{"type":"kernel","name":"lud_diagonal","iter":0,"time_ms":0.512300,"host_time_ms":0.850000,"params":{"size":512,"block_size":256,"verify":"true"}}
{"type":"kernel","name":"lud_perimeter","iter":0,"time_ms":3.123400,"host_time_ms":3.500000,"params":{"size":512,"block_size":256,"verify":"true"}}
{"type":"kernel","name":"lud_internal","iter":0,"time_ms":8.710000,"host_time_ms":9.100000,"params":{"size":512,"block_size":256,"verify":"true"}}
...
{"type":"summary","total_time_ms":86.421000}
```

Derived metrics (e.g. GFLOPS) are computed downstream from `time_ms` and params.

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--size N` | 512 | Matrix dimension (must be divisible by 16) |

| Env var | Description | Default |
|---------|-------------|---------|
| `BENCH_PARAM_size` | Matrix dimension N | 512 |
| `BENCH_PARAM_block_size` | Threads-per-block label echoed into records | 256 |
| `BENCH_PARAM_verify` | Verification mode (true, false, skip) | true |
| `BENCH_PARAM_warmup` | Warmup passes (untimed) | 3 |
| `BENCH_PARAM_iteration` | Timed passes, each recorded | 7 |

## Reference

Rodinia Benchmark Suite — [LUD](https://github.com/yuhc/gpu-rodinia)
S. Che et al., "Rodinia: A benchmark suite for heterogeneous computing,"
IISWC 2009.
