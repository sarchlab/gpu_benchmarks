# tier2/rodinia_lud — Blocked LU Decomposition (Rodinia)

Implements the Rodinia LUD benchmark: blocked LU decomposition (no pivoting)
of a dense NxN single-precision matrix.

## Provenance & fidelity

Derived from the [Rodinia benchmark suite](http://lava.cs.virginia.edu/Rodinia/) — the `lud` workload. Independent reimplementation, spot-verified to reproduce Rodinia's algorithm faithfully.

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

After the timed factorization the benchmark reconstructs **A = L·U** on the
CPU and checks: `||A - LU||_F / ||A||_F < 1e-4`. On FAIL the benchmark exits
with a non-zero status. This checks the actual measured device matrix; there
is no preliminary verification factorization. Non-finite residuals fail the
existing comparison. Every kernel record identifies this policy as
`verification: "full_lu_residual_v1"`.

The full residual remains O(N³) CPU work. Its CPU reconstruction now processes
four rows and 64 columns at a time to reuse matrix reads. Every output keeps
its original increasing-k accumulation order, and both norm sums retain their
original row-major order. Verification coverage, tolerance and the recorded
`full_lu_residual_v1` policy are preserved.

There are three N² host float buffers and one N² device float buffer, plus the
optional 768 MiB warmer. The CPU checker adds 4N doubles of scratch and a
4×64-double tile: 256 KiB plus 2 KiB at N=8192. Measure complete process wall
time on each target before admitting a configuration; the cubic check still
depends on the host CPU and its memory system. GPU event times exclude the
result download and CPU check. AMD execution qualification remains R-19.

The factorization runs **exactly once** (repetition is the harness's job,
`gpubench run --reps N`); the pristine matrix upload happens before the timed
brackets. Recollect results from drivers that ran the old preliminary
factorization, since that extra execution changed the process history.

## Performance Metric

Derived downstream from the emitted per-kernel times:

```
GFLOPS = (2/3 · N³) / time_s / 1e9   (derived downstream)
```

## Build

```bash
# Auto-detect platform (/opt/rocm present → rocm, else → cuda)
make

# Explicit platform
make PLATFORM=cuda
make PLATFORM=rocm
```

Requirements:
- **CUDA**: `nvcc` (CUDA toolkit)
- **ROCm**: `hipcc` (ROCm ≥ 5.0)

## Run

```bash
./rodinia_lud                        # default: N=512
./rodinia_lud --size 1024            # larger matrix
```

`N` must be divisible by the block size (16).

## Output

Progress and verification are printed to **stderr**. **stdout** carries JSON
lines: one `kernel` record per kernel (`lud_diagonal`,
`lud_perimeter`, `lud_internal`) with that kernel's accumulated GPU event time
across all block steps, then a `summary` whose `total_time_ms` is
the sum of all recorded GPU event times:

```json
{"type":"kernel","name":"lud_diagonal","iter":0,"time_ms":0.512300,"host_time_ms":0.850000,"params":{"size":512,"verification":"full_lu_residual_v1"}}
{"type":"kernel","name":"lud_perimeter","iter":0,"time_ms":3.123400,"host_time_ms":3.500000,"params":{"size":512,"verification":"full_lu_residual_v1"}}
{"type":"kernel","name":"lud_internal","iter":0,"time_ms":8.710000,"host_time_ms":9.100000,"params":{"size":512,"verification":"full_lu_residual_v1"}}
{"type":"summary","total_time_ms":12.345700}
```

Derived metrics (e.g. GFLOPS) are computed downstream from `time_ms` and params.

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--size N` | 512 | Matrix dimension (must be divisible by 16) |

| Env var | Description | Default |
|---------|-------------|---------|
| `BENCH_PARAM_size` | Matrix dimension N | 512 |
| `BENCH_PARAM_vector_add_warmup` | Cold-cache calibration before the timed pass | `true` |

`BENCH_PARAM_verify=false` (also `0` or `skip`) preserves the existing manual
debug bypass. It is not a declared sweep parameter, and all emitted kernel
records then say `verification: "disabled"`. Such a run is not a verified
measurement. Other values retain the existing enabled behavior.

`python3 tests/lud_measured_result.py --platform cuda --work-dir /tmp/lud-check`
from the repository root checks the production driver in isolated copies.
Choose a new, nonexistent `--work-dir` for each invocation; previous results
are retained for inspection.
It counts factorization passes and corrupts actual device output after the
recorded factorization, covering diagonal, lower/upper and tail entries with
NaN, infinities and large finite errors. It also checks valid results and
manual bypass reporting. Use `--platform rocm` for AMD; GPU execution is
required, while `--compile-only` only builds the instrumented driver.

`tests/lud_residual_test.cpp` independently compares every reconstructed value
and the final residual with the original unblocked formula, including partial
row/column tiles, signed and extreme inputs, non-finite values and near-threshold
acceptance. It also checks a known factorization and protects output boundaries.
Run `c++ -O3 -std=c++14 -fsanitize=address,undefined tests/lud_residual_test.cpp
-o /tmp/lud_residual_test && /tmp/lud_residual_test` from the repository root.
This host test needs no GPU; CI runs it alongside compilation of the actual
measured-result GPU regression above.

## Reference

Rodinia Benchmark Suite — [LUD](https://github.com/yuhc/gpu-rodinia)
S. Che et al., "Rodinia: A benchmark suite for heterogeneous computing,"
IISWC 2009.
