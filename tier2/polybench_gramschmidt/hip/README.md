# polybench_gramschmidt

PolyBench Gram-Schmidt orthogonalization benchmark.

## Description

Computes the QR factorization of a dense M×N matrix A using the Gram-Schmidt
orthogonalization process:

```
A = Q * R
```

where Q is M×N with orthonormal columns and R is an N×N upper-triangular matrix.
This benchmark is derived from the [PolyBench/GPU](https://web.cse.ohio-state.edu/~pouchet.2/software/polybench/) suite.

## Algorithm

For each column k = 0 .. N−1:

1. **norm step**: compute `nrm = ||A[:,k]||₂`; set `R[k,k] = nrm`
2. **normalize step**: `Q[:,k] = A[:,k] / nrm`
3. **projection step**: for each column j = k+1 .. N−1:
   - `R[k,j] = Q[:,k]ᵀ · A[:,j]`
   - `A[:,j] -= R[k,j] * Q[:,k]`

After all N columns, Q has orthonormal columns: `Qᵀ Q ≈ Iₙ`.

### GPU Kernel Structure

| Kernel | Threads | Work |
|--------|---------|------|
| `gram_norm_zero` | 1 thread | Reset the norm accumulator to zero |
| `gram_norm` (HIP) | M threads, atomicAdd | Accumulate `||A[:,k]||²` |
| `gram_norm_finish` (HIP) | 1 thread | `R[k,k] = sqrt(accum)` |
| `gram_normalize` | M threads | `Q[:,k] = A[:,k] / nrm` |
| `gram_project` | N−k−1 threads | Update columns j > k |

## Files

| File | Description |
|------|-------------|
| `polybench_gramschmidt.hip` | HIP/CUDA implementation (NVIDIA + AMD) |
| `Makefile` | Multi-platform build |
| `README.md` | This file |

## Build

Platform is auto-detected; override with `PLATFORM=cuda|rocm`.

```bash
# Auto-detect
make

# NVIDIA CUDA
make PLATFORM=cuda

# AMD ROCm/HIP
make PLATFORM=rocm

# Clean
make clean
```

### Requirements

| Platform | Toolchain |
|----------|-----------|
| CUDA | NVIDIA CUDA toolkit (`nvcc`) |
| ROCm | AMD ROCm (`hipcc`) |

## Run

```bash
./polybench_gramschmidt                    # default: M=512, N=64
./polybench_gramschmidt --m 1024 --n 64   # larger matrix
```

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `BENCH_PARAM_m` | 512 | Number of rows (overrides `--m`) |
| `BENCH_PARAM_n` | 64 | Number of columns (overrides `--n`) |
| `BENCH_PARAM_vector_add_warmup` | true | Vector-add warmer (clock boost + LLC eviction) before the factorization |

The run executes once per process: it performs a single full QR
factorization and emits one record per kernel accumulating all k-columns.
Repetition across runs is the harness's job (`gpubench run --reps N`).

Resolved M and N must be positive, and both `M*N` and `N*N` must fit a signed
32-bit integer for the existing matrix/index expressions. Invalid dimensions
fail before allocation or device access, without timing records. Launch-grid
rounding uses widened arithmetic. Positive command-line fallback and
environment-variable precedence are unchanged; this does not introduce a new
strict string parser.

These bounds establish arithmetic safety only. They do not guarantee available
memory, full column rank or a bounded process time for arbitrary overrides.
Full Q checking costs O(M N²). H100 budget evidence covers the declared N=64
sizes; accepting an override at the input guard does not qualify it for
collection. Complete QR verification remains separate below.

## Output Format

**stderr** — device info, per-kernel timing, orthogonality check result,
summary

**stdout** — JSON lines: one kernel record per kernel
(`gram_norm_zero`, `gram_norm`, `gram_norm_finish`, `gram_normalize`, `gram_project`) carrying
that kernel's GPU event time (`time_ms`) and host wall-clock time
(`host_time_ms`) accumulated over all k-columns of the factorization,
then a summary whose `total_time_ms` is the sum of all recorded GPU event
times. The matrix A is restored (and Q/R zeroed) before the factorization,
outside the timed brackets and before the optional entry warmer. The scalar
norm reset is required at every column and is a separately timed GPU kernel.
For N positive, there are N reset launches and `5*N - 1` total timed launches;
`gram_project` has an empty accumulator when N=1. No warmer runs between stages.
Prior measurements omitted reset time and must be recollected:


```
{"type":"kernel","name":"gram_norm","iter":0,"time_ms":12.345600,"host_time_ms":13.010000,"params":{"m":512,"n":64,"q_orthogonality":"all_columns_v1"}}
...
{"type":"summary","total_time_ms":45.622600}
```

## Performance Metric (derived downstream)

```
GB/s = (3 × M × N × sizeof(float)) / (time_s × 10⁹)
```

The factor of 3 accounts for the dominant memory traffic in the projection
step: one read of `Q[:,k]`, one read of `A[:,j]`, and one write to `A[:,j]`
per column j > k, summed across all N column iterations. This metric is
computed downstream from the emitted `time_ms` and params.

## Orthogonality Verification

After the timed run, the benchmark downloads the actual measured Q and computes
every entry of `Qᵀ Q` on the CPU, using all M rows and all N columns. The original
FP32 row-order accumulation and absolute tolerance are preserved:

- **Diagonal** elements: `|dot(Q[:,i], Q[:,i]) − 1| ≤ 0.01`
- **Off-diagonal** elements: `|dot(Q[:,i], Q[:,j])| ≤ 0.01`

Nonfinite dot products fail. Every failing column pair is counted, with at most
ten detailed diagnostics. A `PASS` / `FAIL` result is printed to stderr; on
`FAIL` the process exits nonzero. There is no sampled column prefix or
verification-only GPU rerun. The descriptor `q_orthogonality: "all_columns_v1"`
identifies this property check in each kernel record. Download and CPU work are
outside GPU timing but count toward the runner's whole-process budget.

Orthogonality alone does not verify the QR factorization: an unrelated
orthonormal Q would pass, and R and the final A workspace are not checked here.
Complete measured-state verification remains a separate R-13 repair in
[issue #71](https://github.com/sarchlab/gpu_benchmarks/issues/71). This coverage
change alone does not establish overall performance-collection readiness.

From the repository root, use new work directories:

```bash
python3 tests/gramschmidt_q_verification.py --host-only --compiler clang++ --work-dir /tmp/gram-q-host
python3 tests/gramschmidt_q_verification.py --platform cuda --sweep --work-dir /tmp/gram-q-gpu
```

Host fixtures exercise the production checker with independently constructed
orthonormal matrices, every-element corruptions, duplicate columns, nonfinite
values and immutable inputs/guards under fatal sanitizers. GPU regressions
corrupt the measured Q after timing, including the formerly unchecked columns,
and reject skipped normalization and unit-norm duplicate columns. The sweep
covers all 20 declared matrix sizes with both warmer modes, each within a
60-second process limit. Raw logs, hashes and outcomes remain in the work
directory. CI runs host fixtures and compiles GPU variants; execution requires
hardware.

## Reset qualification

`make test PLATFORM=cuda` (or `rocm`) checks the production reset kernel with
NaN, infinity, negative zero and finite initial values, requiring exact positive
zero and untouched guards. CI compiles the test; running it requires a GPU.
The shared [reset ordering test](../../../tests/algorithmic_zero_order.py) uses
actual NVIDIA Nsight timelines to verify reset order, every event bracket,
absence of inter-stage runtime memsets/copies, and record/summary agreement.
See [test instructions](../../../tests/README.md). Full measured QR reconstruction
and A workspace verification remain R-13; AMD runtime qualification remains R-19.
