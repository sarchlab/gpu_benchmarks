# tier2/parboil_spmv/hip — Sparse Matrix-Vector Multiplication (Parboil SpMV, JDS)

Faithful port of the Parboil **SpMV** benchmark: the device kernel `spmv_jds`
is the original (`parboil/benchmarks/spmv/src/cuda/jds_kernels.cu`), and the host
driver builds the original **JDS (Jagged Diagonal Storage)** layout with a port
of Parboil's `coo_to_jds` converter (`common_src/convert-dataset/convert_dataset.c`).
Only the host driver is restructured for the single-run timing harness.

## Provenance & fidelity

Derived from the [Parboil benchmark suite](http://impact.crhc.illinois.edu/parboil/) — the `spmv` workload. Independent reimplementation, spot-verified to reproduce Parboil's algorithm faithfully (JDS format), with one caveat:

- One portability substitution — the dense-vector texture fetch → `__ldg` — documented below; the JDS data layout is preserved verbatim.

The `coo_to_jds` converter previously corrupted the matrix when the final row
held exactly one nonzero (the row-transition and last-element cases were folded
into one branch, over-counting the previous row and dropping the final row).
That is fixed (issue #63, C-01): row-transition accounting and final-row
flushing are now separate. A host-only regression, `./parboil_spmv --selftest`
(no GPU required), reconstructs `y = A·x` from the JDS arrays exactly as
`spmv_jds` indexes them and checks it against a direct COO reference on the
pathological inputs.

## Algorithm (JDS — the faithful part)

- Rows are sorted by **descending non-zero count** (jagged / Fubini transpose)
  and stored **column-major** so thread `ix` walks its row's columns at stride-1
  offsets `jds_ptr[k] + ix`.
- Per-32-row-warp-group column counts (`sh_zcnt_int`) and per-column base offsets
  (`jds_ptr_int`) live in **constant memory** (each capped at 5000 entries →
  `num_rows ≤ 160000`, JDS column depth `< 5000`).
- One thread computes one row over the JDS permutation `d_perm`; the result is
  scattered back to the real row. Launch geometry is the original
  `compute_active_thread` (`gpu_info.cc`).

## Texture → `__ldg` substitution (spec rule 2)

The canonical Parboil SpMV fetches the dense vector `x` through a 1-D texture
(`tex1Dfetch`). The compat layer does not map CUDA texture references, so the
three `x_vec` reads use `__ldg(&x_vec[i])` (read-only data cache) — the closest
portable equivalent. The **JDS data layout is preserved verbatim.** Jagged
padding entries carry column index `0` (instead of the converter's `-1`) with
value `0`, so the read-only gather never dereferences a negative index; the
value is `0` so the product is unchanged.

## Measurement

The `spmv_jds` kernel launches **exactly once** per process execution (see the
root README "Measurement Methodology"). With `BENCH_PARAM_vector_add_warmup`
(default `true`), a vector-add warmer boosts the GPU clock and evicts the
last-level cache immediately before the timed launch. Repetition =
`gpubench run --reps N`.

## Parameters

| Env var / flag | Description | Default |
|----------------|-------------|---------|
| `BENCH_PARAM_num_rows` / `--num_rows` | Matrix rows (≤ ~160000) | 131072 |
| `BENCH_PARAM_avg_nnz_per_row` / `--avg_nnz_per_row` | Avg non-zeros per row (depth < 5000) | 80 |
| `BENCH_PARAM_vector_add_warmup` | Cold-cache clock warmer on/off | true |

Direct-binary controls (not swept by `params.json`):
`BENCH_PARAM_block_size` / `--block_size` overrides the launch block size
(default 0 = the original `compute_active_thread` geometry); `--seed` sets the
matrix RNG seed (default 42). These preserve the existing parsing rules:
positive CLI values are accepted; the environment block-size override takes
precedence, and its zero or negative values select automatic geometry.

Every kernel record includes `seed`, the resolved `requested_block_size`,
`launch_policy` (`auto_v1` or `block_override_v1`), and the effective `block_size`
and `grid_size`. The requested value is recorded after CLI/default/environment
resolution. The regular manifest sweep uses seed 42 and automatic geometry
unless `BENCH_PARAM_block_size` is inherited from the harness environment.
That override is visible in the output even though it is not a sweep axis. A seed alone does not ensure identical matrices across C-library RNG
implementations; portable input generation remains deferred.

Matrix and vector are generated synthetically (no `.mtx` file); sized to a
Parboil-scale matrix (Dubcova3/mixtank are tens of thousands of rows).

## Build / Run

```bash
make               # auto-detect platform (cuda / rocm)
./parboil_spmv
./parboil_spmv --num_rows 100000 --avg_nnz_per_row 40
```

## Output

**stdout** — one `kernel` record (`iter:0`) for the single `spmv_jds` launch,
then a `summary` whose `total_time_ms` is the sum of all recorded GPU event
times. **stderr** — device info, timing, PASS/FAIL (all rows vs. a CPU
reference built from the generated COO); on FAIL the process exits non-zero.
GFLOPS (`2 × nnz / t`) is derived downstream from `time_ms` and params.

`python3 tests/parboil_spmv_controls.py --compiler clang++` checks the actual
parameter resolution, geometry calculation and JSON formatting without a GPU.
Its independent expectations cover CLI/environment precedence, automatic
geometry branches and direct overrides; it does not qualify hardware support
for arbitrary block sizes.
