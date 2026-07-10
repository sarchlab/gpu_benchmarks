# tier2/parboil_spmv — Sparse Matrix-Vector Multiplication (Parboil SpMV, JDS)

Faithful port of the Parboil **SpMV** benchmark: the device kernel `spmv_jds`
is the original (`parboil/benchmarks/spmv/src/cuda/jds_kernels.cu`), and the host
driver builds the original **JDS (Jagged Diagonal Storage)** layout with a port
of Parboil's `coo_to_jds` converter (`common_src/convert-dataset/convert_dataset.c`).
Only the host driver is restructured for the per-iteration timing harness.

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

## Parameters

| Env var / flag | Description | Default |
|----------------|-------------|---------|
| `BENCH_PARAM_num_rows` / `--num_rows` | Matrix rows (≤ ~160000) | 50000 |
| `BENCH_PARAM_avg_nnz_per_row` / `--avg_nnz_per_row` | Avg non-zeros per row (depth < 5000) | 80 |
| `BENCH_PARAM_block_size` / `--block_size` | Override block size (0 = original `compute_active_thread`) | 0 |
| `--seed` | RNG seed for matrix generation | 42 |
| `BENCH_PARAM_warmup` | Warmup launches (untimed) | 3 |
| `BENCH_PARAM_iteration` | Timed launches, each recorded | 7 |

Matrix and vector are generated synthetically (no `.mtx` file); sized to a
Parboil-scale matrix (Dubcova3/mixtank are tens of thousands of rows).

## Build / Run

```bash
make               # auto-detect platform (cuda / rocm / metal)
./parboil_spmv
./parboil_spmv --num_rows 100000 --avg_nnz_per_row 40
```

## Output

**stdout** — one `kernel` record per timed iteration for `spmv_jds` (one launch
per pass), then a `summary` whose `total_time_ms` is the sum of all iterations'
GPU event times. **stderr** — device info, per-iteration timing, PASS/FAIL (all
rows vs. a CPU reference built from the generated COO). GFLOPS (`2 × nnz / t`) is
derived downstream from `time_ms` and params.

> **Metal note:** `parboil_spmv.metal` / `parboil_spmv_metal.mm` still contain the
> earlier CSR implementation and were not updated by this faithful port
> (HIP/CUDA only).
