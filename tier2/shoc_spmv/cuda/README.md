# shoc_spmv — SHOC CSR Sparse Matrix-Vector Multiply Benchmark (native CUDA)

## Description

This benchmark implements CSR (Compressed Sparse Row) Sparse Matrix-Vector Multiplication (SpMV): **y = A × x**, derived from the SHOC (Scalable HeterOgeneous Computing) benchmark suite.

## Provenance & fidelity

Derived from the SHOC benchmark suite (https://github.com/vetter/shoc) — the `SpMV` workload. This is an independent reimplementation for portable measurement, **not** upstream source, and **not** a faithful reproduction of the upstream kernel. It differs from SHOC's version as follows:

- CSR-scalar only vs SHOC's multiple SpMV variants.

Read this as "a CSR-SpMV benchmark in the spirit of SHOC," not as SHOC's `SpMV`.

A synthetic sparse matrix is generated with exactly `nnz_per_row` non-zeros per row (default 32) and `num_rows` rows (default 16384). The row-per-thread CSR SpMV kernel assigns one GPU thread to each output row.

## Algorithm

**CSR Format:**
- `row_ptr[NUM_ROWS+1]`: row i's non-zeros span `[row_ptr[i], row_ptr[i+1])`
- `col_idx[NNZ]`: column index of each non-zero
- `values[NNZ]`: floating-point value of each non-zero
- `x[NUM_ROWS]`: dense input vector
- `y[NUM_ROWS]`: dense output vector

**Kernel (one thread per row):**
```
row = blockIdx.x * blockDim.x + threadIdx.x
if row < NUM_ROWS:
    sum = 0.0
    for k = row_ptr[row] .. row_ptr[row+1]-1:
        sum += values[k] * x[col_idx[k]]
    y[row] = sum
```

**Matrix Generation:**
- `row_ptr[i] = i * NNZ_PER_ROW` (exactly `nnz_per_row` non-zeros per row)
- `col_idx[k]` = random column index in `[0, NUM_ROWS)`
- `values[k]` = random float in `[0, 1)`
- `x[i]` = random float in `[0, 1)`

## Measurement protocol

The SpMV kernel is launched and timed **exactly once** per execution — there is
no in-program warmup or iteration loop. When `BENCH_PARAM_vector_add_warmup` is
`true` (default), the vector-add warmer runs immediately before the timed
launch: ~100 ms of sustained load ramps the GPU clock while its final pass
evicts the last-level cache, so the kernel is measured cold-cache at a boosted
clock. Repetition is the harness's job: `gpubench run --reps N` re-executes the
binary in fresh processes.

## Files

| File | Description |
|------|-------------|
| `shoc_spmv.cu` | CUDA source |
| `Makefile` | nvcc build (`CUDA_ARCH`, default `native`) |
| `README.md` | This file |

## Build Instructions

```bash
make                     # -arch=native (builds for the GPU in this machine)
make CUDA_ARCH=sm_80     # A100
make CUDA_ARCH=sm_90     # H100

# Clean
make clean
```

## Run Instructions

```bash
./shoc_spmv
```

Parameters (env vars override `--flags`):

| Env var | Default | Description |
|---------|---------|-------------|
| `BENCH_PARAM_num_rows` | 16384 | Matrix rows (and columns) |
| `BENCH_PARAM_nnz_per_row` | 32 | Non-zeros per row |
| `BENCH_PARAM_vector_add_warmup` | true | Clock-boost + LLC-evict warmer before the timed run |

## Output Format

**stderr** — Human-readable progress:
```
Device: <device_name>
Rows: 16384  |  NNZ_per_row: 32  |  Total NNZ: 524288

spmv_csr_kernel: gpu = X.XXX ms, host = X.XXX ms
SpMV verification: PASS (16384 rows, 0 mismatches)

Total GPU time: X.XXX ms
```

**stdout** — JSON lines: one `kernel` record for the single timed launch, then
a `summary` whose `total_time_ms` equals that record's GPU event time:
```
{"type":"kernel","name":"spmv_csr_kernel","iter":0,"time_ms":0.234500,"host_time_ms":0.289000,"params":{"num_rows":16384,"nnz_per_row":32,"verification":"full_cpu_v1"}}
{"type":"summary","total_time_ms":0.234500}
```

## Performance Metrics

Both metrics below are derived downstream from the emitted `time_ms` and
params; the binary does not report them.

**Memory Bandwidth (GB/s):**
```
bytes = NNZ * (sizeof(float) + sizeof(int))    # values + col_idx
      + (NUM_ROWS + 1) * sizeof(int)           # row_ptr
      + NUM_ROWS * sizeof(float)               # x (input vector)
      + NUM_ROWS * sizeof(float)               # y (output vector)
GB/s = bytes / time_s / 1e9
```

With the default parameters:
```
bytes = 524288 * 8 + 16385 * 4 + 16384 * 4 + 16384 * 4
      = 4194304 + 65540 + 65536 + 65536
      ≈ 4.39 MB
```

**Arithmetic Throughput (GFLOPS):**
```
GFLOPS = 2 * NNZ / time_s / 1e9
```
(One multiply and one add per non-zero element.)

## Correctness Verification

After the timed run (measurements already emitted), **every row** of the actual
GPU output `y` is compared against the complete CPU reference SpMV. Verification
does not run another GPU SpMV. The existing relative error threshold is retained:
```
rel_err = |gpu - cpu| / (|cpu| + 1e-6) <= 1e-4
```

The benchmark exits with code `0` on PASS and non-zero on FAIL, so the harness
marks the run as failed without losing the measurements. There is no
verification on/off parameter.
Nonfinite outputs, references or relative errors fail. The diagnostic reports
the total rows and mismatches; at most ten individual errors are printed, while
all rows are checked. The kernel record identifies this verification as
`full_cpu_v1`. CPU reference computation, comparison and diagnostics are outside
GPU timing; the GPU algorithm, input generation and launch are unchanged.

Regression commands (use new output directories):

```bash
# [TODO for cuda] these tests exercise ../hip/; not yet ported
python3 tests/shoc_spmv_verification.py --host-only --work-dir /tmp/spmv-host
python3 tests/shoc_spmv_verification.py --platform cuda --work-dir /tmp/spmv-check --sweep
```

The host check exercises the production CPU routine against exact CSR examples.
The GPU check builds an unmodified production executable and a separate test
copy that can corrupt actual device results after timing. It checks rows around
the old 32-row cutoff, interior and final rows, finite/NaN/infinity errors,
skipped GPU work, bounded diagnostics, timing records, and both warmer settings.
`--sweep` also checks all 81 declared workload combinations with the production
binary and the warmer enabled, requiring each process to finish within 60
seconds. These tests establish correctness and process-budget evidence on the
tested device; they do not qualify performance reproducibility.

**[TODO for cuda]** The evidence in this paragraph was collected for [`../hip/`](../hip/); it has not yet been rerun for this CUDA build. H100 PCIe / CUDA 12.9 validation on 2026-09-28 passed all 81 configurations and
123 production/fault cases. The largest case checked all 8,388,608 rows with 64
entries per row; the slowest complete process took 11.377 seconds. Four GPU
sanitizers, eight existing output/reference finite-value cases and six native
CLI/SQLite outcomes also passed. The GPU was shared with other jobs, so these
are correctness and observed process-budget results, not publication timings.
