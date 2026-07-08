# tier2/parboil_spmv — Sparse Matrix-Vector Multiplication (Parboil SpMV)

CSR (Compressed Sparse Row) Sparse Matrix-Vector Multiplication: **y = A × x**

Each GPU thread computes one output row, performing a dot product of the corresponding sparse row of A with the dense input vector x. Emits per-iteration kernel times for irregular memory-access patterns common in scientific computing; GFLOPS is derived downstream.

Derived from the [Parboil benchmark suite](http://impact.crhc.illinois.edu/parboil.aspx) SpMV workload.

## Algorithm

Matrix A is stored in CSR format:
- `row_ptr[i]` and `row_ptr[i+1]` give the range of non-zeros for row `i`
- `col_idx[j]` is the column of the j-th non-zero element
- `values[j]` is the value of the j-th non-zero element

**Kernel** (`spmv_csr`): For each row, iterate over its non-zeros and accumulate:

```
y[row] = sum(values[j] * x[col_idx[j]])  for j in [row_ptr[row], row_ptr[row+1])
```

The matrix is generated randomly (no file I/O required):
- `num_rows` rows, each with a random number of non-zeros uniformly distributed around `avg_nnz_per_row`
- Column indices are spread across `[0, num_rows)` and sorted per row

## Files

| File | Description |
|------|-------------|
| `parboil_spmv.hip`       | HIP source — SpMV kernel + self-contained host code |
| `parboil_spmv.metal`     | Metal compute shader — `spmv_csr_kernel` |
| `parboil_spmv_metal.mm`  | ObjC++ Metal host code |
| `Makefile`               | Auto-detecting build (Darwin→metal, ROCm→rocm, else→cuda) |

## Build

```bash
make               # auto-detect platform
make PLATFORM=cuda
make PLATFORM=rocm
make PLATFORM=metal
```

## Run

```bash
./parboil_spmv                                          # defaults: 65536 rows, 10 nnz/row
./parboil_spmv --num_rows 131072
./parboil_spmv --num_rows 65536 --avg_nnz_per_row 20
```

## Output

**stdout** — JSON lines, one `kernel` record per timed iteration carrying the
GPU event time (`time_ms`) and host wall-clock time (`host_time_ms`), then a
`summary` whose `total_time_ms` is the sum of all iterations' GPU event times:
```
{"type":"kernel","name":"spmv_csr","iter":0,"time_ms":0.823400,"host_time_ms":0.851000,"params":{"num_rows":65536,"avg_nnz_per_row":10,"block_size":256}}
...
{"type":"summary","total_time_ms":5.763800}
```

**stderr** — human-readable results:
```
Device: Apple M4 Pro
Rows: 65536  |  NNZ: 655360  |  Block: 256

spmv_csr iter 0: gpu = 0.823 ms, host = 0.851 ms
...
Total GPU time: 5.764 ms over 7 iterations
PASS
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--num_rows N` | 65536 | Number of matrix rows (= number of columns) |
| `--avg_nnz_per_row K` | 10 | Average non-zeros per row |
| `--block_size B` | 256 | GPU thread-block size (CUDA/ROCm only) |
| `--seed S` | 42 | RNG seed for matrix generation |

Environment variables `BENCH_PARAM_num_rows`, `BENCH_PARAM_avg_nnz_per_row`,
and `BENCH_PARAM_block_size` override the flags. In addition:

| Env var | Default | Description |
|---------|---------|-------------|
| `BENCH_PARAM_num_warmup` | 3 | Warmup launches (untimed) |
| `BENCH_PARAM_num_iters` | 7 | Timed launches, each recorded |

## Performance Metric

```
GFLOPS = 2 × total_nnz / time_seconds / 1e9
```

Factor of 2 accounts for one multiply and one add per non-zero element.
Derived downstream from the emitted `time_ms` and params.

## Platform Notes

- **CUDA/ROCm**: compiled with `nvcc`/`hipcc` from `parboil_spmv.hip`
- **Metal**: compiled with `clang++` from `parboil_spmv_metal.mm`; the Metal shader `parboil_spmv.metal` is loaded at runtime from the same directory as the binary
- Input data is generated synthetically — no external input files required
- Verification checks 100 sampled rows against CPU reference output
