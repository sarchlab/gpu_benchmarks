# tier3/cusparse_spmv — CSR Sparse Matrix-Vector Multiply (Vendor Library)

## Description

Benchmarks **Sparse Matrix-Vector Multiplication (SpMV)** in CSR (Compressed Sparse Row) format using vendor GPU sparse-linear-algebra libraries:

| Platform | Library       | Notes                                      |
|----------|---------------|--------------------------------------------|
| NVIDIA   | cuSPARSE      | `cusparseSpMV` — modern generic API (CUDA 11+) |
| AMD      | rocSPARSE     | `rocsparse_scsrmv` — CSR SpMV              |

## Algorithm

Computes **y = A × x** where:

- **A** — synthetic CSR sparse matrix: `NUM_ROWS × NUM_COLS`, exactly `NNZ_PER_ROW` non-zeros per row
- **x** — dense input vector of length `NUM_COLS`
- **y** — dense output vector of length `NUM_ROWS`

CSR representation:
- `row_ptr[NUM_ROWS+1]` — row offsets
- `col_idx[NNZ]` — column indices (random, uniform in `[0, NUM_COLS)`)
- `values[NNZ]` — floating-point values (random in `[0, 1)`)

## Parameters

| Parameter   | Value   |
|-------------|---------|
| NUM_ROWS    | 16384   |
| NUM_COLS    | 16384   |
| NNZ_PER_ROW | 32      |
| NNZ         | 524288  |
| Warmup      | 5 iters |
| Timed       | 5 iters |

## Files

| File                      | Description                                    |
|---------------------------|------------------------------------------------|
| `cusparse_spmv.hip`       | CUDA (cuSPARSE) and ROCm (rocSPARSE) source    |
| `Makefile`                | Auto-detects platform; builds the binary       |
| `README.md`               | This file                                      |

## Building

The `Makefile` auto-detects the platform:
- **Linux with `/opt/rocm`** → ROCm/rocSPARSE
- **Otherwise** → CUDA/cuSPARSE

```bash
make                  # auto-detect
make PLATFORM=cuda    # NVIDIA CUDA
make PLATFORM=rocm    # AMD ROCm
make clean
```

### CUDA manual build
```bash
nvcc -O3 -std=c++14 -x cu -DUSE_CUSPARSE cusparse_spmv.hip -lcusparse -o cusparse_spmv
```

### ROCm manual build
```bash
hipcc -O3 -std=c++14 -DUSE_ROCSPARSE cusparse_spmv.hip -lrocsparse -o cusparse_spmv
```

## Running

```bash
./cusparse_spmv
```

## Output Format

Progress and diagnostics on **stderr**:
```
Platform: cuSPARSE
Device: NVIDIA A100-SXM4-80GB (id=0)
Rows: 16384  |  NNZ_per_row: 32  |  Total NNZ: 524288
Warmup: 5  |  Iterations: 5

  iter 0: 0.0423 ms
  iter 1: 0.0418 ms
  ...
Correctness check (first 32 elements): PASS

GB/s: 148.63  GFLOPS: 24.77  (avg 0.0422 ms, ...)
```

CSV result on **stdout**:
```
cusparse_spmv,16384,32,<time_ms>,<GBs>,<GFLOPS>
```

## Performance Metrics

| Metric | Formula |
|--------|---------|
| **GB/s** | `NNZ × 12.0 / time_s / 1e9` |
| **GFLOPS** | `2.0 × NNZ / time_s / 1e9` |

The GB/s formula counts 12 bytes per non-zero: 4 bytes for `values[k]` (float), 4 bytes for `col_idx[k]` (int), and ~4 bytes amortized for `x[col_idx[k]]` (float read). The GFLOPS formula counts 2 FLOPs per non-zero (one multiply + one add).

## Correctness Verification

After timing, the GPU result is compared against a CPU SpMV reference on the first 32 elements. A relative tolerance of `1e-4` is used. The program exits with code 0 on pass, 1 on failure.

## Notes

- **cuSPARSE API version**: This benchmark uses the modern generic SpMV API (`cusparseSpMV` / `cusparseCreateCsr`) introduced in CUDA 11. It is **not** compatible with CUDA 10 or earlier.
- **rocSPARSE API**: Uses `rocsparse_scsrmv` (the CSR-format SpMV routine). Available in all ROCm releases.
