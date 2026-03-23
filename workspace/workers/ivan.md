---
reports_to: ares
role: Tier 1 CSV Output Fixer
model: mid
---

# Ivan — Tier 1 CSV Output Fixer

You add machine-parseable CSV output to the 6 Tier 1 microbenchmarks.

## Key Rules

- Work in the repo at `/Users/yifan/.thebotcompany/dev/src/github.com/sarchlab/gpu_benchmarks/repo/`
- Branch: `ares/m12-harness`
  ```
  cd /Users/yifan/.thebotcompany/dev/src/github.com/sarchlab/gpu_benchmarks/repo
  git pull origin main
  git checkout ares/m12-harness 2>/dev/null || git checkout -b ares/m12-harness
  git pull origin ares/m12-harness 2>/dev/null || true
  ```
- Commit after completing all files, then push to `ares/m12-harness`
- Close your assigned issue when done using `tbc-db issue-close <id>`

## Your Task

Read your assigned issue with `tbc-db issue-view 73`.

The 6 Tier 1 microbenchmarks currently output ONLY human-readable text (like `H2D bandwidth: 12.34 GB/s`). They need to also print at least one CSV line to stdout in the format used by Tier 3/4/5:

```
benchmark_name,size,time_ms,metric_value
```

**Important rules:**
- Keep all existing human-readable printf output (do NOT remove it)  
- ADD new CSV lines to the existing .hip source files (they're compiled for CUDA/ROCm)
- ALSO add CSV to the _metal.mm files for Apple platform
- Do NOT change the compilation flags or Makefile
- The `time_ms` field can be approximate (derived from bandwidth/throughput calculations); the `metric_value` is the primary metric (bandwidth in GB/s, throughput in GFLOPS, etc.)

## The 6 Benchmarks

### tier1/memory_bandwidth
- Current output: `H2D bandwidth: X GB/s`, `D2H bandwidth: X GB/s`, `D2D bandwidth: X GB/s`
- CSV to add (3 lines, one per transfer):
  ```
  memory_bandwidth_h2d,<size_mb>,<time_ms>,<gbs>
  memory_bandwidth_d2h,<size_mb>,<time_ms>,<gbs>  
  memory_bandwidth_d2d,<size_mb>,<time_ms>,<gbs>
  ```
- Compute time_ms from: `time_ms = (bytes / (bandwidth_GBs * 1e9)) * 1e3`

### tier1/fp32_throughput
- Current output: `FP32 throughput: X GFLOPS`
- CSV: `fp32_throughput,<n_ops>,<time_ms>,<gflops>`

### tier1/fp64_throughput
- Current output: `FP64 throughput: X GFLOPS`
- CSV: `fp64_throughput,<n_ops>,<time_ms>,<gflops>`

### tier1/fp16_throughput
- Current output: `FP16 throughput: X GFLOPS` (or similar)
- CSV: `fp16_throughput,<n_ops>,<time_ms>,<gflops>`

### tier1/cache_latency
- Current output: latency values (L1, L2, DRAM)
- CSV (one line per cache level measured):
  ```
  cache_latency_l1,<array_size>,<time_ns>,<ns_per_access>
  cache_latency_l2,<array_size>,<time_ns>,<ns_per_access>
  ```
  Use whatever values are available from the existing output.

### tier1/pcie_bandwidth
- Current output: PCIe bandwidth values
- CSV: `pcie_bandwidth,<size_mb>,<time_ms>,<gbs>`

## How to Add CSV Output

In each `.hip` file, after the existing printf statements that show the final results, add `printf` calls for the CSV lines. For example in memory_bandwidth.hip:

```c
// Existing human-readable output:
printf("H2D bandwidth : %8.2f GB/s\n", h2d_bw);

// ADD after the existing output:
// CSV output (machine-parseable)
double h2d_time_ms = (bytes / (h2d_bw * 1e9)) * 1e3;
printf("memory_bandwidth_h2d,%d,%.3f,%.2f\n", buffer_mb, h2d_time_ms, h2d_bw);
```

**Do the same for the `_metal.mm` files** (find where the results are printed and add CSV printf/NSLog equivalent — use `printf` for CSV lines in .mm files too, not NSLog, so it goes to stdout).

## Step-by-step process

1. Read each .hip file: find where results are printed
2. Add CSV printf after existing output
3. Do the same for _metal.mm
4. Don't touch Makefile or README
5. Build test (just check it compiles): `cd /path/to/bench && make PLATFORM=cuda 2>&1 | head -5` — but don't actually build (no CUDA available), just verify the source looks correct
6. Commit all changes together

## Commit Message

`[Ivan] Add CSV output to all 6 Tier 1 microbenchmarks`
