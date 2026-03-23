# Iris Workspace Note

## Last Completed: Issue #146 - params.json for Tier 1 + Tier 3

**Branch:** `iris/params-tier1-tier3`
**Commit:** 27afd14
**Status:** Pushed, ready for PR/review

### What was done:
- Read all 23 .hip source files (12 Tier 1, 11 Tier 3)
- Created params.json for each benchmark with:
  - Scaling parameter: 19-25 values spanning small to large
  - Non-scaling parameters: 2-4 params with 3-6 values each
  - Kernel names extracted from actual `__global__` functions and library calls

### Summary of scaling params chosen:
- **Compute throughput** (fp32/fp64/fp16/int32): `fmas_per_thread` / `mads_per_thread` (4 to 2M)
- **Memory bandwidth/PCIe**: `buffer_size_mb` (1 to 4096 MiB)
- **Cache latency**: `array_bytes` (4KB to 256MB)
- **Tensor core**: `matrix_dim` (256 to 16384)
- **Mixbench**: `buffer_size_mb` (1 to 4096 MiB)
- **Shared mem bw**: `inner_iters` (10 to 100K)
- **Atomics/shuffle**: `ops_per_thread` (4 to 2M)
- **GEMM benchmarks**: `matrix_dim` (128 to 16384)
- **GEMV**: `matrix_dim` (256 to 65536)
- **Batched GEMM**: `matrix_dim` (32 to 4096)
- **FFT 1D**: `n` (1024 to 256M)
- **FFT 2D**: `grid_dim` (32 to 8192)
- **SpMV**: `num_rows` (256 to 1M)
- **Conv**: `input_hw` (28 to 1024)
- **BatchNorm**: `spatial_dim` (7 to 512)
- **Cutlass GEMM**: `matrix_dim` (128 to 16384)

### Note:
- .gitignore excluded tier3/*/* files; used `git add -f` to force-add
- No existing params.json files were present (fresh creation)
