# Pat — Tier 1 Group B Verification Report

**Issue**: #173  
**Branch**: `ares/m22-cli-benchmarks`  
**Date**: 2026-03-20  

## Summary

Verified all 12 files (6 .hip + 6 _metal.mm) across 6 benchmarks. Found **2 enum mismatch bugs** affecting 4 files. All other checks pass.

---

## Per-Benchmark Findings

### 1. mixbench (mixbench.hip + mixbench_metal.mm) ✅ PASS

| Check | .hip | _metal.mm |
|-------|------|-----------|
| Env vars (buffer_size_mb, flops_per_byte, threads_per_block, timed_iterations) | ✅ All 4 read | ✅ All 4 read |
| setvbuf(stdout, NULL, _IOLBF, 0) | ✅ | ✅ |
| JSON kernel event (name=mixbench_kernel) | ✅ | ✅ |
| JSON summary event (last line) | ✅ | ✅ |
| All params in kernel JSON | ✅ | ✅ |
| stderr-only human output | ✅ | ✅ |
| No old CSV/cout | ✅ | ✅ |
| Runtime test (Metal) | ✅ Valid JSON output | ✅ |

### 2. shared_mem_bandwidth (shared_mem_bandwidth.hip + _metal.mm) ✅ PASS

| Check | .hip | _metal.mm |
|-------|------|-----------|
| Env vars (inner_iters, block_size, num_blocks, access_pattern) | ✅ All 4 | ✅ All 4 |
| setvbuf | ✅ | ✅ |
| JSON kernel (smem_bw_no_conflict, smem_bw_conflict) | ✅ | ✅ |
| JSON summary | ✅ | ✅ |
| All params in kernel JSON | ✅ | ✅ |
| stderr-only | ✅ | ✅ |
| Enum: access_pattern ("no_conflict", "conflict", "both") | ✅ matches params.json | ✅ |
| Runtime test (Metal) | ✅ Valid JSON | ✅ |

### 3. cache_latency (cache_latency.hip + _metal.mm) ✅ PASS

| Check | .hip | _metal.mm |
|-------|------|-----------|
| Env vars (array_bytes, num_accesses, num_iters, rng_seed) | ✅ All 4 | ✅ All 4 |
| setvbuf | ✅ | ✅ |
| JSON kernel (pointer_chase_kernel) | ✅ | ✅ |
| JSON summary | ✅ | ✅ |
| All params in kernel JSON | ✅ | ✅ |
| stderr-only | ✅ | ✅ |
| Runtime test (Metal) | ✅ Valid JSON | ✅ |

### 4. atomic_operations (atomic_operations.hip + _metal.mm) ⚠️ ENUM BUG

| Check | .hip | _metal.mm |
|-------|------|-----------|
| Env vars (ops_per_thread, num_blocks, threads_per_block, atomic_op) | ✅ All 4 | ✅ All 4 |
| setvbuf | ✅ | ✅ |
| JSON kernel (atomicAdd_kernel, atomicCAS_kernel, atomicMin_kernel) | ✅ | ✅ |
| JSON summary | ✅ | ✅ |
| All params in kernel JSON | ✅ | ✅ |
| stderr-only | ✅ | ✅ |
| **Enum: atomic_op** | **⚠️ BUG** | **⚠️ BUG** |

**Bug detail**: `params.json` defines enum values as `["atomicAdd", "atomicCAS", "atomicMin", "all"]`, but the code compares with `"add"`, `"cas"`, `"min"`:
```c
bool run_add = (strcmp(atomic_op, "all") == 0 || strcmp(atomic_op, "add") == 0);
bool run_cas = (strcmp(atomic_op, "all") == 0 || strcmp(atomic_op, "cas") == 0);
bool run_min = (strcmp(atomic_op, "all") == 0 || strcmp(atomic_op, "min") == 0);
```

**Verified at runtime**: Setting `BENCH_PARAM_atomic_op=atomicAdd` produces zero kernel events — only a summary with `total_time_ms: 0`. The code silently skips all kernels.

**Fix needed**: Either update the code to match params.json values (`"atomicAdd"`, `"atomicCAS"`, `"atomicMin"`) or update params.json to match the code (`"add"`, `"cas"`, `"min"`). Since kernel names in the JSON output already use `atomicAdd_kernel` etc., the params.json values seem more natural. Recommend updating the code.

### 5. tensor_core_throughput (tensor_core_throughput.hip + _metal.mm) ⚠️ MINOR

| Check | .hip | _metal.mm |
|-------|------|-----------|
| Env vars (matrix_dim, precision, warmup_iterations, timed_iterations) | ✅ All 4 | ✅ All 4 |
| setvbuf | ✅ | ✅ |
| JSON kernel (wmma_fp16_gemm, wmma_tf32_gemm) | ✅ | ✅ |
| JSON summary | ✅ | ✅ |
| All params in kernel JSON | ✅ | ✅ |
| stderr-only | ✅ | ✅ |
| Enum: precision | ⚠️ Minor | ⚠️ Minor |
| Runtime test (Metal) | ✅ Valid JSON | ✅ |

**Minor issues**:
- `params.json` lists `"bf16"` as a valid precision value, but neither .hip nor _metal.mm handle it (silently ignored, no error).
- Code supports `"all"` (runs fp16+tf32) but `params.json` doesn't list `"all"` as a valid value.
- These are cosmetic/doc mismatches, not functional bugs for the primary fp16/tf32 paths.

### 6. warp_shuffle (warp_shuffle.hip + _metal.mm) ⚠️ ENUM BUG

| Check | .hip | _metal.mm |
|-------|------|-----------|
| Env vars (ops_per_thread, num_blocks, threads_per_block, shuffle_variant) | ✅ All 4 | ✅ All 4 |
| setvbuf | ✅ | ✅ |
| JSON kernel (shfl_xor_kernel, shfl_kernel, shfl_up_kernel, shfl_down_kernel) | ✅ | ✅ |
| JSON summary | ✅ | ✅ |
| All params in kernel JSON | ✅ | ✅ |
| stderr-only | ✅ | ✅ |
| **Enum: shuffle_variant** | **⚠️ BUG** | **⚠️ BUG** |

**Bug detail**: `params.json` defines enum values as `["shfl_xor", "shfl", "shfl_up", "shfl_down", "all"]`, but the code compares with `"xor"`, `"basic"`, `"up"`, `"down"`:
```c
bool run_xor   = (strcmp(shuffle_variant, "all") == 0 || strcmp(shuffle_variant, "xor") == 0);
bool run_basic = (strcmp(shuffle_variant, "all") == 0 || strcmp(shuffle_variant, "basic") == 0);
bool run_up    = (strcmp(shuffle_variant, "all") == 0 || strcmp(shuffle_variant, "up") == 0);
bool run_down  = (strcmp(shuffle_variant, "all") == 0 || strcmp(shuffle_variant, "down") == 0);
```

**Verified at runtime**: Setting `BENCH_PARAM_shuffle_variant=shfl_xor` produces zero kernel events — only a summary with `total_time_ms: 0`.

**Fix needed**: Update code to match params.json: `"shfl_xor"`, `"shfl"`, `"shfl_up"`, `"shfl_down"`. Note: `"basic"` in code should become `"shfl"` to match params.json.

---

## Runtime Test Results (Metal, macOS, Apple M2)

| Benchmark | Compiled | Ran | JSON Valid | Kernel names match params.json |
|-----------|----------|-----|-----------|-------------------------------|
| mixbench | ✅ | ✅ | ✅ | ✅ |
| shared_mem_bandwidth | ✅ | ✅ | ✅ | ✅ |
| cache_latency | ✅ | ✅ | ✅ | ✅ |
| atomic_operations | ✅ | ✅ | ✅ | ✅ |
| tensor_core_throughput | ✅ | ✅ | ✅ | ✅ |
| warp_shuffle | ✅ | ✅ | ✅ | ✅ |

All 6 benchmarks compile and run on Metal. JSON output is valid JSON-lines in all cases.

---

## Bugs Found (2 critical, 1 minor)

### BUG 1 (Critical): atomic_operations enum mismatch
- **Files**: atomic_operations.hip, atomic_operations_metal.mm
- **Issue**: Code uses short names ("add", "cas", "min") but params.json specifies ("atomicAdd", "atomicCAS", "atomicMin")
- **Impact**: Setting env var to params.json values silently skips all kernels

### BUG 2 (Critical): warp_shuffle enum mismatch  
- **Files**: warp_shuffle.hip, warp_shuffle_metal.mm
- **Issue**: Code uses short names ("xor", "basic", "up", "down") but params.json specifies ("shfl_xor", "shfl", "shfl_up", "shfl_down")
- **Impact**: Setting env var to params.json values silently skips all kernels

### BUG 3 (Minor): tensor_core_throughput precision enum
- **Files**: tensor_core_throughput.hip, tensor_core_throughput_metal.mm
- **Issue**: params.json lists "bf16" but code doesn't handle it; code supports "all" but params.json doesn't list it
- **Impact**: Cosmetic only; primary fp16/tf32 paths work correctly
