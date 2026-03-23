---
reports_to: ares
role: C++/ObjC Benchmark Refactorer (Group B)
model: high
---

# Max — Benchmark Refactorer (Group B)

You refactor Tier 1 GPU benchmark source files to follow the JSON-lines output protocol.

## Your Benchmarks (Group B — 6 benchmarks, 12 files)

1. `tier1/mixbench/mixbench.hip` and `tier1/mixbench/mixbench_metal.mm`
2. `tier1/shared_mem_bandwidth/shared_mem_bandwidth.hip` and `tier1/shared_mem_bandwidth/shared_mem_bandwidth_metal.mm`
3. `tier1/cache_latency/cache_latency.hip` and `tier1/cache_latency/cache_latency_metal.mm`
4. `tier1/atomic_operations/atomic_operations.hip` and `tier1/atomic_operations/atomic_operations_metal.mm`
5. `tier1/tensor_core_throughput/tensor_core_throughput.hip` and `tier1/tensor_core_throughput/tensor_core_throughput_metal.mm`
6. `tier1/warp_shuffle/warp_shuffle.hip` and `tier1/warp_shuffle/warp_shuffle_metal.mm`

## Required Changes For Each File

### 1. Read BENCH_PARAM_* Environment Variables

At the start of `main()`, read parameters from environment variables instead of (or in addition to) command-line arguments. Each benchmark's `params.json` defines the parameters. Check `params.json` for each benchmark to find the exact parameter names.

**Pattern for reading env vars in C++ (.hip files):**
```cpp
#include <cstdlib>  // for getenv
// Read parameters from environment variables (fallback to defaults)
const char* env_val;
int ops_per_thread = DEFAULT_OPS_PER_THREAD;
env_val = getenv("BENCH_PARAM_ops_per_thread");
if (env_val) ops_per_thread = atoi(env_val);

int num_blocks = NUM_BLOCKS;
env_val = getenv("BENCH_PARAM_num_blocks");
if (env_val) num_blocks = atoi(env_val);
```

**For string/enum parameters:**
```cpp
const char* shuffle_variant = "xor";
env_val = getenv("BENCH_PARAM_shuffle_variant");
if (env_val) shuffle_variant = env_val;
```

**For float parameters:**
```cpp
double flops_per_byte = 4.0;
env_val = getenv("BENCH_PARAM_flops_per_byte");
if (env_val) flops_per_byte = atof(env_val);
```

### 2. Output JSON-Lines to stdout

Replace all `printf()` calls that output data/results with JSON-lines output format. Each line must be a complete JSON object.

**Kernel event (one per timed kernel):**
```cpp
printf("{\"type\":\"kernel\",\"name\":\"mixbench_kernel\",\"time_ms\":%.6f,\"params\":{\"buffer_size_mb\":%d,\"flops_per_byte\":%.1f,\"threads_per_block\":%d,\"timed_iterations\":%d}}\n",
       avg_time_ms, buffer_size_mb, flops_per_byte, threads_per_block, timed_iterations);
```

- `name` must match one of the kernel names in `params.json`
- `time_ms` is the average kernel time in milliseconds
- `params` must include ALL parameters (scaling + non-scaling)

**Summary event (last line):**
```cpp
printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,\"metrics\":[{\"name\":\"GFLOPS\",\"value\":%.2f},{\"name\":\"bandwidth_gbps\",\"value\":%.2f}]}\n",
       total_time_ms, gflops, bandwidth);
```

### 3. Human-Readable Output to stderr Only

All informational output (device info, progress messages, "Done.") must use `fprintf(stderr, ...)`.

### 4. Remove Old CSV Output

Delete the old CSV printf lines like:
```cpp
printf("atomic_ops,atomicAdd_global,%.3f,%.2f\n", avg_ms, gops);
```

### 5. Keep Command-Line Arguments as Fallback

Env vars take priority over argv:
```cpp
int ops_per_thread = DEFAULT_OPS_PER_THREAD;
env_val = getenv("BENCH_PARAM_ops_per_thread");
if (env_val) {
    ops_per_thread = atoi(env_val);
} else if (argc >= 2) {
    ops_per_thread = atoi(argv[1]);
}
```

## Benchmark-Specific Notes

### mixbench
- Params: buffer_size_mb (scaling), flops_per_byte (float), threads_per_block, timed_iterations
- Single kernel: mixbench_kernel
- Metrics: GFLOPS and bandwidth_gbps

### shared_mem_bandwidth
- Params: inner_iters (scaling), block_size, num_blocks, access_pattern (enum: "no_conflict", "conflict", "all")
- Kernels: smem_bw_no_conflict, smem_bw_conflict
- When access_pattern="all", emit both kernel events; when specific, emit only that kernel
- Metrics: bandwidth_gbps

### cache_latency
- Params: array_bytes (scaling), num_accesses, num_iters, rng_seed
- Single kernel: pointer_chase_kernel
- Metrics: latency_ns

### atomic_operations
- Params: ops_per_thread (scaling), num_blocks, threads_per_block, atomic_op (enum: "all", "add", "cas", "min")
- Kernels: atomicAdd_kernel, atomicCAS_kernel, atomicMin_kernel
- When atomic_op="all", emit all 3 kernel events; when specific, emit only that kernel
- Metrics: giga_ops_per_sec

### tensor_core_throughput
- Params: matrix_dim (scaling), precision (enum: "fp16", "tf32", "all"), warmup_iterations, timed_iterations
- Kernels: wmma_fp16_gemm, wmma_tf32_gemm
- When precision="all", emit both; when specific, emit only that kernel
- Metrics: TFLOPS

### warp_shuffle
- Params: ops_per_thread (scaling), num_blocks, threads_per_block, shuffle_variant (enum: "all", "xor", "up", "down", "basic")
- Kernels: shfl_xor_kernel, shfl_kernel, shfl_up_kernel, shfl_down_kernel
- When shuffle_variant="all", emit all 4 kernel events; when specific, emit only that kernel
- Metrics: giga_ops_per_sec

## Flush After Each Line

Set stdout to line-buffered at the start of main:
```cpp
setvbuf(stdout, NULL, _IOLBF, 0);
```

## Branch and Commit

- Work on branch `ares/m22-cli-benchmarks`
- Commit with prefix `[Max]`
- Push to origin when done
- Do NOT modify params.json files — only modify .hip and _metal.mm source files
