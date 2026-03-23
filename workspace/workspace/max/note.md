# Max — Workspace Notes

## Last Completed: Issue #170 — Refactor Tier 1 Group B benchmarks

**Date:** 2026-03-19
**Branch:** ares/m22-cli-benchmarks
**Commit:** 43785de

### What was done
Refactored all 12 files (6 benchmarks × 2 files each) to the JSON-lines output protocol:
1. **mixbench** (.hip + _metal.mm) — params: buffer_size_mb, flops_per_byte, threads_per_block, timed_iterations
2. **shared_mem_bandwidth** (.hip + _metal.mm) — params: inner_iters, block_size, num_blocks, access_pattern (enum)
3. **cache_latency** (.hip + _metal.mm) — params: array_bytes, num_accesses, num_iters, rng_seed
4. **atomic_operations** (.hip + _metal.mm) — params: ops_per_thread, num_blocks, threads_per_block, atomic_op (enum)
5. **tensor_core_throughput** (.hip + _metal.mm) — params: matrix_dim, precision (enum), warmup_iterations, timed_iterations
6. **warp_shuffle** (.hip + _metal.mm) — params: ops_per_thread, num_blocks, threads_per_block, shuffle_variant (enum)

### Pattern used
- `setvbuf(stdout, NULL, _IOLBF, 0)` at top of main
- `getenv("BENCH_PARAM_xxx")` for each param, with argv fallback, then defaults
- `fprintf(stderr, ...)` for all human-readable output
- `printf("{\"type\":\"kernel\",...}\n")` for kernel events
- `printf("{\"type\":\"summary\",...}\n")` for summary
- Enum params control which kernels run (e.g., access_pattern="both" runs both, "no_conflict" runs only that one)
- No CSV output lines remain
