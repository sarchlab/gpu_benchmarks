# Owen — Tier 1 Group A Verification Report

**Branch:** `ares/m22-cli-benchmarks`  
**Issue:** #172  
**Date:** 2026-03-20  

## Summary

**All 12 files (6 benchmarks × 2 platforms) pass all verification checks.** No issues found.

---

## Per-File Verification

### 1. fp32_throughput

#### fp32_throughput.hip ✅
- **Env vars:** Reads `BENCH_PARAM_fmas_per_thread`, `BENCH_PARAM_num_blocks`, `BENCH_PARAM_threads_per_block`, `BENCH_PARAM_num_iterations` — matches params.json (scaling + 3 non-scaling)
- **Defaults:** Falls back to DEFAULT_* constants (4096, 1024, 256, 10) matching params.json defaults
- **setvbuf:** `setvbuf(stdout, NULL, _IOLBF, 0)` at line 130, start of main ✅
- **JSON stdout:** Kernel event with `name:"fp32_fma_kernel"` (matches params.json kernels), `time_ms`, all 4 params. Summary event with `total_time_ms` and `metrics` array.
- **stderr only:** All human output via `fprintf(stderr, ...)`, no `std::cout`, no non-JSON `printf`
- **No old patterns:** No CSV remnants

#### fp32_throughput_metal.mm ✅
- **Env vars:** Same 4 env vars as .hip ✅
- **Defaults:** Same defaults ✅
- **setvbuf:** Line 86 ✅
- **JSON stdout:** Same format, `name:"fp32_fma_kernel"` ✅
- **stderr only:** All human output via `fprintf(stderr, ...)` ✅

### 2. fp64_throughput

#### fp64_throughput.hip ✅
- **Env vars:** `BENCH_PARAM_fmas_per_thread`, `BENCH_PARAM_num_blocks`, `BENCH_PARAM_threads_per_block`, `BENCH_PARAM_num_iterations` ✅
- **Defaults:** 4096, 1024, 256, 10 ✅
- **setvbuf:** Line 103 ✅
- **JSON stdout:** `name:"fp64_fma_kernel"` (matches params.json), all 4 params in kernel event ✅
- **stderr only:** ✅

#### fp64_throughput_metal.mm ✅
- **Env vars:** Same 4 ✅
- **setvbuf:** Line 78 ✅
- **JSON stdout:** `name:"fp64_fma_kernel"` ✅
- **stderr only:** ✅ (includes note about FP64 proxy on Apple GPUs — correctly on stderr)

### 3. fp16_throughput

#### fp16_throughput.hip ✅
- **Env vars:** `BENCH_PARAM_fmas_per_thread`, `BENCH_PARAM_num_blocks`, `BENCH_PARAM_threads_per_block`, `BENCH_PARAM_num_iterations` ✅
- **Defaults:** 4096, 1024, 256, 10 ✅
- **setvbuf:** Line 100 ✅
- **JSON stdout:** `name:"fp16_fma_kernel"` (matches params.json), all 4 params ✅
- **stderr only:** ✅

#### fp16_throughput_metal.mm ✅
- **Env vars:** Same 4 ✅
- **setvbuf:** Line 72 ✅
- **JSON stdout:** `name:"fp16_fma_kernel"` ✅
- **stderr only:** ✅

### 4. int32_throughput

#### int32_throughput.hip ✅
- **Env vars:** `BENCH_PARAM_mads_per_thread`, `BENCH_PARAM_num_blocks`, `BENCH_PARAM_threads_per_block`, `BENCH_PARAM_num_iterations` — matches params.json (scaling param is `mads_per_thread`, not `fmas_per_thread`) ✅
- **Defaults:** 4096, 1024, 256, 5 (params.json default for num_iterations is 5) ✅
- **setvbuf:** Line 92 ✅
- **JSON stdout:** `name:"int32_mad_kernel"` (matches params.json kernels), metric is "GIOPS" ✅
- **stderr only:** ✅

#### int32_throughput_metal.mm ✅
- **Env vars:** Same 4 ✅
- **Defaults:** Same (mads_per_thread=4096, num_iterations=5) ✅
- **setvbuf:** Line 71 ✅
- **JSON stdout:** `name:"int32_mad_kernel"` ✅
- **stderr only:** ✅

### 5. memory_bandwidth

#### memory_bandwidth.hip ✅
- **Env vars:** `BENCH_PARAM_buffer_size_mb`, `BENCH_PARAM_num_iterations`, `BENCH_PARAM_transfer_type`, `BENCH_PARAM_use_pinned_memory` — all 4 params from params.json ✅
- **Defaults:** 256 MB, 20 iterations, "all", "true" — matches params.json ✅
- **setvbuf:** Line 161 ✅
- **JSON stdout:** Kernel events for `memcpy_h2d`, `memcpy_d2h`, `memcpy_d2d` (all match params.json kernels). Summary with bandwidth metrics per direction. All params included in each kernel event. ✅
- **stderr only:** ✅

#### memory_bandwidth_metal.mm ✅
- **Env vars:** Same 4 ✅
- **Defaults:** Same ✅
- **setvbuf:** Line 263 ✅
- **JSON stdout:** Same kernel names, same format ✅
- **stderr only:** ✅

### 6. pcie_bandwidth

#### pcie_bandwidth.hip ✅
- **Env vars:** `BENCH_PARAM_buffer_size_mb`, `BENCH_PARAM_num_iterations`, `BENCH_PARAM_transfer_type`, `BENCH_PARAM_use_pinned_memory` — all 4 ✅
- **Defaults:** 256 MB, 20 iterations, "all", "true" ✅
- **setvbuf:** Line 191 ✅
- **JSON stdout:** Kernel events for `memcpy_h2d`, `memcpy_d2h`, `memcpy_bidir` (all match params.json kernels). Summary with bandwidth metrics. ✅
- **stderr only:** ✅

#### pcie_bandwidth_metal.mm ✅
- **Env vars:** Same 4 ✅
- **Defaults:** Same ✅
- **setvbuf:** Line 262 ✅
- **JSON stdout:** Same kernel names ✅
- **stderr only:** ✅ (includes note about unified memory — correctly on stderr)

---

## Runtime Validation (Metal builds on Apple M2)

### fp32_throughput ✅
```
$ BENCH_PARAM_fmas_per_thread=64 BENCH_PARAM_num_iterations=2 ./fp32_throughput 2>/dev/null
{"type":"kernel","name":"fp32_fma_kernel","time_ms":6.854521,"params":{"fmas_per_thread":64,"num_blocks":1024,"threads_per_block":256,"num_iterations":2}}
{"type":"summary","total_time_ms":13.709042,"metrics":[{"name":"GFLOPS","value":4.90}]}
```
- 2 JSON lines, both parse as valid JSON ✅
- Kernel event has correct fields (type, name, time_ms, params with all 4 params) ✅
- Summary event has total_time_ms and metrics array ✅
- stderr output goes to stderr only (confirmed by redirecting separately) ✅

### memory_bandwidth ✅
```
$ BENCH_PARAM_buffer_size_mb=4 BENCH_PARAM_num_iterations=2 ./memory_bandwidth 2>/dev/null
{"type":"kernel","name":"memcpy_h2d","time_ms":1.457729,"params":{"buffer_size_mb":4,"num_iterations":2,"transfer_type":"all","use_pinned_memory":"true"}}
{"type":"kernel","name":"memcpy_d2h","time_ms":3.333417,"params":{"buffer_size_mb":4,"num_iterations":2,"transfer_type":"all","use_pinned_memory":"true"}}
{"type":"kernel","name":"memcpy_d2d","time_ms":1.312292,"params":{"buffer_size_mb":4,"num_iterations":2,"transfer_type":"all","use_pinned_memory":"true"}}
{"type":"summary","total_time_ms":12.206875,"metrics":[{"name":"h2d_bandwidth_gbps","value":2.88},{"name":"d2h_bandwidth_gbps","value":1.26},{"name":"d2d_bandwidth_gbps","value":6.39}]}
```
- 4 JSON lines (3 kernels + 1 summary), all valid JSON ✅
- All kernel names match params.json (memcpy_h2d, memcpy_d2h, memcpy_d2d) ✅
- All 4 params included in each kernel event ✅

### pcie_bandwidth ✅
```
$ BENCH_PARAM_buffer_size_mb=4 BENCH_PARAM_num_iterations=2 BENCH_PARAM_transfer_type=h2d ./pcie_bandwidth 2>/dev/null
{"type":"kernel","name":"memcpy_h2d","time_ms":3.376854,"params":{"buffer_size_mb":4,"num_iterations":2,"transfer_type":"h2d","use_pinned_memory":"true"}}
{"type":"summary","total_time_ms":6.753708,"metrics":[{"name":"h2d_bandwidth_gbps","value":1.24}]}
```
- 2 JSON lines, valid JSON ✅
- Correctly filters to only H2D when transfer_type=h2d ✅
- Summary is always last line ✅

---

## Checklist Summary

| Check | Result |
|-------|--------|
| All 12 files present | ✅ |
| Env vars match params.json for all files | ✅ (4 params each for throughput benchmarks, 4 params each for bandwidth benchmarks) |
| Default fallbacks present | ✅ |
| setvbuf(stdout, NULL, _IOLBF, 0) at start of main | ✅ all 12 files |
| JSON-lines to stdout (kernel + summary events) | ✅ |
| Kernel name matches params.json kernels | ✅ |
| All params included in kernel events | ✅ |
| Summary is last JSON line with total_time_ms | ✅ |
| Human output to stderr only | ✅ |
| No std::cout usage | ✅ (0 occurrences across all 12 files) |
| No CSV remnants | ✅ |
| No non-JSON printf to stdout | ✅ (only JSON strings and JSON-building fragments like `","` and `"]}\n"`) |
| Runtime validation passes (3 benchmarks tested) | ✅ |

**Verdict: All 12 files are correctly refactored. No issues found.**
