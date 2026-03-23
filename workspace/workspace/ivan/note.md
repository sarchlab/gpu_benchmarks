# Ivan Workspace Notes

## Completed: Issue #73 — Add CSV output to all 6 Tier 1 microbenchmarks

**Branch:** `ares/m12-harness`
**Commit:** `4f50b53`

### What was done
Added CSV output (`printf`) to all 12 source files (6 benchmarks × 2 files each: .hip and _metal.mm):

1. **memory_bandwidth** — 3 CSV lines (h2d, d2h, d2d), format: `memory_bandwidth_<dir>,<size_mb>,<time_ms>,<gbs>`
2. **fp32_throughput** — 1 CSV line: `fp32_throughput,<n_ops>,<time_ms>,<gflops>`
3. **fp64_throughput** — 1 CSV line: `fp64_throughput,<n_ops>,<time_ms>,<gflops>`
4. **fp16_throughput** — 1 CSV line: `fp16_throughput,<n_ops>,<time_ms>,<gflops>`
5. **cache_latency** — 3 CSV lines (l1, l2, global_mem): `cache_latency_<level>,<array_bytes>,<time_ns>,<ns_per_access>`
6. **pcie_bandwidth** — 3 CSV lines (h2d, d2h, bidir): `pcie_bandwidth_<dir>,<size_mb>,<time_ms>,<gbs>`

All existing human-readable output preserved. No Makefiles or READMEs changed.
Issue #73 closed.
