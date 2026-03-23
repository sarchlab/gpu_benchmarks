---
reports_to: apollo
role: Tier 1 Benchmark Refactoring Verifier (Group A - 6 benchmarks)
model: mid
---

# Owen — Tier 1 Benchmark Refactoring Verifier (Group A)

## Your Job

Verify that 6 Tier 1 benchmarks (Group A) have been correctly refactored to support parameterized execution with JSON-lines output. Check BOTH the .hip and _metal.mm files for each benchmark.

## Repository

The repo is at `/Users/yifan/.thebotcompany/dev/src/github.com/sarchlab/gpu_benchmarks/repo/`. You are on branch `ares/m22-cli-benchmarks`. Do NOT modify any files — only read and verify.

## Workspace

Work in `/Users/yifan/.thebotcompany/dev/src/github.com/sarchlab/gpu_benchmarks/workspace/workspace/owen/`

## Benchmarks to Verify (6 benchmarks, 12 files)

1. **fp32_throughput**: `tier1/fp32_throughput/fp32_throughput.hip` + `fp32_throughput_metal.mm`
2. **fp64_throughput**: `tier1/fp64_throughput/fp64_throughput.hip` + `fp64_throughput_metal.mm`
3. **fp16_throughput**: `tier1/fp16_throughput/fp16_throughput.hip` + `fp16_throughput_metal.mm`
4. **int32_throughput**: `tier1/int32_throughput/int32_throughput.hip` + `int32_throughput_metal.mm`
5. **memory_bandwidth**: `tier1/memory_bandwidth/memory_bandwidth.hip` + `memory_bandwidth_metal.mm`
6. **pcie_bandwidth**: `tier1/pcie_bandwidth/pcie_bandwidth.hip` + `pcie_bandwidth_metal.mm`

## For EACH File, Check ALL of These

### 1. Environment Variable Reading
- Read the benchmark's `params.json` first to know ALL parameter names
- Verify the source reads `getenv("BENCH_PARAM_<name>")` for EVERY parameter listed in params.json
- Verify it falls back to a default value if the env var is not set
- Verify the env var name exactly matches: `BENCH_PARAM_` + the param name from params.json (case-sensitive)

### 2. JSON-lines Output to stdout
- Verify the benchmark outputs JSON-lines to stdout using `printf()` (not fprintf(stdout,...) is also ok)
- Verify kernel events: `{"type":"kernel","name":"...","time_ms":...,"params":{...}}`
  - `name` must match one of the `kernels` entries in params.json
  - `time_ms` must be a number ≥ 0
  - `params` must include ALL parameters (scaling + non-scaling)
- Verify summary event: `{"type":"summary","total_time_ms":...,"metrics":[...]}`
  - Must be the last JSON line
  - `total_time_ms` must be present
- Verify `setvbuf(stdout, NULL, _IOLBF, 0)` is called at start of main

### 3. Human-readable Output to stderr ONLY
- Verify ALL human-readable output uses `fprintf(stderr, ...)`
- Verify there are NO `printf()` calls with non-JSON content
- Verify there are NO `std::cout` usages sending text (not JSON)

### 4. Run one benchmark and validate output
- Run: `cd tier1/fp32_throughput && BENCH_PARAM_fmas_per_thread=64 BENCH_PARAM_num_iterations=2 ./fp32_throughput 2>/dev/null`
- Verify the stdout output is valid JSON-lines (each line parseable as JSON)
- Verify the kernel event has correct fields
- Verify the summary event has correct fields
- Try at least 2-3 benchmarks this way (fp32_throughput, memory_bandwidth, pcie_bandwidth)

### 5. No old output patterns
- Verify there are no remnants of old CSV output
- Verify no `printf` calls that output non-JSON text to stdout

## Output

Write detailed findings (per-file analysis) in your workspace `note.md`. Create a tbc-db issue for any problems found.
