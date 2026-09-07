# Benchmark Output Protocol

## Overview

This document defines how benchmarks communicate results to the CLI tool (`gpubench`). The protocol uses **JSON-lines on stdout** — each line is one self-contained JSON object. Human-readable messages go to **stderr**.

The CLI captures stdout, parses and validates the events, and stores run results in SQLite. `info` events are accepted but discarded; they do not provide persistent provenance.

For Tier 1 and Tier 2, one process executes one complete algorithm. Algorithmic loops still run to completion; they are not measurement repetitions. `gpubench run --reps N` starts N fresh processes. Emit raw GPU event times and workload descriptors, then derive throughput or latency downstream. Use the minimum successful total GPU time across repetitions of the same configuration for the least-perturbed sample. Keep that run’s kernel breakdown together rather than summing independent per-kernel minima.

## Design Principles

1. **stdout is structured data only** — one JSON object per line, no blank lines, no text
2. **stderr is for humans** — progress messages, warnings, debug info
3. **Each line is self-contained** — parseable independently, no multi-line JSON
4. **Order is descriptive** — emit accumulated kernel records in first-use order, then the summary; record order is not a launch trace
5. **Failures are reported in-band** — error events on stdout so the CLI can record them

## Message Types

### 1. `kernel` — Kernel Execution Result

Emit one record per timed kernel name in the complete run. Accumulate GPU event times across all algorithmic launches of that kernel before emitting its record.

```json
{
  "type": "kernel",
  "name": "polybench_gemm_kernel",
  "time_ms": 12.345,
  "params": {
    "size": 1024
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | **yes** | Always `"kernel"` |
| `name` | string | **yes** | Kernel name (must match the manifest’s `kernels` list when nonempty) |
| `time_ms` | number | **yes** | Accumulated GPU event time in milliseconds for Tier 1/Tier 2; finite and ≥ 0 |
| `params` | object | **yes** | Actual workload parameters for this run (scaling + non-scaling, excluding harness measurement controls) |
| `host_time_ms` | number | no | Optional compatibility field, stored if supplied; finite and ≥ 0. It is not included in the GPU time total |
| `iter` | integer | no | Optional legacy annotation stored as `kernel_results.iteration`; it does not permit duplicate kernel records |

**Notes:**
- Emit **exactly one** `kernel` event per kernel per run. If a benchmark launches the same kernel multiple times (e.g., an iterative algorithm), it must accumulate the launch times and emit a single record with the summed time; the CLI rejects duplicate records for the same kernel name.
- `params` must include the scaling parameter and all non-scaling workload parameters with their actual values. The harness control `vector_add_warmup` is excluded from this required echo and is recorded in the run configuration.
- Tier 1/Tier 2 producers do not need host timing or `iter`. Compatibility fields do not change the one-record-per-name contract.
- Higher-tier producers may use their documented timing instrumentation, but all tiers must satisfy the validator’s finite-time and summary-sum rules.

### 2. `summary` — Run Summary

Emitted once at the end of a benchmark run. Provides the sum of the per-kernel GPU times.

```json
{
  "type": "summary",
  "total_time_ms": 12.345
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | **yes** | Always `"summary"` |
| `total_time_ms` | number | **yes** | Sum of all `kernel.time_ms` records; finite and ≥ 0. It excludes setup, verification, and host elapsed time |
| `metrics` | array | no | Legacy/higher-tier compatibility field, stored in `metrics`. Tier 1/Tier 2 omit it and derive metrics downstream |

If the compatibility `metrics` array is supplied, each metric object has:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | **yes** | Metric name (e.g., `"GFLOPS"`, `"bandwidth_gbps"`) |
| `value` | number | **yes** | Computed metric value |

### 3. `error` — Error Report

Emitted when the benchmark encounters a fatal or non-fatal error.

```json
{
  "type": "error",
  "severity": "fatal",
  "message": "CUDA out of memory: tried to allocate 8192 MiB",
  "params": {
    "size": 16384
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | **yes** | Always `"error"` |
| `severity` | string | **yes** | `"fatal"` (run failed) or `"warning"` (diagnostic that does not invalidate the result). For compatibility, an omitted severity is treated as fatal |
| `message` | string | **yes** | Human-readable error description |
| `params` | object | no | Parameters that triggered the error, if applicable |

### 4. `info` — Informational Metadata

Optional. Emitted at the start to report benchmark-specific metadata.

```json
{
  "type": "info",
  "benchmark": "fp32_throughput",
  "version": "1.0",
  "gpu_name": "NVIDIA A100-SXM4-80GB"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | **yes** | Always `"info"` |
| All other fields | any | no | Free-form key-value pairs for metadata |

The parser recognizes `info` and discards its fields. They are not stored in SQLite or included in exports, and do not update the CLI-detected `platforms` row. Use stderr or separately retained raw stdout when this metadata is needed for diagnosis. Persistent build/device provenance needs its own explicit storage contract.

## Complete Example Session

### stdout (captured by CLI)

```jsonl
{"type":"info","benchmark":"polybench_gemm","version":"1.0"}
{"type":"kernel","name":"polybench_gemm_kernel","time_ms":0.234,"params":{"size":256}}
{"type":"summary","total_time_ms":0.234}
```

### stderr (illustrative diagnostics, shown with `--verbose` or on failure)

```
[polybench_gemm] Platform: CUDA (NVIDIA A100-SXM4-80GB)
[polybench_gemm] size=256
[polybench_gemm] One complete algorithm: GPU time=0.234 ms
```

## Protocol Rules

### Parsing Rules (CLI side)

1. Read stdout line by line
2. Skip empty lines (shouldn't occur, but be lenient)
3. Parse each line as JSON; a line that fails to parse is a **protocol violation**
4. Dispatch on `type` field
5. An unknown `type`, or an event missing a required field, is a **protocol violation**
6. After the process exits, check exit code:
   - Exit non-zero: failure (record any `error` events; mark run as failed)
   - Exit 0: the run is marked `ok` only if the output passes full validation (below); any violation marks the run as failed, and the violations are recorded in its `error_message`

### Validation Rules (exit 0)

A run with exit code 0 is accepted only when all of the following hold:

1. No protocol violations from parsing (rules 3 and 5)
2. At least one `kernel` event with exactly one record per kernel name, and **exactly one** `summary` event with a numeric `total_time_ms`
3. All `time_ms`, `host_time_ms`, and `total_time_ms` values are finite and non-negative
4. When the manifest declares a nonempty `kernels` list, every emitted name must appear in it. An empty list disables name checking for compatibility; it does not waive the requirement to emit a timed record
5. Every `kernel` event carries a `params` object whose values match the requested configuration (the scaling param and all non-scaling params, excluding harness measurement controls such as `vector_add_warmup`)
6. `total_time_ms` equals the sum of all `kernel` event `time_ms` values, within a small floating-point tolerance
7. No fatal `error` event was emitted (a fatal error with exit 0 is a contradiction)

### Emission Rules (Benchmark side)

1. Print one JSON object per line to stdout — no pretty-printing, no newlines within
2. All human-readable output goes to stderr
3. Emit each accumulated `kernel` record after its final algorithmic launch
4. Emit `summary` as the last event before exit
5. On fatal error, emit an `error` event and exit with non-zero code
6. Execute one configuration per process. If it fails (including OOM), report a fatal error and exit nonzero; the CLI controls continuation to later configurations. Use warnings only when the complete reported result remains valid
7. Flush stdout after each line to ensure the CLI receives events promptly

### Parameter Contract

The CLI sets parameters via environment variables (`BENCH_PARAM_{name}={value}`). The benchmark:

1. Reads each `BENCH_PARAM_*` variable
2. Falls back to its own default if a variable is not set
3. Reports the **actual values used** in every `kernel` event's `params` field

This ensures the CLI can verify what parameters were actually used.

## Error Handling Matrix

| Scenario | Benchmark Action | CLI Action |
|----------|-----------------|------------|
| Normal execution | Emit `kernel` + `summary`, exit 0 | Record results |
| OOM for this configuration | Emit `error` (fatal), exit nonzero | Record failed run; continuation follows CLI policy (`--continue-on-error`) |
| GPU not found | Emit `error` (fatal), exit 1 | Record error, mark run failed |
| Benchmark crashes | No output (or partial) | Detect non-zero exit, mark run failed |
| Timeout | Killed by CLI | Mark run as timed out |
| Invalid parameter | Emit `error` (fatal), exit 1 | Record error, mark run failed |

## Encoding

- All output is UTF-8
- Numbers use JSON number format: no NaN or Infinity. Required times cannot be `null`; report a fatal error if a valid timing result cannot be produced
- Timestamps are not required in the protocol — the CLI adds them
- No trailing commas in JSON objects
