# Benchmark Output Protocol

## Overview

This document defines how benchmarks communicate results to the CLI tool (`gpubench`). The protocol uses **JSON-lines on stdout** — each line is one self-contained JSON object. Human-readable messages go to **stderr**.

The CLI tool captures stdout, parses each line as JSON, and stores the results in the SQLite database.

## Design Principles

1. **stdout is structured data only** — one JSON object per line, no blank lines, no text
2. **stderr is for humans** — progress messages, warnings, debug info
3. **Each line is self-contained** — parseable independently, no multi-line JSON
4. **Order matters** — kernel events should appear in execution order; summary comes last
5. **Failures are reported in-band** — error events on stdout so the CLI can record them

## Message Types

### 1. `kernel` — Kernel Execution Result

Emitted once per kernel invocation that is timed. This is the primary data record.

```json
{
  "type": "kernel",
  "name": "gemm_kernel",
  "time_ms": 12.345,
  "params": {
    "matrix_size": 1024,
    "block_size": 16
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | **yes** | Always `"kernel"` |
| `name` | string | **yes** | Kernel name (must match one of `kernels` in params.json) |
| `time_ms` | number | **yes** | Wall-clock or GPU-timer execution time in milliseconds. Must be ≥ 0 |
| `params` | object | **yes** | All parameter values used for this invocation (scaling + non-scaling) |

**Notes:**
- If a benchmark launches the same kernel multiple times (e.g., across iterations), it should emit one `kernel` event per timed invocation, or a single `kernel` event with the average time. The recommended approach is one event with the **averaged** time across internal iterations.
- `params` must include **all** parameters (both the scaling param and all non-scaling params with their current values).

### 2. `summary` — Run Summary

Emitted once at the end of a benchmark run. Provides aggregate metrics.

```json
{
  "type": "summary",
  "total_time_ms": 12.357,
  "metrics": [
    {"name": "GFLOPS", "value": 173.5},
    {"name": "bandwidth_gbps", "value": 42.1}
  ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | **yes** | Always `"summary"` |
| `total_time_ms` | number | **yes** | Total benchmark execution time in ms (sum of all kernel times or wall-clock) |
| `metrics` | array | no | Derived metrics computed by the benchmark |

Each metric object:

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
    "matrix_size": 16384,
    "block_size": 16
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | **yes** | Always `"error"` |
| `severity` | string | **yes** | `"fatal"` (benchmark cannot continue) or `"warning"` (non-fatal, some results may be missing) |
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

The CLI records these but does not require any specific fields.

## Complete Example Session

### stdout (captured by CLI)

```
{"type":"info","benchmark":"polybench_gemm","version":"1.0"}
{"type":"kernel","name":"gemm_kernel","time_ms":0.234,"params":{"matrix_size":256,"block_size":16,"alpha":1.0,"beta":1.0}}
{"type":"summary","total_time_ms":0.234,"metrics":[{"name":"GFLOPS","value":143.2}]}
```

### stderr (displayed to user)

```
[polybench_gemm] Platform: CUDA (NVIDIA A100-SXM4-80GB)
[polybench_gemm] matrix_size=256, block_size=16
[polybench_gemm] Warmup: 3 iterations
[polybench_gemm] Timing: 10 iterations, avg=0.234 ms
[polybench_gemm] GFLOPS: 143.2
```

## Protocol Rules

### Parsing Rules (CLI side)

1. Read stdout line by line
2. Skip empty lines (shouldn't occur, but be lenient)
3. Parse each line as JSON; if parse fails, log warning and skip
4. Dispatch on `type` field
5. If `type` is unknown, log warning and skip (forward compatibility)
6. After the process exits, check exit code:
   - Exit 0: success (all `kernel` + `summary` events are valid)
   - Exit non-zero: failure (record any `error` events; mark run as failed)

### Emission Rules (Benchmark side)

1. Print one JSON object per line to stdout — no pretty-printing, no newlines within
2. All human-readable output goes to stderr
3. Emit `kernel` events as they complete
4. Emit `summary` as the last event before exit
5. On fatal error, emit an `error` event and exit with non-zero code
6. On non-fatal error (e.g., one parameter value fails OOM), emit `error` with `"warning"` severity, continue with next configuration
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
| OOM on one param value | Emit `error` (warning), skip value, continue | Record error, continue sweep |
| GPU not found | Emit `error` (fatal), exit 1 | Record error, mark run failed |
| Benchmark crashes | No output (or partial) | Detect non-zero exit, mark run failed |
| Timeout | Killed by CLI | Mark run as timed out |
| Invalid parameter | Emit `error` (fatal), exit 1 | Record error, mark run failed |

## Encoding

- All output is UTF-8
- Numbers use JSON number format (no NaN, no Infinity — use `null` for unmeasurable values)
- Timestamps are not required in the protocol — the CLI adds them
- No trailing commas in JSON objects
