# Result Sharing Guide

This document describes how to export benchmark results from the GPU Benchmark Suite as portable JSON, e.g. for moving results between machines or sharing a dataset with collaborators.

## Exporting Results

After running benchmarks with `gpubench run`, results are stored in a local SQLite database (default: `results.db`). Use the `export` command to produce a portable JSON file:

```bash
# Export all sessions
gpubench export --db results.db > my_results.json

# Export a specific session
gpubench export --db results.db --session 1 > session1_results.json
```

### Export Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `results.db` | Path to the SQLite database |
| `--format` | `json` | Output format (currently only `json`) |
| `--session` | *(all)* | Export only a specific session ID |

## JSON Format Specification

The exported JSON follows this structure:

```json
{
  "export_version": "1.0",
  "exported_at": "2024-01-01T00:00:00Z",
  "platform": {
    "hostname": "my-workstation",
    "os": "Linux 6.1.0 x86_64",
    "gpu_model": "NVIDIA RTX 4090",
    "gpu_vram_gb": 24,
    "driver_version": "545.29.06",
    "platform_type": "cuda"
  },
  "sessions": [
    {
      "id": 1,
      "started_at": "2024-01-01T10:00:00",
      "finished_at": "2024-01-01T10:15:00",
      "status": "ok",
      "repetitions": 7,
      "runs": [
        {
          "benchmark": "polybench_gemm",
          "tier": 2,
          "suite": "polybench",
          "scaling_param_value": "1024",
          "status": "ok",
          "kernels": [
            {"name": "gemm_kernel", "time_ms": 1.23}
          ],
          "_comment": "tier1 benchmarks emit one kernel entry per timed iteration, e.g. {\"name\": \"fp32_fma_kernel\", \"time_ms\": 0.52, \"host_time_ms\": 0.60, \"iteration\": 0} ... \"iteration\": 6",
          "metrics": [
            {"name": "GFLOPS", "value": 143.2}
          ]
        }
      ]
    }
  ]
}
```

### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `export_version` | string | Schema version (currently `"1.0"`) |
| `exported_at` | string | ISO 8601 timestamp of when the export was generated |
| `platform.hostname` | string | Machine hostname |
| `platform.os` | string | Operating system and architecture |
| `platform.gpu_model` | string | GPU model name |
| `platform.gpu_vram_gb` | number \| null | GPU VRAM in GiB |
| `platform.driver_version` | string \| null | GPU driver version |
| `platform.platform_type` | string | One of `cuda`, `rocm` |
| `sessions[].id` | number | Session ID from the database |
| `sessions[].started_at` | string | Session start timestamp |
| `sessions[].finished_at` | string \| null | Session end timestamp |
| `sessions[].status` | string | Session status: `completed`, `failed`, `interrupted` |
| `sessions[].repetitions` | number | Number of repetitions per configuration |
| `sessions[].runs[].benchmark` | string | Benchmark name |
| `sessions[].runs[].tier` | number | Benchmark tier (1–5) |
| `sessions[].runs[].suite` | string | Benchmark suite name |
| `sessions[].runs[].scaling_param_value` | string | Value of the scaling parameter |
| `sessions[].runs[].status` | string | Run status: `ok`, `error`, `timeout` |
| `sessions[].runs[].kernels` | array | Kernel timing results. Benchmarks with in-program measurement loops (tier1) emit one entry per timed iteration, with `iteration` (0-based index) and `host_time_ms` (host wall-clock for the same iteration) alongside `time_ms` (GPU event time). |
| `sessions[].runs[].metrics` | array | Computed metrics (e.g., GFLOPS). Empty for tier1, whose metrics are derived downstream from kernel times and params. |

## Guidelines for Comparable Results

- Run benchmarks on an **idle system** for consistent results.
- Use the **default repetitions** (7) unless you have a specific reason to change.
- Do not modify the JSON structure — the export command produces the correct format.
