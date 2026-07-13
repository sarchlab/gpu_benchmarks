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
  "export_version": "1.1",
  "exported_at": "2024-01-01T00:00:00Z",
  "sessions": [
    {
      "id": 1,
      "started_at": "2024-01-01T10:00:00",
      "finished_at": "2024-01-01T10:15:00",
      "status": "completed",
      "repetitions": 7,
      "platform": {
        "hostname": "my-workstation",
        "os": "Linux 6.1.0 x86_64",
        "gpu_model": "NVIDIA RTX 4090",
        "gpu_vram_gb": 24,
        "driver_version": "545.29.06",
        "platform_type": "cuda"
      },
      "runs": [
        {
          "benchmark": "fp32_throughput",
          "tier": 1,
          "suite": "compute_bound",
          "scaling_param_name": "fmas_per_thread",
          "scaling_param_value": "4096",
          "non_scaling_params": {"threads_per_block": 256},
          "repetition": 1,
          "status": "ok",
          "kernels": [
            {
              "name": "fp32_fma_kernel",
              "time_ms": 0.524128,
              "host_time_ms": 0.601342,
              "iteration": 0,
              "params": {"fmas_per_thread": 4096, "num_blocks": 1, "threads_per_block": 256}
            }
          ],
          "metrics": []
        }
      ]
    }
  ]
}
```

Each session embeds the `platform` it ran on, so an export spanning machines
faithfully attributes every session. A run is a single process execution (one
repetition); the `scaling_param_name`, `non_scaling_params`, and `repetition`
fields make runs that differ only by non-scaling parameters or repetition
distinguishable.

### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `export_version` | string | Schema version (currently `"1.1"`) |
| `exported_at` | string | ISO 8601 timestamp of when the export was generated |
| `sessions[].id` | number | Session ID from the database |
| `sessions[].started_at` | string | Session start timestamp |
| `sessions[].finished_at` | string \| null | Session end timestamp |
| `sessions[].status` | string | Session status: `running`, `completed`, `failed`, or `interrupted` |
| `sessions[].repetitions` | number | Number of repetitions per configuration |
| `sessions[].platform.hostname` | string | Machine hostname |
| `sessions[].platform.os` | string | Operating system and architecture |
| `sessions[].platform.gpu_model` | string | GPU model name |
| `sessions[].platform.gpu_vram_gb` | number \| null | GPU VRAM in GiB |
| `sessions[].platform.driver_version` | string \| null | GPU driver version |
| `sessions[].platform.platform_type` | string | One of `cuda`, `rocm` |
| `sessions[].runs[].benchmark` | string | Benchmark name |
| `sessions[].runs[].tier` | number | Benchmark tier (1–5) |
| `sessions[].runs[].suite` | string | Benchmark suite name |
| `sessions[].runs[].scaling_param_name` | string | Name of the scaling parameter |
| `sessions[].runs[].scaling_param_value` | string | Value of the scaling parameter |
| `sessions[].runs[].non_scaling_params` | object | Remaining benchmark parameters held fixed for this run (`{}` if none) |
| `sessions[].runs[].repetition` | number | 1-based repetition index of this run within its session |
| `sessions[].runs[].status` | string | Run status: `ok`, `error`, `timeout` |
| `sessions[].runs[].kernels` | array | Kernel timing results — one entry per timed launch, with `time_ms` (GPU event time), `host_time_ms` (host wall-clock, when reported), `iteration` (0-based index the benchmark reported), and `params` (kernel parameters, when reported). Benchmarks launch and time their kernel once per process; repetitions are separate runs. |
| `sessions[].runs[].metrics` | array | Computed metrics (e.g., GFLOPS). Empty for tier1, whose metrics are derived downstream from kernel times and params. |

## Guidelines for Comparable Results

- Run benchmarks on an **idle system** for consistent results.
- Use the **default repetitions** (7) unless you have a specific reason to change.
- Do not modify the JSON structure — the export command produces the correct format.
