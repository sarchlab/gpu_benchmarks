# params.json Format Specification

## Overview

Every benchmark directory contains a `params.json` file that fully describes the benchmark's identity, parameters, and kernel structure. The CLI tool discovers benchmarks by walking `tier*/*/params.json` and uses these files to drive compilation, parameterized execution, and result recording.

## Schema

```json
{
  "benchmark": "<string>",
  "tier": "<integer>",
  "suite": "<string>",
  "description": "<string>",
  "language": "<string>",
  "scaling_param": { ... },
  "non_scaling_params": [ ... ],
  "kernels": [ "<string>", ... ],
  "metrics": [ ... ]
}
```

## Field Reference

### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `benchmark` | string | **yes** | Unique benchmark name matching directory name (e.g., `"fp32_throughput"`) |
| `tier` | integer | **yes** | Tier number: 1–5 |
| `suite` | string | **yes** | Suite name within the tier (e.g., `"microbench"`, `"rodinia"`, `"shoc"`, `"vendor_library"`, `"triton"`, `"ml_inference"`) |
| `description` | string | **yes** | Human-readable description of what the benchmark measures |
| `language` | string | no | `"cpp"` (default) or `"python"`. Determines compile/run strategy. Tier 1–3 default to `"cpp"`, Tier 4–5 default to `"python"` |
| `scaling_param` | object | **yes** | The single scaling parameter (see below) |
| `non_scaling_params` | array | **yes** | Zero or more non-scaling parameters (see below) |
| `kernels` | array | **yes** | List of kernel names emitted by this benchmark. May be empty `[]` for benchmarks that don't report kernel-level timing (e.g., pure memory transfers) |
| `metrics` | array | no | Optional list of metric descriptors (see below) |

### Scaling Parameter Object

The scaling parameter is the single parameter that controls the "size" of the benchmark workload. Performance should be predictable (linear, polynomial, or exponential) as a function of this parameter.

```json
{
  "name": "matrix_size",
  "description": "Square matrix dimension (N×N)",
  "type": "int",
  "values": [64, 128, 256, 512, 1024, 2048, 4096]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | **yes** | Parameter name, used as CLI flag and in output protocol |
| `description` | string | **yes** | Human-readable description |
| `type` | string | **yes** | One of: `"int"`, `"float"`, `"string"`, `"enum"` |
| `values` | array | **yes** | Ordered list of values to sweep. Should contain 20–30 values for scaling params |

For **Tier 1 and Tier 2 non-ML benchmarks**, choose scaling values that give good
coverage of the workload's behavior. There is no restriction on alignment or
magnitude — aligned (power-of-two) and non-aligned sizes are both worth sampling,
and the useful upper bound is workload- and device-dependent (a size that saturates
one GPU may not saturate another), so pick ranges that expose the behavior you care
about. Keep individual runs within the harness timeout.

### Non-Scaling Parameter Object

Non-scaling parameters are configuration knobs that affect execution but are not the primary scaling axis.

```json
{
  "name": "block_size",
  "description": "Thread block dimension",
  "type": "int",
  "default": 256,
  "values": [64, 128, 256, 512]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | **yes** | Parameter name |
| `description` | string | **yes** | Human-readable description |
| `type` | string | **yes** | One of: `"int"`, `"float"`, `"string"`, `"enum"` |
| `default` | any | **yes** | Default value used when not explicitly swept |
| `values` | array | **yes** | List of valid values (3–5 values recommended) |

### Metrics Array (Optional)

Describes the derived metrics the benchmark computes.

```json
{
  "metrics": [
    {"name": "GFLOPS", "description": "Billions of floating-point operations per second"},
    {"name": "bandwidth_gbps", "description": "Effective memory bandwidth in GB/s"}
  ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | **yes** | Metric name, matches `metric_name` in output protocol |
| `description` | string | **yes** | Human-readable description |

## Type Rules

- `"int"`: values must be JSON integers
- `"float"`: values must be JSON numbers (integer or decimal)
- `"string"`: values must be JSON strings
- `"enum"`: values must be JSON strings; semantically a fixed set of choices

## Parameter Passing Convention

The CLI passes parameters to benchmarks via environment variables:

```bash
BENCH_PARAM_matrix_size=1024 BENCH_PARAM_block_size=16 ./benchmark
```

Each parameter is passed as `BENCH_PARAM_{name}={value}`. Benchmarks read these environment variables at startup and fall back to their own hardcoded defaults if not set.

For Python benchmarks:
```bash
BENCH_PARAM_matrix_size=1024 BENCH_PARAM_block_size=16 python3 bench.py
```

## Complete Example

### Tier 1 Microbenchmark

```json
{
  "benchmark": "fp32_throughput",
  "tier": 1,
  "suite": "microbench",
  "description": "Measures peak single-precision (FP32) floating-point compute throughput using FMA instructions. Reports GFLOPS.",
  "scaling_param": {
    "name": "fmas_per_thread",
    "description": "Number of FMA operations per thread per kernel invocation",
    "type": "int",
    "values": [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536]
  },
  "non_scaling_params": [
    {
      "name": "num_blocks",
      "description": "Number of thread blocks in the grid",
      "type": "int",
      "default": 1024,
      "values": [256, 512, 1024, 2048, 4096]
    },
    {
      "name": "threads_per_block",
      "description": "Number of threads per block",
      "type": "int",
      "default": 256,
      "values": [64, 128, 256, 512]
    },
    {
      "name": "num_iterations",
      "description": "Number of timed kernel launches for averaging",
      "type": "int",
      "default": 10,
      "values": [5, 10, 20]
    }
  ],
  "kernels": ["fp32_fma_kernel"],
  "metrics": [
    {"name": "GFLOPS", "description": "Billions of FP32 FMA operations per second"}
  ]
}
```

### Tier 2 Application Benchmark

```json
{
  "benchmark": "rodinia_hotspot",
  "tier": 2,
  "suite": "rodinia",
  "description": "Iterative 2D stencil thermal simulation for chip temperature estimation.",
  "scaling_param": {
    "name": "grid_size",
    "description": "Grid dimension (NxN temperature grid)",
    "type": "int",
    "values": [64, 128, 192, 256, 384, 512, 640, 768, 1024, 1280, 1536, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192]
  },
  "non_scaling_params": [
    {
      "name": "block_size",
      "description": "Thread-block dimension (BxB threads)",
      "type": "int",
      "default": 16,
      "values": [8, 16, 32]
    },
    {
      "name": "num_iterations",
      "description": "Stencil time-steps per timed launch",
      "type": "int",
      "default": 10,
      "values": [1, 5, 10, 20, 50]
    }
  ],
  "kernels": ["hotspot_kernel"],
  "metrics": [
    {"name": "bandwidth_gbps", "description": "Effective memory bandwidth in GB/s"}
  ]
}
```

### Tier 5 ML Benchmark

```json
{
  "benchmark": "resnet50_train",
  "tier": 5,
  "suite": "ml_training",
  "language": "python",
  "description": "ResNet-50 training benchmark using PyTorch. Measures throughput in images/sec.",
  "scaling_param": {
    "name": "batch_size",
    "description": "Training batch size",
    "type": "int",
    "values": [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768]
  },
  "non_scaling_params": [
    {
      "name": "image_size",
      "description": "Input image dimension (HxW)",
      "type": "int",
      "default": 224,
      "values": [112, 224, 320]
    },
    {
      "name": "num_steps",
      "description": "Number of training steps to time",
      "type": "int",
      "default": 10,
      "values": [5, 10, 20]
    },
    {
      "name": "precision",
      "description": "Training precision mode",
      "type": "enum",
      "default": "fp32",
      "values": ["fp32", "fp16", "bf16"]
    }
  ],
  "kernels": ["forward", "backward", "optimizer_step"],
  "metrics": [
    {"name": "images_per_sec", "description": "Training throughput in images/second"}
  ]
}
```

## Validation Rules

A valid `params.json` must satisfy:

1. `benchmark` matches the directory name
2. `tier` matches the parent directory (e.g., `tier2/rodinia_hotspot/` → `tier: 2`)
3. `scaling_param.values` contains at least 10 values (recommended 20–30)
4. Each non-scaling param has `default` that appears in its `values` list
5. Each non-scaling param has 3–5 values
6. All `kernels` entries are unique strings
7. Parameter names are valid identifiers (`[a-z][a-z0-9_]*`)
8. No duplicate parameter names across scaling + non-scaling params
