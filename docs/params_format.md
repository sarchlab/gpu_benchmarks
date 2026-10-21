# params.json Format Specification

## Overview

Every benchmark directory contains a `params.json` file that fully describes the benchmark's identity, parameters, and kernel structure. The CLI tool discovers benchmarks by walking `tier*/*/params.json` and, for benchmarks with one source tree per platform, `tier*/*/hip/params.json` and `tier*/*/cuda/params.json` (it uses `cuda/` for `--platform cuda` and `hip/` for `rocm`; the two manifests must be identical). It uses these files to drive compilation, parameterized execution, and result recording.

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
| `kernels` | array | **yes** | List of kernel names emitted by this benchmark. An empty `[]` disables output name checking for compatibility, but a successful run still needs at least one timed `kernel` event |
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
| `values` | array | **yes** | Nonempty ordered list of distinct values to sweep. Numeric values must increase; categorical values retain their declared order. Typically 20–30 values |

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

Describes metrics emitted by legacy or higher-tier producers. Tier 1/Tier 2 report raw times and workload descriptors, omit this array, and compute derived metrics downstream. The runner does not compute metrics from these descriptors.

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
| `name` | string | **yes** | Metric name, matches the metric object’s `name` in the output protocol |
| `description` | string | **yes** | Human-readable description |

## Type Rules

- `"int"`: values must be JSON integers
- `"float"`: values must be JSON numbers (integer or decimal)
- `"string"`: values must be JSON strings
- `"enum"`: values must be JSON strings; semantically a fixed set of choices

`gpubench validate` checks manifest structure as well as value types. Wrong
container or field types, non-object parameter/metric entries, empty scaling
sweeps, and descending numeric scaling values are errors. A short, nonempty
sweep remains valid with an advisory warning.

Validation scans manifest files directly so malformed JSON and structures are
reported even when normal benchmark discovery cannot load them. Tier and
benchmark filters select directories. A valid suite field can exclude another
suite; files whose suite cannot be read are retained for diagnostics within
the selected directories.

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

This example uses the current FP32 workload controls and sweep. Work inside the kernel is controlled by `fmas_per_thread`; statistical repetitions use fresh processes via `gpubench run --reps N`.

```json
{
  "benchmark": "fp32_throughput",
  "tier": 1,
  "suite": "microbench",
  "description": "One FP32 FMA kernel execution; emits raw GPU time and work descriptors for downstream throughput analysis.",
  "scaling_param": {
    "name": "fmas_per_thread",
    "description": "Number of FMA operations per thread per kernel invocation; controls total compute work",
    "type": "int",
    "values": [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
  },
  "non_scaling_params": [
    {
      "name": "num_blocks",
      "description": "Number of thread blocks in the grid",
      "type": "int",
      "default": 1,
      "values": [1]
    },
    {
      "name": "threads_per_block",
      "description": "Number of threads per block",
      "type": "int",
      "default": 256,
      "values": [1, 32, 64, 256, 1024]
    },
    {
      "name": "vector_add_warmup",
      "description": "Harness measurement control: run the clock/cache warmer once before the complete measured algorithm",
      "type": "enum",
      "default": "true",
      "values": ["true", "false"]
    }
  ],
  "kernels": ["fp32_fma_kernel"]
}
```

### Tier 2 Application Benchmark

`total_iterations` is part of the thermal simulation, not a repetition count. The benchmark completes all time steps and emits one accumulated `calculate_temp` record. `pyramid_height` controls how many time steps each algorithmic launch advances.

```json
{
  "benchmark": "rodinia_hotspot",
  "tier": 2,
  "suite": "rodinia",
  "description": "Complete pyramidal thermal simulation; emits accumulated GPU time for all algorithmic launches.",
  "scaling_param": {
    "name": "grid_size",
    "description": "Grid dimension (NxN temperature grid)",
    "type": "int",
    "values": [64, 128, 192, 256, 384, 512, 640, 768, 1024, 1280, 1536, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192]
  },
  "non_scaling_params": [
    {
      "name": "total_iterations",
      "description": "Total thermal time steps simulated per pass",
      "type": "int",
      "default": 360,
      "values": [360]
    },
    {
      "name": "pyramid_height",
      "description": "Time steps advanced per pyramidal launch (must be < BLOCK_SIZE/2 = 8)",
      "type": "int",
      "default": 2,
      "values": [1, 2, 4]
    },
    {
      "name": "vector_add_warmup",
      "description": "Harness measurement control: run the clock/cache warmer once before the complete measured algorithm",
      "type": "enum",
      "default": "true",
      "values": ["true", "false"]
    }
  ],
  "kernels": ["calculate_temp"]
}
```

### Tier 5 ML Benchmark

This higher-tier example illustrates the optional metric descriptors; it does not define the Tier 1/Tier 2 timing methodology.

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
2. `tier` matches the parent directory (e.g., `tier2/rodinia_hotspot/` → `tier: 2`); for a platform tree such as `tier2/rodinia_hotspot/cuda/`, `benchmark` and `tier` are checked against `rodinia_hotspot` and `tier2`
3. Each non-scaling param has `default` that appears in its `values` list
4. All `kernels` entries are unique strings
5. Parameter names are valid identifiers (`[a-z][a-z0-9_]*`)
6. No duplicate parameter names across scaling + non-scaling params

### Value-count guidance (advisory, not enforced as errors)

The value *counts* are empirical rules of thumb, not hard requirements — a benchmark whose
meaningful range is narrower may declare fewer values:

- `scaling_param.values`: typically 20–30 ordered values, ideally at least 10, spanning
  cache-resident/under-saturating sizes up to sizes that exceed the last-level cache.
- Non-scaling params: typically 3–5 values each.

`gpubench validate` reports counts below the guidance as warnings (exit code stays 0);
everything under "must satisfy" above is an error.
