# CLI Design — `gpubench`

## Overview

`gpubench` is a Python CLI tool (no external dependencies beyond stdlib + sqlite3) that compiles, runs, and collects results from the GPU benchmark suite. It discovers benchmarks by walking `tier*/*/params.json`, drives parameterized execution, and stores all results in an SQLite database.

**Location:** `scripts/gpubench` (executable Python script)

## Commands

### `gpubench list`

List available benchmarks discovered from `params.json` files.

```
Usage: gpubench list [--tier T] [--suite S] [--json]

Options:
  --tier T      Filter by tier number (1-5)
  --suite S     Filter by suite name
  --json        Output as JSON array instead of table

Examples:
  gpubench list
  gpubench list --tier 2
  gpubench list --suite rodinia
  gpubench list --tier 1 --json
```

**Output (default table):**

```
Tier  Suite            Benchmark              Scaling Param     Kernels  Lang
----  -----            ---------              -------------     -------  ----
1     microbench       fp32_throughput         fmas_per_thread   1        cpp
1     microbench       memory_bandwidth        buffer_size_mb    0        cpp
2     rodinia          rodinia_hotspot         grid_size          1        cpp
2     shoc             shoc_gemm               matrix_size        1        cpp
3     vendor_library   cublas_sgemm            matrix_size        1        cpp
4     triton           triton_gemm             matrix_size        1        python
5     ml_training      resnet50_train          batch_size         3        python

Total: 107 benchmarks (12 tier1, 51 tier2, 11 tier3, 7 tier4, 26 tier5)
```

### `gpubench compile`

Compile C++ benchmarks (Tier 1–3). Python benchmarks (Tier 4–5) are skipped.

```
Usage: gpubench compile [--tier T] [--suite S] [--benchmark B] [--platform P]
                        [--jobs J] [--verbose]

Options:
  --tier T          Compile only this tier
  --suite S         Compile only this suite
  --benchmark B     Compile only this benchmark
  --platform P      Force platform: cuda, rocm, metal (default: auto-detect)
  --jobs J          Parallel compilation jobs (default: number of CPU cores)
  --verbose         Show full compiler output

Examples:
  gpubench compile                          # compile everything
  gpubench compile --tier 1                 # compile all tier 1
  gpubench compile --benchmark shoc_gemm    # compile one benchmark
  gpubench compile --platform cuda          # force CUDA platform
```

**Behavior:**

1. Discover benchmarks matching filters
2. For each C++ benchmark (language != "python"):
   - Run `make -C <dir> PLATFORM=<platform> clean` (if `--clean` flag)
   - Run `make -C <dir> PLATFORM=<platform>`
3. Report pass/fail summary
4. Exit 0 if all pass, exit 1 if any fail

### `gpubench run`

Run benchmarks with parameter sweeping and repetitions. Store results in SQLite.

```
Usage: gpubench run [--tier T] [--suite S] [--benchmark B]
                    [--reps N] [--db FILE] [--platform P]
                    [--sweep-non-scaling] [--timeout SECS]
                    [--scaling-values V1,V2,...] [--dry-run]
                    [--continue-on-error] [--verbose]

Options:
  --tier T              Run only this tier
  --suite S             Run only this suite
  --benchmark B         Run only this benchmark
  --reps N              Repetitions per configuration (default: 7)
  --db FILE             SQLite database file (default: results.db)
  --platform P          Force platform (default: auto-detect)
  --sweep-non-scaling   Sweep non-scaling params too (default: use defaults only)
  --timeout SECS        Per-benchmark timeout in seconds (default: 300)
  --scaling-values V    Override scaling param values (comma-separated)
  --dry-run             Show what would be run without executing
  --continue-on-error   Continue even if a benchmark fails
  --verbose             Show benchmark stderr output

Examples:
  gpubench run                                    # run everything, 7 reps, results.db
  gpubench run --tier 1 --reps 3                  # run tier 1 with 3 reps
  gpubench run --benchmark shoc_gemm --reps 10    # run one benchmark, 10 reps
  gpubench run --db my_results.db                 # custom database file
  gpubench run --sweep-non-scaling                # full parameter sweep
  gpubench run --dry-run                          # preview execution plan
```

**Execution Flow:**

```
gpubench run [options]
  │
  ├─ 1. Detect platform (or use --platform)
  ├─ 2. Collect platform info → platforms table
  ├─ 3. Create session → sessions table
  ├─ 4. Discover benchmarks (filter by --tier/--suite/--benchmark)
  │
  └─ For each benchmark:
       ├─ Read params.json
       ├─ Determine parameter configurations:
       │    ├─ Default mode: sweep scaling_param.values × {non_scaling defaults}
       │    └─ --sweep-non-scaling: sweep scaling × cartesian(non_scaling values)
       │
       └─ For each parameter configuration:
            └─ For each repetition (1..N):
                 ├─ Set BENCH_PARAM_* environment variables
                 ├─ Execute benchmark (with timeout)
                 ├─ Parse stdout JSON-lines (output protocol)
                 ├─ Record run → runs table
                 ├─ Record kernel results → kernel_results table
                 └─ Record metrics → metrics table
```

### `gpubench info`

Display detected platform information.

```
Usage: gpubench info [--json]

Examples:
  gpubench info
  gpubench info --json
```

**Output:**

```
Platform Information
  Platform:    cuda
  GPU:         NVIDIA A100-SXM4-80GB
  GPU VRAM:    80 GiB
  Driver:      535.183.01
  Runtime:     CUDA 12.2
  Compiler:    nvcc 12.2.140
  OS:          Linux 6.5.0-44-generic x86_64
  CPU:         AMD EPYC 7763 64-Core (128 cores)
  RAM:         512 GiB
  Hostname:    gpu-node-01
  Python:      3.11.5
```

### `gpubench validate`

Validate all `params.json` files without running anything.

```
Usage: gpubench validate [--tier T] [--suite S] [--benchmark B]

Examples:
  gpubench validate
  gpubench validate --tier 1
```

**Checks:**
- JSON is valid
- Required fields are present
- `benchmark` matches directory name
- `tier` matches parent directory
- `default` appears in `values`
- Parameter names are valid identifiers
- No duplicate parameter names
- Scaling param has ≥10 values

## Platform Auto-Detection

The CLI auto-detects the platform using the following logic:

```python
def detect_platform():
    if sys.platform == "darwin":
        return "metal"
    if os.path.isdir("/opt/rocm"):
        return "rocm"
    return "cuda"
```

### Platform Info Collection

The CLI gathers platform info from system commands:

| Info | CUDA | ROCm | Metal |
|------|------|------|-------|
| GPU model | `nvidia-smi --query-gpu=name --format=csv,noheader` | `rocm-smi --showproductname` | `system_profiler SPDisplaysDataType` |
| GPU VRAM | `nvidia-smi --query-gpu=memory.total --format=csv,noheader` | `rocm-smi --showmeminfo vram` | `system_profiler SPDisplaysDataType` |
| Driver | `nvidia-smi --query-gpu=driver_version --format=csv,noheader` | `cat /opt/rocm/.info/version` | `system_profiler SPDisplaysDataType` |
| Runtime | `nvcc --version` | `hipcc --version` | macOS version |
| Compiler | `nvcc --version` | `hipcc --version` | `clang++ --version` |
| OS | `uname -srm` | `uname -srm` | `uname -srm` |
| CPU | `/proc/cpuinfo` or `lscpu` | `/proc/cpuinfo` or `lscpu` | `sysctl -n machdep.cpu.brand_string` |
| RAM | `/proc/meminfo` | `/proc/meminfo` | `sysctl -n hw.memsize` |

If any detection fails, the field is stored as `null` — the tool should not crash due to missing platform info.

## Benchmark Discovery

Benchmarks are discovered by walking the repository directory tree:

```python
def discover_benchmarks(repo_root, tier=None, suite=None, benchmark=None):
    results = []
    for params_path in sorted(glob.glob(f"{repo_root}/tier*/*/params.json")):
        with open(params_path) as f:
            params = json.load(f)
        # Apply filters
        if tier and params["tier"] != tier:
            continue
        if suite and params["suite"] != suite:
            continue
        if benchmark and params["benchmark"] != benchmark:
            continue
        params["_dir"] = os.path.dirname(params_path)
        results.append(params)
    return results
```

## Benchmark Execution

### C++ Benchmarks (Tier 1–3)

```bash
# Compile (if not already compiled)
make -C tier2/rodinia_hotspot PLATFORM=cuda

# Run with parameters
cd tier2/rodinia_hotspot
BENCH_PARAM_grid_size=1024 BENCH_PARAM_block_size=16 BENCH_PARAM_num_iterations=10 \
    ./rodinia_hotspot
```

The binary name matches the benchmark name. The binary is expected at `<dir>/<benchmark_name>` after compilation.

### Python Benchmarks (Tier 4–5)

```bash
# No compile step needed

# Run with parameters
cd tier4/triton_gemm
BENCH_PARAM_matrix_size=1024 BENCH_PARAM_block_size=64 \
    python3 bench.py
```

For Metal platform, use `bench_mps.py` if it exists, otherwise `bench.py`.

### Execution Wrapper

For each run, the CLI:

1. Sets `BENCH_PARAM_*` environment variables
2. Starts the process with `subprocess.Popen(stdout=PIPE, stderr=PIPE)`
3. Reads stdout line-by-line, parsing JSON events
4. Enforces `--timeout` with `SIGTERM` then `SIGKILL`
5. Captures exit code
6. Writes results to database within a transaction

## Parameter Sweeping

### Default Mode (scaling only)

By default, the CLI sweeps only the scaling parameter. Non-scaling parameters use their `default` values.

For a benchmark with:
- Scaling param: 20 values
- 7 repetitions

Total runs: **20 × 7 = 140**

### Full Sweep Mode (`--sweep-non-scaling`)

When `--sweep-non-scaling` is passed, the CLI generates the Cartesian product of all non-scaling parameter values, crossed with all scaling parameter values.

For a benchmark with:
- Scaling param: 20 values
- Non-scaling param A: 4 values
- Non-scaling param B: 3 values
- 7 repetitions

Total runs: **20 × 4 × 3 × 7 = 1,680**

**Warning:** Full sweeps can be very large. The CLI should print the total run count before starting and prompt for confirmation if it exceeds 10,000 (unless `--yes` is passed).

### Partial Override

Users can override specific scaling values:

```bash
gpubench run --benchmark shoc_gemm --scaling-values 256,512,1024,2048
```

This overrides the values from params.json for the scaling parameter only.

## Database Management

### Database Creation

On first run (or if the file doesn't exist), the CLI creates the database using the schema from `docs/schema.sql`.

### Transaction Strategy

- Each **run** (one benchmark × one param config × one repetition) is wrapped in a transaction
- If a run fails, only that run's data is lost; previous runs are committed
- The session record is updated with `finished_at` and `status` at the end

### Database Location

Default: `results.db` in the current working directory. Override with `--db`.

## Output During Execution

The CLI provides real-time progress output to stderr:

```
=== GPU Benchmark Suite ===
Platform: CUDA (NVIDIA A100-SXM4-80GB)
Database: results.db
Repetitions: 7
Benchmarks: 107

[  1/107] tier1/fp32_throughput
  Scaling: fmas_per_thread (20 values × 7 reps = 140 runs)
    fmas_per_thread=4       rep 1/7  0.012ms  ✓
    fmas_per_thread=4       rep 2/7  0.011ms  ✓
    ...
    fmas_per_thread=2097152 rep 7/7  45.2ms   ✓
  ✓ 140/140 passed (0 failed)

[  2/107] tier1/memory_bandwidth
  Scaling: buffer_size_mb (22 values × 7 reps = 154 runs)
    buffer_size_mb=1        rep 1/7  0.05ms   ✓
    ...

=== Summary ===
  Total:   14,980 runs
  Passed:  14,923
  Failed:  57 (OOM on large sizes)
  Time:    3h 42m
  Database: results.db (142 MB)
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All runs completed successfully |
| 1 | Some runs failed (results still saved) |
| 2 | CLI usage error (bad arguments) |
| 3 | No benchmarks found matching filters |
| 4 | Database error |
| 130 | Interrupted (Ctrl+C) — partial results saved |

## Configuration File (Optional Future Extension)

A `gpubench.toml` or similar file could override defaults:

```toml
[defaults]
reps = 7
timeout = 300
db = "results.db"
platform = "auto"

[sweep]
non_scaling = false
```

This is **not required for the initial implementation** but the CLI should be structured to support it later.

## File Layout

```
scripts/
  gpubench              # main CLI entry point (#!/usr/bin/env python3)
docs/
  cli_design.md         # this document
  output_protocol.md    # stdout JSON-lines protocol
  schema.sql            # SQLite schema
  params_format.md      # params.json specification
```

## Implementation Notes

1. **Single file preferred.** The CLI should be a single Python file (`scripts/gpubench`) for simplicity. If it grows beyond ~1000 lines, split into a `scripts/gpubench_lib/` package.

2. **No pip dependencies.** Only Python stdlib + sqlite3. This ensures the tool works on any system with Python 3.7+.

3. **Idempotent platform detection.** Cache platform info for the session — don't re-detect for every benchmark.

4. **Graceful interruption.** Handle SIGINT (Ctrl+C): finish current run, write partial results, update session status to "interrupted", exit 130.

5. **Dry-run support.** `--dry-run` should print the full execution plan (benchmarks, parameter configs, total run counts) without executing anything.

6. **Logging.** Use Python `logging` module. Default level: INFO. `--verbose` sets DEBUG. `--quiet` sets WARNING.
