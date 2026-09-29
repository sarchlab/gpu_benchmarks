# CLI Design — `gpubench`

## Overview

`gpubench` is a Go CLI tool that compiles, runs, and collects results from the GPU benchmark suite. It discovers benchmarks by walking `tier*/*/params.json`, drives parameterized execution, and stores all results in an SQLite database.

**Location:** `cmd/gpubench/` (Go package, build with `go build -o gpubench ./cmd/gpubench`)

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
  --platform P      Force platform: cuda, rocm (default: auto-detect)
  --jobs J          Parallel make jobs per benchmark (default: serial)
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
   - Tier 1–2: run `make -B -C <dir> PLATFORM=<platform>` to rebuild the selected
     binaries even when source timestamps are unchanged. This applies the requested
     platform and inherited build settings such as `CUDA_ARCH` on every explicit
     compile invocation; it can take longer than an incremental build.
   - Other C++ tiers: run `make -C <dir> PLATFORM=<platform>` (incremental).
3. Report pass/fail summary
4. Exit 0 if all pass, exit 1 if any fail

### `gpubench run`

Run benchmarks with parameter sweeping and fresh-process repetitions. Store results in SQLite. Tier 1/Tier 2 emit accumulated raw GPU times; downstream analysis takes the minimum successful total time for each fixed configuration and derives throughput or latency from its work descriptors. `info` protocol events are accepted but not stored or exported; CLI-detected platform metadata is separate.

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
  --sweep-non-scaling   Sweep non-scaling params too (default: true;
                        use --sweep-non-scaling=false for defaults only)
  --timeout SECS        Per-benchmark timeout in seconds (default: 60)
  --scaling-values V    Override scaling param values (comma-separated)
  --dry-run             Show what would be run without executing
  --continue-on-error   Continue even if a benchmark fails
  --verbose             Show benchmark stderr output

Examples:
  gpubench run                                    # run everything, 7 reps, results.db
  gpubench run --tier 1 --reps 3                  # run tier 1 with 3 reps
  gpubench run --benchmark shoc_gemm --reps 10    # run one benchmark, 10 reps
  gpubench run --db my_results.db                 # custom database file
  gpubench run --sweep-non-scaling=false          # non-scaling defaults only
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
       │    ├─ Default mode: sweep scaling × cartesian(workload non_scaling values)
       │    └─ --sweep-non-scaling=false: sweep scaling_param.values × {non_scaling defaults}
       │
       └─ For each parameter configuration:
            └─ For each repetition (1..N):
                 ├─ Set BENCH_PARAM_* environment variables
                 ├─ Execute one complete algorithm in a fresh process (with timeout)
                 ├─ Parse stdout JSON-lines (output protocol)
                 ├─ Validate exit status, output structure, parameters, and timing sum
                 ├─ Record run and status → runs table
                 ├─ Record kernel results → kernel_results table
                 └─ Record optional legacy/higher-tier metrics → metrics table
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
  Compiler:    nvcc 12.2.140
  OS:          Linux 6.5.0-44-generic x86_64
  CPU:         AMD EPYC 7763 64-Core (128 cores)
  RAM:         512 GiB
  Hostname:    gpu-node-01
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

```go
func detectPlatform() string {
    if _, err := os.Stat("/opt/rocm/bin/hipcc"); err == nil {
        return "rocm"
    }
    return "cuda"
}
```

### Platform Info Collection

The CLI gathers platform info from system commands:

| Info | CUDA | ROCm |
|------|------|------|
| GPU model | `nvidia-smi --query-gpu=name --format=csv,noheader` | `rocm-smi --showproductname` |
| GPU VRAM | `nvidia-smi --query-gpu=memory.total --format=csv,noheader` | Unknown (`null`) |
| Driver | `nvidia-smi --query-gpu=driver_version --format=csv,noheader` | Unknown (`null`) |
| Benchmark runtime | Unknown (`null`) | Unknown (`null`) |
| Installed compiler | `nvcc --version` | `hipcc --version` |
| OS | `uname -srm` | `uname -srm` |
| CPU | OS-specific host queries | OS-specific host queries |
| RAM | OS-specific host queries | OS-specific host queries |

Optional unavailable version fields are stored as `null` in SQLite and JSON;
the text `info` display omits them. Compiler output
describes the installed tool; it does not identify the tool or flags that built
the benchmark, or the runtime loaded by that executable. The ROCm release file
does not identify the driver. Actual per-executable build/runtime provenance and
visible-device mapping require separate qualification.

Platform row reuse includes the observed runtime and compiler fields. Opening an
existing database replaces the old fingerprint index while preserving historical
rows and session links. New sessions with corrected unknown version fields get a
different row from legacy sessions with compiler/ROCm-release strings in those
fields. Old values remain historical, unqualified observations; the migration
does not reinterpret them as verified runtime or driver versions.
Continue writing with the updated CLI: older writers recreate the narrower
unique index and can reject a database containing distinct version observations.
Existing read-only exports remain compatible because table columns are unchanged.

## Benchmark Discovery

Benchmarks are discovered by walking the repository directory tree:

```go
func discoverBenchmarks(repoRoot string, tier int, suite, benchmark string) []Benchmark {
    // Walks tier*/*/params.json, parses each, applies filters,
    // returns sorted list of discovered benchmarks.
}
```

## Benchmark Execution

### C++ Benchmarks (Tier 1–3)

```bash
# Compile (if not already compiled)
make -C tier2/rodinia_hotspot PLATFORM=cuda

# Run with parameters
cd tier2/rodinia_hotspot
BENCH_PARAM_grid_size=1024 BENCH_PARAM_total_iterations=360 BENCH_PARAM_pyramid_height=2 \
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


### Execution Wrapper

For each run, the CLI:

1. Sets `BENCH_PARAM_*` environment variables
2. Starts the process with `exec.Command` (stdout and stderr piped)
3. Reads stdout line-by-line, parsing JSON events
4. Enforces `--timeout` via `context.WithTimeout`; on expiry it SIGKILLs the
   benchmark's entire process group (so subprocesses die too) and bounds the
   post-kill wait with `WaitDelay`
5. Captures exit code and validates the output before marking a run successful
6. Writes results to database within a transaction; a transaction that fails to
   commit marks the run **failed** (data loss is not reported as a pass)

## Parameter Sweeping

### Default Mode (full parameter sweep)

By default, the CLI generates the Cartesian product of workload non-scaling parameter values, crossed with all scaling parameter values. Harness measurement controls such as `vector_add_warmup` are fixed to their default or CLI override, recorded with each configuration, and never swept.

For a benchmark with:
- Scaling param: 20 values
- Non-scaling param A: 4 values
- Non-scaling param B: 3 values
- 7 repetitions

Total runs: **20 × 4 × 3 × 7 = 1680**

### Defaults-Only Mode (`--sweep-non-scaling=false`)

When `--sweep-non-scaling=false` is passed, the CLI sweeps only the scaling parameter. Non-scaling parameters use their `default` values.

For a benchmark with:
- Scaling param: 20 values
- 7 repetitions

Total runs: **20 × 7 = 140**

**Warning:** Full sweeps can be very large. Use `--dry-run` to preview the execution plan and total run count before starting.

### Partial Override

Users can override specific scaling values:

```bash
gpubench run --benchmark shoc_gemm --scaling-values 256,512,1024,2048
```

This overrides the values from params.json for the scaling parameter only.

## Database Management

### Database Creation

On first run (or if the file does not exist), the CLI creates the database using `SCHEMA_SQL` in `cmd/gpubench/schema.go`, corresponding to the schema documented in `docs/schema.sql`.

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
timeout = 60
db = "results.db"
platform = "auto"

[sweep]
non_scaling = false
```

This is **not required for the initial implementation** but the CLI should be structured to support it later.

## File Layout

```
cmd/
  gpubench/
    main.go             # CLI entry point and argument parsing
    commands.go         # Command implementations (list, compile, run, info, validate)
    database.go         # SQLite database helpers
    discover.go         # Benchmark discovery from params.json
    platform.go         # Platform detection and info collection
    protocol.go         # JSON-lines output protocol parsing
    schema.go           # Embedded SQLite schema
docs/
  cli_design.md         # this document
  output_protocol.md    # stdout JSON-lines protocol
  schema.sql            # SQLite schema
  params_format.md      # params.json specification
```

## Implementation Notes

1. **Go package layout.** The CLI lives in `cmd/gpubench/` as a Go package. Build with `go build -o gpubench ./cmd/gpubench`.

2. **Uses go-sqlite3.** SQLite access via the `github.com/mattn/go-sqlite3` driver (requires CGO). The schema is embedded in the binary.

3. **Idempotent platform detection.** Cache platform info for the session — don't re-detect for every benchmark.

4. **Graceful interruption.** Handle SIGINT (Ctrl+C) via `os/signal`: finish current run, write partial results, update session status to "interrupted", exit 130.

5. **Dry-run support.** `--dry-run` prints the full execution plan (benchmarks, parameter configs, total run counts) without executing anything.

6. **Build with `go build`.** The binary is self-contained; no runtime dependencies beyond the OS. Add `gpubench` to `.gitignore`.
