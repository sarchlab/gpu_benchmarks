# NVBit Instruction Trace Collection

This guide explains how to collect NVBit instruction traces from GPU benchmarks
using `gpubench run --collect-trace`, and how to use those traces with the
BBSim GPU simulator.

## Overview

`gpubench` supports trace-driven simulation via NVBit. When `--collect-trace`
is set, `gpubench` injects a tracer shared library into each benchmark process
using `LD_PRELOAD`. The tracer records per-kernel instruction traces in
seekable zstd-compressed format (`.trace.zst`), which can then be fed into the
BBSim (bb_triosim) simulator.

**gpubench does NOT build the tracer.** You must build it separately from the
[bb_triosim](https://github.com/sarchlab/bb_triosim) repository.

## Step 1: Build the NVBit Tracer

Clone and build the tracer tool from the bb_triosim repository:

```bash
git clone https://github.com/sarchlab/bb_triosim.git
cd bb_triosim/tracer_tool
make
```

This produces `tracer_tool.so`. Note the full path to this file — you will need
it in Step 2.

Requirements:
- CUDA toolkit (nvcc) must be installed
- libzstd development headers (`apt install libzstd-dev` or equivalent)
- NVBit (included in bb_triosim as a submodule) supports CUDA 10.x–12.x on x86_64/aarch64

## Step 2: Collect Traces

Run `gpubench run` with the `--collect-trace` flag pointing to `tracer_tool.so`:

```bash
gpubench run \
  --benchmark gemm \
  --collect-trace /path/to/bb_triosim/tracer_tool/tracer_tool.so \
  --trace-output ./my_traces \
  --reps 1
```

### Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--collect-trace <path>` | Path to `tracer_tool.so` (required to enable trace mode) | — |
| `--trace-output <dir>` | Directory where traces are written | `./traces` |
| `--post-process <binary>` | Optional post-processing binary run after each benchmark | — |
| `--resume-trace` | Skip configs that already have complete or timed-out trace outputs | `false` |
| `--trace-clean-incomplete` | When resuming, remove incomplete config trace directories before retrying | `false` |
| `--tracer <v1\|v2.0>` | Trace pipeline: `v1` (`.trace.zst`, described below) or `v2.0` (Accel-Sim 2.0 `.tracez`, see [Accel-Sim 2.0 traces](#accel-sim-20-traces---tracer-v20)) | `v1` |
| `--param <name=value>` | Pin a non-scaling parameter to one declared value (repeatable, needs `--benchmark`); also works without tracing | — |

### What happens internally

For each benchmark and parameter configuration, `gpubench`:

1. Creates the output directory `<trace-output>/<benchmark>/<config>/`
2. Sets the following environment variables for the subprocess:
   - `LD_PRELOAD=<collect-trace>` — injects the NVBit tracer
   - `USER_DEFINED_FOLDERS=1` — tells the tracer to use a custom output path
   - `TRACES_FOLDER=<trace-output>/<benchmark>/<config>/` — where traces are written
3. Runs the benchmark normally (with LD_PRELOAD active, NVBit intercepts kernel launches)
4. If `--post-process` is set, runs `<post-process> <trace-output>/<benchmark>/<config>/traces/kernelslist`

> **Note:** When `--collect-trace` is set, **no timing data is written to SQLite**:
> NVBit instrumentation adds overhead that invalidates performance timings.
> An existing `--db` file (default `results.db`) is read-only eligibility guidance.

### Selecting previously successful configurations

If the selected database exists, a trace configuration must match a run with
`status='ok'` on its tier, suite, benchmark, scaling parameter name/value, and
complete non-scaling parameter object. JSON key order, whitespace, and equivalent
numeric spellings do not change identity; scalar types and distinct numeric values
remain distinct. A successful block size or algorithm variant does not admit a
failed or untested variant at the same scaling point.

The scope is all sessions in the selected database, preserving the existing
any-success policy. A later failed repetition does not erase an earlier success
of the same full configuration. The CLI reports the matching source session and
run IDs (the earliest matching successful run) with each selected configuration.
Choose the input database for the dataset you intend to trace; this match alone
does not establish binary, device, or input provenance equivalence.

The requested `vector_add_warmup` setting participates in eligibility, as recorded
in the timing run. After selection, trace execution still forces the warmer off
to exclude it from capture. For example, default `true` selection can use matching
`true` timing runs even though the trace process receives `false`.

An existing but unreadable database, invalid schema, or malformed successful row
is an error; the CLI does not fall back to unfiltered tracing. An empty valid
database admits no configurations. If the database file is absent, tracing remains
unfiltered and the CLI reports that no database guidance is available. Trace
artifact validation is separate from this eligibility check.

### Resuming Interrupted Trace Collection

For long trace jobs, use `--resume-trace` so a restarted command skips configs
that already produced complete trace outputs or were previously marked as timed
out:

```bash
gpubench run \
  --collect-trace /path/to/bb_triosim/tracer_tool/tracer_tool.so \
  --trace-output ./my_traces \
  --reps 1 \
  --continue-on-error \
  --resume-trace
```

A config is considered complete only when its `traces/` directory contains a
`kernelslist*` file, a matching `stats*` file, and all `.trace.zst` files
referenced by the kernel list. This avoids treating timeout artifacts as
complete; for example, a config with only `kernelslist_ctx_*` and `stats_ctx_*`
but no `.trace.zst` files will be retried.

Add `--trace-clean-incomplete` when you want retries to start from a clean
config directory:

```bash
gpubench run \
  --collect-trace /path/to/bb_triosim/tracer_tool/tracer_tool.so \
  --trace-output ./my_traces \
  --reps 1 \
  --continue-on-error \
  --resume-trace \
  --trace-clean-incomplete
```

If a trace run times out, `gpubench` deletes that config's trace data and writes
`<trace-output>/<benchmark>/<config>/traces/trace_timeout.json`. This keeps
large partial traces from consuming disk space and lets future
`--resume-trace` runs skip the same timeout-prone config instead of retrying it.

### Accel-Sim 2.0 traces (`--tracer v2.0`)

`--tracer v2.0` drives the upstream
[Accel-Sim](https://github.com/accel-sim/accel-sim-framework) NVBit tracer and
its post-processor instead of the `v1` tracer:

```bash
gpubench run --benchmark cuda_nbody --scaling-values 256 --param timesteps=10 \
  --collect-trace <accel-sim>/util/tracer_nvbit/tracer_tool/tracer_tool.so \
  --tracer v2.0 \
  --post-process <accel-sim>/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing \
  --trace-output ./my_traces --reps 1
```

- The tracer writes `kernel-*.trace.xz` (memory addresses already stored as
  base+stride or base+delta, then streamed through `xz -1`); gpubench pins
  `TOOL_COMPRESS=1` and `TRACE_FILE_COMPRESS=1`.
- `--post-process` is required. It receives the config's `traces/` directory
  (every context's kernel list) and turns each `.trace.xz` into a `.tracez`:
  instructions grouped per thread block and warp, in 2048-instruction chunks,
  each zstd-compressed with a dictionary trained on the kernel, plus an index
  so the simulator can read it page by page. It writes `kernelslist.g`.
- Post-processing gets the trace timeout instead of `v1`'s 300 s, and a failed
  or incomplete post-process counts as a failed config.
- A config is complete when a `kernelslist*.g` lists at least one `.tracez`
  and every listed `.tracez` exists; this is also what `--resume-trace` checks.
  The `.trace.xz` files are intermediates and can be deleted afterwards.
- `--trace-sample` (live block sampling) is `v1`-only.

`scripts/trace_collect.py` uses this mode to trace every successful
configuration of a timing DB and records the size of each stage (raw text,
`.trace.xz`, `.tracez`); see its `--help`.

## Step 3: Output Structure

After collection, traces are organized by benchmark and parameter configuration:

```
my_traces/
  gemm/
    M_512/                          # scaling param only
      traces/
        kernelslist
        kernel-1-ctx_0x5555xxxx.trace.zst
        kernel-2-ctx_0x5555xxxx.trace.zst
        ...
    M_1024/
      traces/
        ...
  matmul/
    N_256__TILE_16__ALGO_naive/     # scaling + non-scaling params
      traces/
        ...
    N_256__TILE_32__ALGO_naive/
      traces/
        ...
```

The directory name encodes the full parameter configuration:
`<scaling_param>_<value>` when there are no non-scaling parameters, or
`<scaling_param>_<value>__<ns_param1>_<value1>__<ns_param2>_<value2>` when
non-scaling parameters are present (sorted alphabetically by parameter name).

Each `.trace.zst` file is a seekable zstd-compressed trace containing
per-block instruction data with a shared dictionary for efficient compression.

The `kernelslist` file inside each configuration directory is the entry point
for BBSim simulation.

## Step 4: Simulate with BBSim

Point BBSim at the `kernelslist` file:

```bash
cd bb_triosim
bbsim simulate --trace my_traces/gemm/M_512/traces/kernelslist
```

Refer to the [bb_triosim documentation](https://github.com/sarchlab/bb_triosim)
for full simulation options and configuration.

## Tips

- Use `--reps 1` during trace collection to avoid redundant traces.
- Use `--benchmark <name>` to target a single benchmark rather than the full suite.
- Benchmark warmup is not a benchmark parameter; trace collection runs the same
  invocation that normal `gpubench run` uses for the selected parameter config.
- Traces are compressed with zstd and are typically much smaller than raw traces.
- NVBit only works on NVIDIA GPUs with CUDA. The ROCm platform is not supported.
