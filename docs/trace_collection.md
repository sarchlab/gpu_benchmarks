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

### What happens internally

For each benchmark and parameter configuration, `gpubench`:

1. Creates the output directory `<trace-output>/<benchmark>/<config>/`
2. Sets the following environment variables for the subprocess:
   - `LD_PRELOAD=<collect-trace>` — injects the NVBit tracer
   - `USER_DEFINED_FOLDERS=1` — tells the tracer to use a custom output path
   - `TRACES_FOLDER=<trace-output>/<benchmark>/<config>/` — where traces are written
3. Runs the benchmark normally (with LD_PRELOAD active, NVBit intercepts kernel launches)
4. If `--post-process` is set, runs `<post-process> <trace-output>/<benchmark>/<config>/traces/kernelslist`

> **Note:** When `--collect-trace` is set, **no timing data is written to SQLite**.
> The `--db` flag is ignored. This is by design — NVBit instrumentation adds
> overhead that invalidates timing measurements.

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
- NVBit only works on NVIDIA GPUs with CUDA. ROCm and Metal platforms are not supported.
