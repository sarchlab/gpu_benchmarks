#!/usr/bin/env python3
"""Collect a dynamic-count fingerprint of a real run with ncu, in the
CountsReport JSON shape that `ptrace countcheck` consumes.

This is the measured side of the generation gate: traces produced by
`ptrace gen` (live-sampled collection) or `ptrace synth --grid` / `scale`
(extrapolation) are unverified at the target size; comparing the counts they
predict (`ptrace predict <dir> pred.json`) against this profile catches a
different binary, a different code path, or wrong synthesis immediately.
Counts-only profiling is orders of magnitude cheaper than address tracing.

Usage:
    scripts/collect_counts.py -o meas.json -- ./polybench_gemm --size 256
    ptrace predict gen-traces/ pred.json
    ptrace countcheck pred.json meas.json [tolerance]

Metric mapping (KernelCounts fields left at 0 are skipped by countcheck):
    warp_insts   <- smsp__inst_executed.sum
    thread_insts <- smsp__thread_inst_executed.sum
    blocks       <- launch__grid_size (product of grid dims)

Caveats vs trace-predicted counts:
  - The tracer skips CALL.REL.NOINC (NVBit 1.7.4 bug workaround) and merges
    LDGSTS pairs; ncu counts every issued instruction. Expect predicted
    warp_insts to sit slightly BELOW measured. Use the tolerance accordingly
    (default 5%; tighten once a benchmark's offset is known).
  - Kernel order follows launch order in both reports. With PTRACE_DEDUP the
    kernelslist repeats deduped launches, but `ptrace predict` runs on the
    trace *files*, so profile with the same dedup view in mind: per-kernel
    rows pair correctly only for dedup=off or single-launch benchmarks;
    otherwise compare totals manually or pass --total-only.
"""

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import tempfile

METRICS = [
    "smsp__inst_executed.sum",
    "smsp__thread_inst_executed.sum",
    "launch__grid_size",
]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TOOL = os.path.join(REPO_ROOT, "tracer", "nvbit_tool", "nvbit_tool.so")


def run_nvbit_counts(cmd, tool):
    """Measured side via the gpubench tracer's counting-only mode
    (PTRACE_SAMPLE=count): every block early-exits but bumps the warp-inst
    counter; stats.csv carries per-kernel totals. Used when GPU performance
    counters are admin-restricted (ERR_NVGPUCTRPERM). Slower than ncu but
    needs no privileges, and its skip rules (CALL.REL.NOINC, LDGSTS pairing)
    match the trace-predicted side exactly."""
    with tempfile.TemporaryDirectory(prefix="counts-") as td:
        env = dict(os.environ)
        env.update(
            LD_PRELOAD=tool,
            PTRACE_SAMPLE="count",
            TRACES_FOLDER=td,
            USER_DEFINED_FOLDERS="1",
        )
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            sys.exit(f"command failed under counting tracer: {proc.returncode}")
        kernels = []
        tdir = os.path.join(td, "traces")
        for name in sorted(os.listdir(tdir)):
            if not name.startswith("stats"):
                continue
            with open(os.path.join(tdir, name)) as f:
                for row in csv.DictReader(f, skipinitialspace=True):
                    blocks = int(row["#blocks"])
                    warps_per_block = (int(row["#threads"]) + 31) // 32
                    kernels.append(
                        {
                            "trace_file": row["kernel id"].strip(),
                            "blocks": blocks,
                            "warps": blocks * warps_per_block,
                            "warp_insts": int(row["total_insts"]),
                        }
                    )
        return kernels


def run_ncu_counts(cmd):
    """Measured side via ncu. Returns kernels list, or None on
    ERR_NVGPUCTRPERM (counters restricted to admin)."""
    ncu_cmd = [
        "ncu",
        "--metrics", ",".join(METRICS),
        "--csv",
        "--target-processes", "all",
        "--print-units", "base",
    ] + cmd
    proc = subprocess.run(ncu_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        if "ERR_NVGPUCTRPERM" in proc.stdout + proc.stderr:
            return None
        sys.stderr.write(proc.stderr)
        sys.exit(f"ncu failed with exit code {proc.returncode}")

    # ncu --csv prints non-CSV preamble lines; the table starts at the header
    # row beginning with "ID" (quoted).
    lines = proc.stdout.splitlines()
    start = next(
        (i for i, l in enumerate(lines) if l.startswith('"ID"')), None
    )
    if start is None:
        sys.stderr.write(proc.stdout[-2000:])
        sys.exit("no CSV table in ncu output")

    # rows: one per (kernel launch, metric)
    per_launch = {}  # launch id -> {"name":..., metric: value}
    order = []
    for row in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        lid = row["ID"]
        if lid not in per_launch:
            per_launch[lid] = {"name": row["Kernel Name"]}
            order.append(lid)
        val = row["Metric Value"].replace(",", "")
        per_launch[lid][row["Metric Name"]] = int(float(val))

    kernels = []
    for lid in order:
        m = per_launch[lid]
        kernels.append(
            {
                "trace_file": m["name"],
                "blocks": m.get("launch__grid_size", 0),
                "warp_insts": m.get("smsp__inst_executed.sum", 0),
                "thread_insts": m.get("smsp__thread_inst_executed.sum", 0),
            }
        )
    return kernels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=True, help="output JSON path")
    ap.add_argument(
        "--mode",
        choices=["auto", "ncu", "nvbit"],
        default="auto",
        help="auto: try ncu, fall back to the nvbit counting tracer on ERR_NVGPUCTRPERM",
    )
    ap.add_argument(
        "--tool",
        default=DEFAULT_TOOL,
        help="nvbit_tool.so path for nvbit mode",
    )
    ap.add_argument(
        "--total-only",
        action="store_true",
        help="emit only the run total (use when launch order differs from trace files, e.g. PTRACE_DEDUP)",
    )
    ap.add_argument("command", nargs=argparse.REMAINDER, help="-- <command...>")
    args = ap.parse_args()
    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("no command given (use: collect_counts.py -o out.json -- <cmd>)")

    kernels = None
    if args.mode in ("auto", "ncu"):
        kernels = run_ncu_counts(cmd)
        if kernels is None:
            if args.mode == "ncu":
                sys.exit(
                    "ncu: GPU performance counters restricted (ERR_NVGPUCTRPERM); "
                    "rerun with --mode nvbit or enable counters "
                    "(NVreg_RestrictProfilingToAdminUsers=0)"
                )
            print(
                "ncu counters restricted (ERR_NVGPUCTRPERM) — "
                "falling back to nvbit counting tracer",
                file=sys.stderr,
            )
    if kernels is None:
        kernels = run_nvbit_counts(cmd, args.tool)

    total = {}
    for k in kernels:
        for f, v in k.items():
            if f != "trace_file" and v:
                total[f] = total.get(f, 0) + v

    report = {"total": total}
    if not args.total_only:
        report["kernels"] = kernels
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(
        f"{len(kernels)} kernel launch(es) profiled -> {args.output} "
        f"(total warp_insts={total.get('warp_insts', 0)})"
    )


if __name__ == "__main__":
    main()
