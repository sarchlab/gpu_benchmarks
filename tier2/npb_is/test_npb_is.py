#!/usr/bin/env python3
"""Behavioral regressions for a built npb_is; --trace also needs NVIDIA nsys."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "params.json").read_text())
KERNELS = MANIFEST["kernels"]


def environment(size, block=256, verify=True, warm=True):
    env = {k: v for k, v in os.environ.items() if not k.startswith("BENCH_PARAM_")}
    env.update(BENCH_PARAM_size=str(size), BENCH_PARAM_block_size=str(block),
               BENCH_PARAM_verify=str(verify).lower(),
               BENCH_PARAM_vector_add_warmup=str(warm).lower())
    return env


def run(command, env=None):
    return subprocess.run(command, env=env, text=True, capture_output=True, timeout=180)


def check_output(result, size, block, verify):
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in result.stdout.splitlines()]
    kernels = [r for r in records if r.get("type") == "kernel"]
    summaries = [r for r in records if r.get("type") == "summary"]
    assert len(records) == 7 and len(summaries) == 1, records
    assert Counter(r["name"] for r in kernels) == Counter(KERNELS), records
    for record in kernels:
        assert record["params"] == {"size": size, "block_size": block}, record
        assert math.isfinite(record["time_ms"]) and record["time_ms"] >= 0, record
    total = summaries[0]["total_time_ms"]
    assert math.isfinite(total) and total >= 0, summaries
    assert math.isclose(total, sum(r["time_ms"] for r in kernels), abs_tol=4e-6), records
    if verify:
        assert re.search(r"Verification: (\d+)/\1 partial \+ full_verify ok "
                         r"\+ rank array ok => PASS", result.stderr), result.stderr
    else:
        assert "Verification:" not in result.stderr, result.stderr


def check_invalid(binary):
    # These used to reach rank() with MAX_KEY <= iteration and corrupt memory.
    for size in (-1, 0, 63, 64, 128, 160, 175):
        commands = [(str(binary),), (str(binary), "--size", str(size))]
        for command in commands:
            env = environment(size)
            if len(command) > 1:
                del env["BENCH_PARAM_size"]
            result = run(command, env)
            assert result.returncode > 0, (command, result.returncode, result.stderr)
            records = [json.loads(line) for line in result.stdout.splitlines()]
            assert records == [{"type": "error", "message": "size must be >= 176"}], records
    print("PASS: invalid sizes rejected before GPU setup; no timing records", flush=True)


def check_trace(binary, size, block, verify):
    assert shutil.which("nsys"), "--trace requires NVIDIA Nsight Systems (nsys)"
    with tempfile.TemporaryDirectory(prefix="npb-is-trace-") as directory:
        output = Path(directory) / "rank"
        env = environment(size, block, verify, warm=False)
        result = run(["nsys", "profile", "--trace=cuda", "--sample=none",
                      "--cpuctxsw=none", "--force-overwrite=true", "-o", str(output),
                      str(binary)], env)
        assert result.returncode == 0, result.stdout + result.stderr
        # nsys also writes progress messages. Check the benchmark's JSON lines.
        result.stdout = "\n".join(line for line in result.stdout.splitlines()
                                  if line.startswith('{"type":'))
        check_output(result, size, block, verify)
        database = output.with_suffix(".sqlite")
        exported = run(["nsys", "export", "--type=sqlite", "--force-overwrite=true",
                        "-o", str(database), str(output.with_suffix(".nsys-rep"))])
        assert exported.returncode == 0, exported.stdout + exported.stderr
        with sqlite3.connect(database) as conn:
            names = [row[0] for row in conn.execute(
                "SELECT s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                "JOIN StringIds s ON s.id = k.demangledName ORDER BY k.start")]
        expected = [name for _ in range(10) for name in KERNELS]
        actual = [re.sub(r"^void\s+", "", name.split("(", 1)[0]) for name in names]
        assert actual == expected, actual
    print(f"PASS: CUDA trace size={size} block={block} verify={verify}: "
          "exactly 60 launches, six rank kernels x 10, no extra launches", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=HERE / "npb_is")
    parser.add_argument("--invalid-only", action="store_true", help="no GPU required")
    parser.add_argument("--sweep", action="store_true", help="verify all manifest sizes")
    parser.add_argument("--trace", action="store_true", help="check actual CUDA launch sequence")
    args = parser.parse_args()
    binary = args.binary.resolve()
    assert binary.is_file(), f"Build the benchmark first: {binary}"
    check_invalid(binary)
    if args.invalid_only:
        return
    cases = [(176, 32, True, True), (176, 256, True, True),
             (1024, 256, True, True), (1536, 128, True, True),
             (12288, 256, True, True), (65536, 256, True, True),
             (1048576, 256, True, True), (8388608, 256, True, True),
             (8388608, 128, True, True), (8388608, 256, False, True),
             (65536, 256, True, False)]
    if args.sweep:
        cases.extend((size, 256, True, True) for size in MANIFEST["scaling_param"]["values"])
    for size, block, verify, warm in dict.fromkeys(cases):
        result = run([str(binary)], environment(size, block, verify, warm))
        check_output(result, size, block, verify)
        print(f"PASS: size={size} block={block} verify={verify} warm={warm}", flush=True)
    if args.trace:
        for size, block, verify in [(176, 256, True), (1024, 256, True),
                                    (1024, 256, False), (1048576, 32, True)]:
            check_trace(binary, size, block, verify)


if __name__ == "__main__":
    main()
