#!/usr/bin/env python3
"""Check exact raw triangle totals using post-timing device-output faults."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

FAULT = r'''
    if (const char* delta_string=std::getenv("TC_TEST_DELTA")) {
        const int location=std::atoi(std::getenv("TC_TEST_LOCATION"));
        const int at=location==0?0:location==1?num_edges/2:num_edges-1;
        long long value;
        HIP_CHECK(hipMemcpy(&value,d_edge_counts+at,sizeof(value),hipMemcpyDeviceToHost));
        value+=std::atoll(delta_string);
        HIP_CHECK(hipMemcpy(d_edge_counts+at,&value,sizeof(value),hipMemcpyHostToDevice));
        std::fprintf(stderr,"TC_TEST_INJECT %s %d\n",delta_string,at);
    }
'''


def instrument(source):
    output = "    // Sum up triangle counts from GPU\n"
    reference = "        long long cpu_triangles = cpu_raw / 3;"
    for anchor in (output, reference):
        if source.count(anchor) != 1:
            raise ValueError("expected one production anchor: " + anchor)
    return source.replace(output, FAULT + output).replace(
        reference, reference + '\n        std::fprintf(stderr,"TC_TEST_RAW %lld %lld\\n",gpu_total,cpu_raw);')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--work-dir", type=Path, required=True, help="New, nonexistent output directory")
    ap.add_argument("--platform", choices=("cuda", "rocm"), required=True)
    ap.add_argument("--compile-only", action="store_true")
    args = ap.parse_args()
    args.source, args.work_dir = args.source.resolve(), args.work_dir.resolve()
    try:
        args.work_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        ap.error("--work-dir must name a new, nonexistent directory")
    src, dst = args.source / "tier2/graph_tc/hip", args.work_dir / "tier2/graph_tc/hip"
    dst.mkdir(parents=True)
    for name in ("graph_tc.hip", "Makefile", "params.json"):
        shutil.copy2(src / name, dst / name)
    for header in src.glob("*.h"):
        shutil.copy2(header, dst / header.name)
    shutil.copytree(args.source / "common", args.work_dir / "common")
    source = (src / "graph_tc.hip").read_text()
    (dst / "graph_tc.hip").write_text(instrument(source))
    with (args.work_dir / "build.log").open("w") as log:
        subprocess.run(["make", "PLATFORM=" + args.platform], cwd=dst,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    if args.compile_only:
        print("Triangle-total regression compiled; GPU checks not run")
        return 0
    clean_env = {k: v for k, v in os.environ.items()
                 if not k.startswith(("BENCH_PARAM_", "TC_TEST_"))}
    results = []

    def report(complete=False):
        data = dict(complete=complete, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    cases=results, total=len(results), passed=sum(r["passed"] for r in results))
        (args.work_dir / "report.json").write_text(json.dumps(data, indent=2) + "\n")
        return data

    def run(n, degree, delta=0, location=0, verify="true", warm="false"):
        label = f"n{n}-d{degree}-delta{delta}-location{location}-{verify}-warm{warm}"
        env = dict(clean_env, BENCH_PARAM_vertices=str(n), BENCH_PARAM_avg_degree=str(degree),
                   BENCH_PARAM_verify=verify, BENCH_PARAM_vector_add_warmup=warm)
        if delta:
            env.update(TC_TEST_DELTA=str(delta), TC_TEST_LOCATION=str(location))
        begin = time.monotonic()
        try:
            proc = subprocess.run([str(dst / "graph_tc")], env=env, cwd=dst,
                                  text=True, capture_output=True, timeout=60)
        except subprocess.TimeoutExpired as exc:
            for suffix, value in (("stdout", exc.stdout), ("stderr", exc.stderr)):
                if isinstance(value, bytes):
                    value = value.decode(errors="replace")
                (args.work_dir / (label + "." + suffix)).write_text(value or "")
            results.append(dict(case=label, passed=False, errors=["timeout"],
                                exit_code=None, seconds=time.monotonic()-begin))
            report()
            raise SystemExit("Triangle-total test timed out; partial evidence retained in " + str(args.work_dir))
        (args.work_dir / (label + ".stdout")).write_text(proc.stdout)
        (args.work_dir / (label + ".stderr")).write_text(proc.stderr)
        errors = []
        disabled = verify in ("false", "0", "skip")
        expected_exit = 1 if delta and not disabled else 0
        if proc.returncode != expected_exit:
            errors.append(f"exit {proc.returncode}, expected {expected_exit}")
        if delta and f"TC_TEST_INJECT {delta} " not in proc.stderr:
            errors.append("device fault injection did not execute")
        if not disabled:
            match = re.search(r"TC_TEST_RAW (-?\d+) (-?\d+)", proc.stderr)
            if not match or int(match[1]) != int(match[2]) + delta or int(match[2]) % 3:
                errors.append("unexpected raw totals or reference multiplicity")
        try:
            records = [json.loads(line) for line in proc.stdout.splitlines()]
            if len(records) != 2 or records[0].get("type") != "kernel" or records[0].get("name") != "triangle_count_kernel" or records[1].get("type") != "summary":
                raise ValueError("expected one kernel and one summary")
            params = records[0]["params"]
            policy = "disabled" if disabled else "raw_triangle_total_cpu_v1"
            if params.get("verification") != policy or params.get("vertices") != n or params.get("avg_degree") != degree:
                errors.append("incorrect policy or workload parameters")
            timing = records[0]["time_ms"]
            if not math.isfinite(timing) or timing < 0 or not math.isclose(timing, records[1]["total_time_ms"], rel_tol=1e-5, abs_tol=1e-6):
                errors.append("invalid timing sum")
        except (ValueError, KeyError, TypeError) as exc:
            errors.append("invalid output: " + str(exc))
        results.append(dict(case=label, passed=not errors, errors=errors,
                            exit_code=proc.returncode, seconds=time.monotonic()-begin))
        report()
        print(("FAIL " if errors else "PASS ") + label + (": " + "; ".join(errors) if errors else ""), flush=True)

    for n, degree in ((256, 8), (256, 32), (256, 128), (1024, 32)):
        run(n, degree)
        for delta in (-3, -2, -1, 1, 2, 3):
            for location in (0, 1, 2):
                run(n, degree, delta, location)
    # A single undirected edge has no triangles. C++ division toward zero
    # also hid raw errors -1/-2 in the old quotient-only comparison.
    run(2, 1)
    for delta in (-3, -2, -1, 1, 2, 3):
        run(2, 1, delta)
    run(16384, 32)
    run(256, 32, warm="true")
    for setting in ("false", "0", "skip"):
        run(256, 32, verify=setting)
        run(256, 32, 1, 2, verify=setting)
    for setting in ("1", "unexpected"):
        run(256, 32, verify=setting)
    final = report(complete=True)
    print(f"{final['passed']}/{final['total']} triangle-total cases passed")
    return 0 if final["passed"] == final["total"] == 93 else 1


if __name__ == "__main__":
    raise SystemExit(main())
