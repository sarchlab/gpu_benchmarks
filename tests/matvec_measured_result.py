#!/usr/bin/env python3
"""Check complete matrix-vector verification with post-timing device faults."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

FIELDS = {"atax": (("y", "ny"),), "bicg": (("s", "n"), ("q", "m")),
          "mvt": (("x1", "n"), ("x2", "n"))}
HOOK = r'''
#include <limits>
static void matvec_test_fault(const char* name, float* output, int count) {
    const char* field = std::getenv("MATVEC_TEST_FIELD");
    if (!field || std::strcmp(field, name)) return;
    const char* index = std::getenv("MATVEC_TEST_INDEX");
    const char* kind = std::getenv("MATVEC_TEST_KIND");
    if (!index || !kind) std::abort();
    const size_t at = std::strtoull(index, nullptr, 10);
    if (at >= static_cast<size_t>(count)) std::abort();
    float value;
    HIP_CHECK(hipMemcpy(&value, output+at, sizeof(value), hipMemcpyDeviceToHost));
    if (!std::strcmp(kind, "finite")) value = 1000.0f * (std::fabs(value) + 1.0f);
    else if (!std::strcmp(kind, "nan")) value = std::numeric_limits<float>::quiet_NaN();
    else if (!std::strcmp(kind, "pinf")) value = std::numeric_limits<float>::infinity();
    else if (!std::strcmp(kind, "ninf")) value = -std::numeric_limits<float>::infinity();
    else std::abort();
    HIP_CHECK(hipMemcpy(output+at, &value, sizeof(value), hipMemcpyHostToDevice));
    std::fprintf(stderr, "MATVEC_TEST_INJECT %s %zu %s\n", name, at, kind);
}
'''


def instrument(source, name):
    declaration = "#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)\n"
    output = "    HIP_CHECK(hipMemcpy(h_" + FIELDS[name][0][0] + ","
    if source.count(declaration) != 1 or source.count(output) != 1:
        raise ValueError("expected one production declaration/download anchor")
    calls = "".join(f'    matvec_test_fault("{field}", d_{field}, {count});\n'
                    for field, count in FIELDS[name])
    return source.replace(declaration, declaration + HOOK).replace(output, calls + output)


def dimensions(name, rows, columns):
    if name == "atax":
        return dict(nx=rows, ny=columns, block_size=256)
    if name == "bicg":
        return dict(m=rows, n=columns, block_size=256)
    return dict(n=rows)


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
    shutil.copytree(args.source / "common", args.work_dir / "common")
    sources = {}
    for name in FIELDS:
        relative = Path("tier2") / ("polybench_" + name) / "hip"
        src, dst = args.source / relative, args.work_dir / relative
        dst.mkdir(parents=True)
        for file in ("Makefile", "params.json"):
            shutil.copy2(src / file, dst / file)
        for header in src.glob("*.h"):
            shutil.copy2(header, dst / header.name)
            sources[str(relative / header.name)] = hashlib.sha256(header.read_bytes()).hexdigest()
        filename = "polybench_" + name + ".hip"
        source = (src / filename).read_text()
        sources[str(relative / filename)] = hashlib.sha256(source.encode()).hexdigest()
        (dst / filename).write_text(instrument(source, name))
        with (dst / "build.log").open("w") as log:
            subprocess.run(["make", "PLATFORM=" + args.platform], cwd=dst,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    if args.compile_only:
        print("Matrix-vector regression compiled; GPU checks not run")
        return 0
    clean_env = {k: v for k, v in os.environ.items()
                 if not k.startswith(("BENCH_PARAM_", "MATVEC_TEST_"))}
    results = []

    def report(complete=False):
        data = dict(complete=complete, source_sha256=sources, cases=results,
                    total=len(results), passed=sum(r["passed"] for r in results))
        (args.work_dir / "report.json").write_text(json.dumps(data, indent=2) + "\n")
        return data

    def run(name, rows, columns, field=None, index=0, kind=None, warm="false"):
        label = f"{name}-{rows}-{columns}-{field}-{index}-{kind}-warm{warm}"
        params = dimensions(name, rows, columns)
        env = dict(clean_env, **{"BENCH_PARAM_" + k: str(v) for k, v in params.items()},
                   BENCH_PARAM_vector_add_warmup=warm)
        if field:
            env.update(MATVEC_TEST_FIELD=field, MATVEC_TEST_INDEX=str(index), MATVEC_TEST_KIND=kind)
        dst = args.work_dir / "tier2" / ("polybench_" + name) / "hip"
        begin = time.monotonic()
        try:
            proc = subprocess.run([str(dst / ("polybench_" + name))], cwd=dst, env=env,
                                  text=True, capture_output=True, timeout=60)
        except subprocess.TimeoutExpired as exc:
            for suffix, value in (("stdout", exc.stdout), ("stderr", exc.stderr)):
                if isinstance(value, bytes):
                    value = value.decode(errors="replace")
                (args.work_dir / (label + "." + suffix)).write_text(value or "")
            results.append(dict(case=label, passed=False, errors=["timeout"],
                                exit_code=None, seconds=time.monotonic()-begin))
            report()
            raise SystemExit("Matrix-vector test timed out; partial evidence retained in " + str(args.work_dir))
        for suffix, value in (("stdout", proc.stdout), ("stderr", proc.stderr)):
            (args.work_dir / (label + "." + suffix)).write_text(value)
        errors = []
        expected_exit = 1 if field else 0
        if proc.returncode != expected_exit:
            errors.append(f"exit {proc.returncode}, expected {expected_exit}")
        expected_diagnostic = "FAIL" if field else "PASS"
        if "Correctness check (" not in proc.stderr or expected_diagnostic not in proc.stderr:
            errors.append("missing normal verification diagnostic")
        if field:
            if f"MATVEC_TEST_INJECT {field} {index} {kind}" not in proc.stderr:
                errors.append("device fault injection did not execute")
            if f"{field}[{index}]" not in proc.stderr:
                errors.append("missing mismatch at the corrupted output")
        try:
            records = [json.loads(line) for line in proc.stdout.splitlines()]
            if len(records) != 3 or records[-1].get("type") != "summary":
                raise ValueError("expected two kernel records and one summary")
            for i, kernel in enumerate(records[:2]):
                if kernel.get("type") != "kernel" or kernel.get("name") != name + "_kernel" + str(i+1) or kernel.get("iter") != 0:
                    raise ValueError("unexpected kernel identity")
                if kernel["params"] != dict(params, verification="full_cpu_v1"):
                    raise ValueError("incorrect verification or workload identity")
                for key in ("time_ms", "host_time_ms"):
                    if not math.isfinite(kernel[key]) or kernel[key] < 0:
                        raise ValueError("invalid timing")
            total = records[-1]["total_time_ms"]
            if not math.isfinite(total) or not math.isclose(total, sum(k["time_ms"] for k in records[:2]), rel_tol=1e-5, abs_tol=2e-6):
                raise ValueError("invalid timing sum")
        except (ValueError, KeyError, TypeError) as exc:
            errors.append("invalid output: " + str(exc))
        results.append(dict(case=label, passed=not errors, errors=errors,
                            exit_code=proc.returncode, seconds=time.monotonic()-begin))
        report()
        print(("FAIL " if errors else "PASS ") + label + (": " + "; ".join(errors) if errors else ""), flush=True)

    for name, fields in FIELDS.items():
        for rows, columns in ((1, 1), (3, 5), (7, 9), (8, 8), (9, 7), (17, 19), (64, 64)):
            run(name, rows, columns)
        for field, _ in fields:
            for index in (0, 7, 8, 32, 63):
                for kind in ("finite", "nan", "pinf", "ninf"):
                    run(name, 64, 64, field, index, kind)
        # Unequal vector lengths catch swapped row/column comparison bounds.
        for rows, columns in ((17, 19), (19, 17)):
            params = dimensions(name, rows, columns)
            for field, count in fields:
                for kind in ("finite", "nan"):
                    run(name, rows, columns, field, params[count]-1, kind)
        run(name, 17, 19, warm="true")
        run(name, 4096, 4096)
    final = report(complete=True)
    print(f"{final['passed']}/{final['total']} matrix-vector cases passed")
    return 0 if final["passed"] == final["total"] == 147 else 1


if __name__ == "__main__":
    raise SystemExit(main())
