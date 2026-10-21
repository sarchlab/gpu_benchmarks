#!/usr/bin/env python3
"""Check the production LUD pass and its measured-result verifier on a GPU.

Only isolated copies are instrumented. Faults modify actual device output
immediately after the timed factorization, before its first emitted record.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

POISON = r'''
    if (const char* kind = std::getenv("LUD_TEST_FAULT")) {
        const int at = std::atoi(std::getenv("LUD_TEST_INDEX"));
        float value = !std::strcmp(kind,"nan") ? std::numeric_limits<float>::quiet_NaN() :
                      !std::strcmp(kind,"pinf") ? std::numeric_limits<float>::infinity() :
                      !std::strcmp(kind,"ninf") ? -std::numeric_limits<float>::infinity() : 100000.0f;
        HIP_CHECK(hipMemcpy(d_a + at, &value, sizeof(value), hipMemcpyHostToDevice));
        std::fprintf(stderr,"LUD_TEST_INJECT %s %d\n",kind,at);
    }
'''


def instrument(source):
    pass_anchor = "        bench_lap_reset(&acc_diagonal);"
    result_anchor = '    bench_emit_iter("lud_diagonal",  0, &acc_diagonal,  params);'
    for anchor in (pass_anchor, result_anchor):
        if source.count(anchor) != 1:
            raise ValueError("expected one production anchor: " + anchor)
    source = '#include <limits>\n' + source
    source = source.replace(pass_anchor, '        std::fprintf(stderr,"LUD_TEST_PASS\\n");\n' + pass_anchor)
    return source.replace(result_anchor, POISON + result_anchor)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--platform", choices=("cuda", "rocm"), required=True)
    ap.add_argument("--compile-only", action="store_true")
    args = ap.parse_args()
    args.source = args.source.resolve()
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=False)
    src = args.source / "tier2/rodinia_lud/hip"
    dst = args.work_dir / "tier2/rodinia_lud/hip"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("rodinia_lud"))
    shutil.copytree(args.source / "common", args.work_dir / "common")
    source = (src / "rodinia_lud.hip").read_text()
    (dst / "rodinia_lud.hip").write_text(instrument(source))
    with (args.work_dir / "build.log").open("w") as log:
        subprocess.run(["make", "PLATFORM=" + args.platform], cwd=dst,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    if args.compile_only:
        print("LUD measured-result regression compiled; GPU checks not run")
        return 0

    clean_env = {k: v for k, v in os.environ.items()
                 if not k.startswith(("BENCH_PARAM_", "LUD_TEST_"))}
    results = []

    def run(n, kind=None, at=None, verify="true", warm="false"):
        disabled = verify in ("false", "0", "skip")
        label = f"n{n}-{kind or 'valid'}-{at}-{verify}-warm{warm}"
        env = dict(clean_env, BENCH_PARAM_size=str(n), BENCH_PARAM_verify=verify,
                   BENCH_PARAM_vector_add_warmup=warm)
        if kind:
            env.update(LUD_TEST_FAULT=kind, LUD_TEST_INDEX=str(at))
        begin = time.monotonic()
        proc = subprocess.run([str(dst / "rodinia_lud")], env=env, cwd=dst,
                              text=True, capture_output=True, timeout=60)
        seconds = time.monotonic() - begin
        (args.work_dir / (label + ".stdout")).write_text(proc.stdout)
        (args.work_dir / (label + ".stderr")).write_text(proc.stderr)
        errors = []
        expected_exit = 1 if kind and not disabled else 0
        if proc.returncode != expected_exit:
            errors.append(f"exit {proc.returncode}, expected {expected_exit}")
        if proc.stderr.count("LUD_TEST_PASS\n") != 1:
            errors.append("factorization did not run exactly once")
        if kind and f"LUD_TEST_INJECT {kind} {at}\n" not in proc.stderr:
            errors.append("device fault injection did not execute")
        try:
            records = [json.loads(line) for line in proc.stdout.splitlines()]
            kernels = [r for r in records if r.get("type") == "kernel"]
            summaries = [r for r in records if r.get("type") == "summary"]
            if len(records) != 4 or [r["name"] for r in kernels] != ["lud_diagonal", "lud_perimeter", "lud_internal"] or len(summaries) != 1:
                raise ValueError("expected three kernel records and one summary")
            policy = "disabled" if disabled else "full_lu_residual_v1"
            if any(r["params"].get("verification") != policy or r["params"].get("size") != n for r in kernels):
                errors.append("incorrect verification policy or size")
            times = [r["time_ms"] for r in kernels]
            if any(not math.isfinite(t) or t < 0 for t in times) or not math.isclose(sum(times), summaries[0]["total_time_ms"], rel_tol=1e-5, abs_tol=3e-6):
                errors.append("invalid timing sum")
        except (ValueError, KeyError, TypeError) as exc:
            errors.append("invalid output: " + str(exc))
        results.append(dict(case=label, passed=not errors, errors=errors, exit_code=proc.returncode, seconds=seconds))
        print(("PASS " if not errors else "FAIL ") + label + (": " + "; ".join(errors) if errors else ""), flush=True)

    for n in (64, 96, 512):
        run(n)
        for at in (0, 1, n, n+1, 15*n+15, 16*n+16, (n-1)*n, n*n-1):
            for kind in ("nan", "pinf", "ninf", "finite"):
                run(n, kind, at)
    run(64, warm="true")
    for setting in ("false", "0", "skip"):
        run(64, verify=setting)
        run(64, "nan", 64*64-1, verify=setting)
    for setting in ("1", "unexpected"):
        run(64, verify=setting)
    report = dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  cases=results, passed=sum(r["passed"] for r in results), total=len(results))
    (args.work_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['passed']}/{report['total']} measured-result cases passed")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
