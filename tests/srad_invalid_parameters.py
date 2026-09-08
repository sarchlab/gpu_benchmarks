#!/usr/bin/env python3
"""Check SRAD's early parameter rejection; optional positive controls need a GPU."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time


def run_case(binary, cli, env, label):
    start = time.monotonic()
    try:
        p = subprocess.run([str(binary), *cli], env=env, capture_output=True,
                           text=True, timeout=60)
        result = dict(exit=p.returncode, stdout=p.stdout, stderr=p.stderr, timed_out=False)
    except subprocess.TimeoutExpired as exc:
        # TimeoutExpired may carry bytes even when subprocess.run uses text=True.
        def decoded(output):
            return output.decode(errors="replace") if isinstance(output, bytes) else (output or "")

        result = dict(exit=None, stdout=decoded(exc.stdout), stderr=decoded(exc.stderr), timed_out=True)
    result.update(label=label, seconds=time.monotonic()-start)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary",type=Path,required=True)
    ap.add_argument("--gpu-controls",action="store_true")
    ap.add_argument("--report",type=Path)
    args = ap.parse_args()
    binary = args.binary.resolve()
    clean = {k:v for k,v in os.environ.items() if not k.startswith(("BENCH_PARAM_","SRAD_TEST_"))}
    cases = []
    for n in (0,-1,-16,-17,-2147483648):
        cases.append((f"image-{n}",[],n,1))
    for t in (0,-1,-2147483648):
        cases.append((f"iterations-{t}",[],16,t))
    cases += [("both-zero",[],0,0),("both-negative",[],-16,-1),
              ("environment-image-overrides-cli",["--image_size","64","--num_iterations","2"],0,1),
              ("environment-iterations-overrides-cli",["--image_size","64","--num_iterations","2"],16,0)]
    results = []
    for label,cli,n,t in cases:
        env = dict(clean,BENCH_PARAM_image_size=str(n),BENCH_PARAM_num_iterations=str(t),
                   BENCH_PARAM_vector_add_warmup="false")
        result = run_case(binary,cli,env,label)
        expected = f"Invalid SRAD parameters: image_size={n} num_iterations={t}; both must be positive\n"
        result["passed"] = result["exit"]==1 and result["stdout"]=="" and result["stderr"]==expected
        results.append(result)
    if args.gpu_controls:
        controls = [
            ("default",[],{},512,100),
            ("single-block",["--image_size","16","--num_iterations","1"],{},16,1),
            ("rounded",["--image_size","17","--num_iterations","2"],{},32,2),
            ("cli-image-fallback",["--image_size","0","--num_iterations","2"],{},512,2),
            ("cli-iteration-fallback",["--image_size","16","--num_iterations","0"],{},16,100),
            ("worker-below",[],{"image_size":"496","num_iterations":"2"},496,2),
            ("worker-threshold",[],{"image_size":"512","num_iterations":"2"},512,2),
            ("worker-above",[],{"image_size":"528","num_iterations":"2"},528,2),
            ("positive-environment-precedence",["--image_size","64","--num_iterations","1"],
             {"image_size":"32","num_iterations":"2"},32,2),
            ("largest",[],{"image_size":"8192","num_iterations":"200"},8192,200)]
        for label,cli,params,n,t in controls:
            env = dict(clean)
            if label!="default":env["BENCH_PARAM_vector_add_warmup"]="false"
            env.update({"BENCH_PARAM_"+k:v for k,v in params.items()})
            result = run_case(binary,cli,env,label)
            passed = result["exit"]==0 and result["seconds"]<60
            passed &= f"SRAD verification: PASS ({6*n*n} values, {7*n*n} comparisons, 0 mismatches)" in result["stderr"]
            try:
                first,second,summary = [json.loads(line) for line in result["stdout"].splitlines()]
                for kernel,name in zip((first,second),("srad_cuda_1","srad_cuda_2")):
                    passed &= kernel["type"]=="kernel" and kernel["name"]==name and kernel["iter"]==0
                    passed &= kernel["params"]=={"image_size":n,"num_iterations":t,"block_size":16,
                        "lambda":.5,"verification":"full_cpu_trajectory_and_step_v1"}
                    passed &= all(math.isfinite(kernel[f]) and kernel[f]>=0 for f in ("time_ms","host_time_ms"))
                passed &= summary["type"]=="summary" and abs(summary["total_time_ms"]-first["time_ms"]-second["time_ms"])<.000004
            except (ValueError,KeyError,TypeError):
                passed = False
            result["passed"] = bool(passed)
            results.append(result)
    report = dict(total=len(results),passed=sum(r["passed"] for r in results),gpu_controls=args.gpu_controls,results=results)
    if args.report:args.report.write_text(json.dumps(report,indent=2)+"\n")
    for result in results:
        if not result["passed"]:
            print("FAIL",result["label"],"(timeout after 60s)" if result["timed_out"] else "",
                  repr(result["stdout"]),repr(result["stderr"]))
    print(f"{report['passed']}/{report['total']} SRAD parameter cases passed")
    return 0 if report["passed"]==report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
