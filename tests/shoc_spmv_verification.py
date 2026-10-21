#!/usr/bin/env python3
"""Exercise full SHOC SpMV verification on the actual measured device output."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time


HOOK = r'''
#include <limits>
static void spmv_test_fault(float* output, int count) {
    const char* kind = std::getenv("SPMV_TEST_KIND");
    if (!kind) return;
    const char* index = std::getenv("SPMV_TEST_INDEX");
    if (!index) std::abort();
    const int at = std::atoi(index);
    const int length = !std::strcmp(kind,"many") ? 12 : 1;
    if (at < 0 || at > count-length) std::abort();
    for (int i = at; i < at+length; ++i) {
        float value;
        HIP_CHECK(hipMemcpy(&value,output+i,sizeof(value),hipMemcpyDeviceToHost));
        if (!std::strcmp(kind,"nan")) value=std::numeric_limits<float>::quiet_NaN();
        else if (!std::strcmp(kind,"pinf")) value=std::numeric_limits<float>::infinity();
        else if (!std::strcmp(kind,"ninf")) value=-std::numeric_limits<float>::infinity();
        else if (!std::strcmp(kind,"finite") || !std::strcmp(kind,"many"))
            value=1000.0f*(std::fabs(value)+1.0f);
        else std::abort();
        HIP_CHECK(hipMemcpy(output+i,&value,sizeof(value),hipMemcpyHostToDevice));
    }
    std::fprintf(stderr,"SPMV_TEST_INJECT %d %s %d\n",at,kind,length);
}
'''


def instrument(source):
    declaration = "#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)\n"
    download = "    // Copy result back\n"
    launch = "            spmv_csr_kernel<<<grid_size, BLOCK_SIZE>>>("
    for anchor in (declaration, download, launch):
        if source.count(anchor) != 1:
            raise ValueError("expected one production anchor: " + anchor)
    source = source.replace(declaration, declaration + HOOK)
    # Initialize skipped-work output outside timing, only in this test copy.
    anchor = '    double total_gpu_ms =\n'
    if source.count(anchor) != 1:
        raise ValueError("expected one timing anchor")
    source = source.replace(anchor, '    if (std::getenv("SPMV_TEST_SKIP"))\n'
                            '        HIP_CHECK(hipMemset(d_y,0,(size_t)num_rows*sizeof(float)));\n' + anchor)
    source = source.replace(launch, '            if (!std::getenv("SPMV_TEST_SKIP"))\n' + launch)
    return source.replace(download, '    spmv_test_fault(d_y,num_rows);\n' + download)


def host_reference(source, work, compiler):
    # Exercise the actual production CPU routine, with a separately derived
    # exact answer: empty row, duplicate columns, signed inputs and a tail row.
    start = source.index("static void cpu_spmv(")
    end = source.index("// ---------------------------------------------------------------------------", start)
    program = source[start:end] + r'''
int main() {
    const int rows[]={0,0,2,3,6,8}, columns[]={1,1,4,0,2,4,3,0};
    const float values[]={2,-0.5f,3,1,2,-1,-2,4}, x[]={1,2,3,4,5};
    float guarded[]={123,99,99,99,99,99,456};
    const float expected[]={123,0,3,15,2,-4,456};
    cpu_spmv(rows,columns,values,x,guarded+1,5);
    for (int i=0;i<7;++i) if (guarded[i]!=expected[i]) return 1;
    cpu_spmv(rows,columns,values,x,guarded,0);
    return guarded[0]==123 ? 0 : 1;
}
'''
    path = work / "reference.cpp"
    path.write_text(program)
    binary = work / "reference"
    subprocess.run([compiler, "-std=c++14", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                    str(path), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=20)
    print("PASS: production CPU reference, exact CSR cases and output guards", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--work-dir", type=Path, required=True, help="New, nonexistent directory")
    ap.add_argument("--host-only", action="store_true")
    ap.add_argument("--compiler", default="c++")
    ap.add_argument("--platform", choices=("cuda", "rocm"))
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="Also run all production manifest configurations")
    args = ap.parse_args()
    if not args.host_only and not args.platform:
        ap.error("--platform is required for GPU builds")
    args.source, args.work_dir = args.source.resolve(), args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=False)
    relative = Path("tier2/shoc_spmv/hip")
    src = args.source / relative
    source = (src / "shoc_spmv.hip").read_text()
    if args.host_only:
        host_reference(source, args.work_dir, args.compiler)
        return 0
    inputs = [relative / name for name in ("shoc_spmv.hip", "Makefile", "params.json")]
    inputs += [Path("common") / name for name in
               ("bench_timing.h", "bench_warmup.h", "hip_cuda_compat.h")]
    hashes = {str(p): hashlib.sha256((args.source/p).read_bytes()).hexdigest() for p in inputs}
    binaries = {}
    for variant in ("production", "faults"):
        root = args.work_dir / variant
        dst = root / relative
        dst.mkdir(parents=True)
        shutil.copytree(args.source / "common", root / "common")
        for name in ("Makefile", "params.json"):
            shutil.copy2(src/name, dst/name)
        (dst/"shoc_spmv.hip").write_text(source if variant == "production" else instrument(source))
        with (root/"build.log").open("w") as log:
            subprocess.run(["make", "PLATFORM="+args.platform], cwd=dst, check=True,
                           stdout=log, stderr=subprocess.STDOUT)
        binaries[variant] = dst/"shoc_spmv"
    if args.compile_only:
        print("Production and device-fault binaries compiled; GPU tests not run")
        return 0
    env_base = {k:v for k,v in os.environ.items() if not k.startswith(("BENCH_PARAM_", "SPMV_TEST_"))}
    cases = [("production", n, e, "false", None, 0) for n in (1,31,32,33,257,1024) for e in (4,32,64)]
    cases += [("faults", 257, 32, "false", kind, index)
              for kind in ("finite","nan","pinf","ninf") for index in (0,31,32,128,256)]
    cases += [("faults",257,32,"false","many",32), ("faults",257,32,"false","skip",0),
              ("faults",257,32,"true",None,0), ("faults",257,32,"true","finite",256)]
    if args.sweep:
        manifest = json.loads((src/"params.json").read_text())
        amounts = next(x["values"] for x in manifest["non_scaling_params"] if x["name"] == "nnz_per_row")
        cases += [("production", n, e, "true", None, 0)
                  for n in manifest["scaling_param"]["values"] for e in amounts]
    results = []

    def report(complete=False):
        data = dict(complete=complete, source_sha256=hashes,
                    binary_sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in binaries.items()},
                    expected=len(cases), total=len(results), passed=sum(x["passed"] for x in results), cases=results)
        (args.work_dir/"report.json").write_text(json.dumps(data, indent=2)+"\n")
        return data

    for variant,n,e,warm,kind,index in cases:
        label = f"{variant}-{n}-{e}-warm{warm}-{kind or 'control'}-{index}"
        env = dict(env_base, BENCH_PARAM_num_rows=str(n), BENCH_PARAM_nnz_per_row=str(e),
                   BENCH_PARAM_vector_add_warmup=warm)
        if kind == "skip": env["SPMV_TEST_SKIP"] = "1"
        elif kind: env.update(SPMV_TEST_KIND=kind, SPMV_TEST_INDEX=str(index))
        start = time.monotonic()
        try:
            p = subprocess.run([str(binaries[variant])], env=env, capture_output=True, text=True, timeout=60)
            stdout,stderr,code = p.stdout,p.stderr,p.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or b"").decode(errors="replace")
            stderr = (exc.stderr or b"").decode(errors="replace")
            code = "timeout"
        seconds = time.monotonic()-start
        (args.work_dir/(label+".stdout")).write_text(stdout)
        (args.work_dir/(label+".stderr")).write_text(stderr)
        errors = n if kind == "skip" else 12 if kind == "many" else 1 if kind else 0
        passed = code == (1 if kind else 0) and seconds < 60
        passed &= f"SpMV verification: {'FAIL' if kind else 'PASS'} ({n} rows, {errors} mismatches)" in stderr
        passed &= stderr.count("MISMATCH y[") == min(errors,10)
        if kind and kind != "skip":
            passed &= f"SPMV_TEST_INJECT {index} {kind} {errors}" in stderr
            passed &= f"MISMATCH y[{index}]" in stderr
        try:
            kernel,summary = [json.loads(line) for line in stdout.splitlines()]
            passed &= kernel["type"] == "kernel" and kernel["name"] == "spmv_csr_kernel" and kernel["iter"] == 0
            passed &= kernel["params"] == dict(num_rows=n,nnz_per_row=e,verification="full_cpu_v1")
            passed &= all(math.isfinite(kernel[k]) and kernel[k] >= 0 for k in ("time_ms","host_time_ms"))
            passed &= summary["type"] == "summary" and math.isfinite(summary["total_time_ms"])
            passed &= abs(summary["total_time_ms"]-kernel["time_ms"]) < 0.000002
        except (ValueError,KeyError,TypeError): passed = False
        results.append(dict(label=label,variant=variant,num_rows=n,nnz_per_row=e,warm=warm,
                            kind=kind,index=index,exit_code=code,seconds=seconds,passed=bool(passed)))
        report()
        print(f"{'PASS' if passed else 'FAIL'} {label}: {seconds:.3f}s", flush=True)
    final = report(complete=True)
    print(f"{final['passed']}/{final['total']} SHOC SpMV cases passed")
    return 0 if final["passed"] == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
