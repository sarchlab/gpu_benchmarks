#!/usr/bin/env python3
"""Check every measured FIR sample, including startup, tails and device faults."""
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
static void fir_test_fault(float* output, int n) {
    const char* kind=std::getenv("FIR_TEST_KIND");
    if (!kind) return;
    const char* index=std::getenv("FIR_TEST_INDEX");
    if (!index) std::abort();
    int at=std::atoi(index), count=!std::strcmp(kind,"many")?12:1;
    if (at<0 || at>n-count) std::abort();
    for (int i=at;i<at+count;++i) {
        float value;
        if (!std::strcmp(kind,"nan")) value=std::numeric_limits<float>::quiet_NaN();
        else if (!std::strcmp(kind,"pinf")) value=std::numeric_limits<float>::infinity();
        else if (!std::strcmp(kind,"ninf")) value=-std::numeric_limits<float>::infinity();
        else if (!std::strcmp(kind,"finite") || !std::strcmp(kind,"many")) value=1000.0f;
        else std::abort();
        HIP_CHECK(hipMemcpy(output+i,&value,sizeof(value),hipMemcpyHostToDevice));
    }
    std::fprintf(stderr,"FIR_TEST_INJECT %d %s %d\n",at,kind,count);
}
'''


def instrument(source):
    declaration = "#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)\n"
    download = "        HIP_CHECK(hipMemcpy(h_output, d_output, output_bytes, hipMemcpyDeviceToHost));"
    for anchor in (declaration, download):
        if source.count(anchor) != 1:
            raise ValueError("expected one production anchor: " + anchor)
    source = source.replace(declaration, declaration + HOOK)
    return source.replace(download, "        fir_test_fault(d_output,N);\n" + download)


def host_reference(source, work, compiler):
    start = source.index("static void fir_cpu(")
    end = source.index("// ---------------------------------------------------------------------------", start)
    program = "#include <vector>\n" + source[start:end] + r'''
int main() {
    const float input[]={1,2,-1,4,-2}, coeff[]={0.5f,-0.25f,0.125f};
    float guarded[]={123,99,99,99,99,99,456};
    const float expected[]={123,0.5f,0.75f,-0.875f,2.5f,-2.125f,456};
    fir_cpu(input,guarded+1,coeff,5,3);
    for (int i=0;i<7;++i) if(guarded[i]!=expected[i]) return 1;
    fir_cpu(input,guarded,coeff,0,3);
    if(guarded[0]!=123) return 2;

    const float scale=-2;
    std::vector<float> signed_input(1033),scaled(1035,123);
    for(int i=0;i<1033;++i) signed_input[i]=(float)(i%11-5);
    fir_cpu(signed_input.data(),scaled.data()+1,&scale,1033,1);
    for(int i=0;i<1033;++i) if(scaled[i+1]!=scale*signed_input[i]) return 3;
    if(scaled.front()!=123 || scaled.back()!=123) return 4;

    // An impulse beyond the old cutoff has a known complete response: twice
    // the coefficients, then zero. No copy of the production recurrence.
    for(int taps: {1,16,32,64,128}) {
        std::vector<float> impulse(1153,0),c(taps),actual(1155,123);
        impulse[1024]=2;
        for(int k=0;k<taps;++k) c[k]=(float)(k%7+1)*0.125f;
        fir_cpu(impulse.data(),actual.data()+1,c.data(),1153,taps);
        for(int i=0;i<1153;++i) {
            float want=(i>=1024 && i<1024+taps)?2*c[i-1024]:0;
            if(actual[i+1]!=want) return 5;
        }
        if(actual.front()!=123 || actual.back()!=123) return 6;
    }
}
'''
    path = work / "reference.cpp"
    path.write_text(program)
    binary = work / "reference"
    subprocess.run([compiler, "-O2", "-std=c++14", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                    str(path), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=20)
    print("PASS: production CPU FIR startup, signed taps, complete impulse responses and output guards", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--work-dir", type=Path, required=True, help="New, nonexistent directory")
    ap.add_argument("--host-only", action="store_true")
    ap.add_argument("--compiler", default="c++")
    ap.add_argument("--platform", choices=("cuda", "rocm"))
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    if not args.host_only and not args.platform:
        ap.error("--platform is required for GPU builds")
    args.source, args.work_dir = args.source.resolve(), args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=False)
    relative = Path("tier2/heteromark_fir/hip")
    src = args.source / relative
    source = (src / "heteromark_fir.hip").read_text()
    if args.host_only:
        host_reference(source, args.work_dir, args.compiler)
        return 0
    inputs = [relative / name for name in ("heteromark_fir.hip", "Makefile", "params.json")]
    inputs += [Path("common") / name for name in ("bench_timing.h", "bench_warmup.h", "hip_cuda_compat.h")]
    hashes = {str(p): hashlib.sha256((args.source / p).read_bytes()).hexdigest() for p in inputs}
    binaries = {}
    for variant in ("production", "faults"):
        root = args.work_dir / variant
        dst = root / relative
        dst.mkdir(parents=True)
        shutil.copytree(args.source / "common", root / "common")
        for name in ("Makefile", "params.json"):
            shutil.copy2(src / name, dst / name)
        (dst / "heteromark_fir.hip").write_text(source if variant == "production" else instrument(source))
        with (root / "build.log").open("w") as log:
            subprocess.run(["make", "PLATFORM=" + args.platform], cwd=dst, check=True,
                           stdout=log, stderr=subprocess.STDOUT)
        binaries[variant] = dst / "heteromark_fir"
    if args.compile_only:
        print("Production and measured-output fault binaries compiled; GPU tests not run")
        return 0
    clean_env = {k:v for k,v in os.environ.items() if not k.startswith(("BENCH_PARAM_", "FIR_TEST_"))}
    cases = [("production", n, 128, "false", None, 0)
             for n in (1,15,16,17,127,128,129,255,256,257,1023,1024,1025,2049)]
    cases += [("production", 2049, t, "false", None, 0) for t in (16,32,64)]
    cases += [("faults", 2049, 128, "false", kind, index)
              for kind in ("finite","nan","pinf","ninf") for index in (0,1023,1024,1536,2048)]
    cases += [("faults",2049,128,"false","many",1024),
              ("production",2049,128,"true",None,0), ("faults",2049,128,"true","finite",2048)]
    if args.sweep:
        manifest = json.loads((src / "params.json").read_text())
        taps = next(x["values"] for x in manifest["non_scaling_params"] if x["name"] == "num_taps")
        cases += [("production", n, t, "true", None, 0)
                  for n in manifest["scaling_param"]["values"] for t in taps]
    results = []

    def report(complete=False):
        data = dict(complete=complete, source_sha256=hashes,
                    binary_sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in binaries.items()},
                    expected=len(cases), total=len(results), passed=sum(x["passed"] for x in results), cases=results)
        (args.work_dir / "report.json").write_text(json.dumps(data, indent=2) + "\n")
        return data

    for variant,n,taps,warm,kind,index in cases:
        label = f"{variant}-{n}-{taps}-warm{warm}-{kind or 'control'}-{index}"
        env = dict(clean_env, BENCH_PARAM_size=str(n), BENCH_PARAM_num_taps=str(taps),
                   BENCH_PARAM_block_size="256", BENCH_PARAM_vector_add_warmup=warm)
        if kind:
            env.update(FIR_TEST_KIND=kind, FIR_TEST_INDEX=str(index))
        start = time.monotonic()
        try:
            p = subprocess.run([str(binaries[variant])], env=env, capture_output=True, text=True, timeout=60)
            stdout,stderr,code = p.stdout,p.stderr,p.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or b"").decode(errors="replace")
            stderr = (exc.stderr or b"").decode(errors="replace")
            code = "timeout"
        seconds = time.monotonic() - start
        (args.work_dir / (label + ".stdout")).write_text(stdout)
        (args.work_dir / (label + ".stderr")).write_text(stderr)
        errors = 12 if kind == "many" else 1 if kind else 0
        passed = code == (1 if kind else 0) and seconds < 60
        passed &= f"FIR verification: {'FAIL' if kind else 'PASS'} ({n} samples, {errors} mismatches)" in stderr
        passed &= stderr.count("Mismatch at ") == min(errors,10)
        if kind:
            passed &= f"FIR_TEST_INJECT {index} {kind} {errors}" in stderr
            passed &= f"Mismatch at {index}:" in stderr
        try:
            k,s = [json.loads(line) for line in stdout.splitlines()]
            passed &= k["type"] == "kernel" and k["name"] == "fir_filter_kernel" and k["iter"] == 0
            passed &= k["params"] == dict(size=n,num_taps=taps,block_size=256,verification="full_cpu_v1")
            passed &= all(math.isfinite(k[f]) and k[f] >= 0 for f in ("time_ms", "host_time_ms"))
            passed &= s["type"] == "summary" and math.isfinite(s["total_time_ms"])
            passed &= abs(s["total_time_ms"] - k["time_ms"]) < 0.000002
        except (ValueError,KeyError,TypeError):
            passed = False
        results.append(dict(label=label,variant=variant,size=n,num_taps=taps,warm=warm,
                            kind=kind,index=index,exit_code=code,seconds=seconds,passed=bool(passed)))
        report()
        print(f"{'PASS' if passed else 'FAIL'} {label}: {seconds:.3f}s", flush=True)
    final = report(complete=True)
    print(f"{final['passed']}/{final['total']} FIR cases passed")
    return 0 if final["passed"] == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
