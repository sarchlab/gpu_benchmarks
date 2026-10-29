#!/usr/bin/env python3
"""Verify ordinary SYR2K output by corrupting the actual measured GPU buffers."""
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
static void syr2k_test_fault(float* output, size_t elements) {
    const char* kind=std::getenv("SYR2K_TEST_KIND");
    if (!kind) return;
    const char* location=std::getenv("SYR2K_TEST_LOCATION");
    if (!location) std::abort();
    const int selection=std::atoi(location);
    if (selection<0 || selection>4) std::abort();
    const size_t positions[]={0,std::min(size_t(15),elements-1),
        std::min(size_t(16),elements-1),elements/2,elements-1};
    const bool many=!std::strcmp(kind,"many");
    const size_t count=many?12:1;
    if (count>elements) std::abort();
    for (size_t i=0;i<count;++i) {
        const size_t at=many?i:positions[selection];
        float value;
        if (!std::strcmp(kind,"nan")) value=std::numeric_limits<float>::quiet_NaN();
        else if (!std::strcmp(kind,"inf")) value=std::numeric_limits<float>::infinity();
        else if (!std::strcmp(kind,"negative_inf")) value=-std::numeric_limits<float>::infinity();
        else if (!std::strcmp(kind,"finite") || many) {
            HIP_CHECK(hipMemcpy(&value,output+at,sizeof(value),hipMemcpyDeviceToHost));
            value+=std::max(1000.0f,std::fabs(value)*0.25f);
        } else std::abort();
        HIP_CHECK(hipMemcpy(output+at,&value,sizeof(value),hipMemcpyHostToDevice));
    }
    std::fprintf(stderr,"SYR2K_TEST_INJECT %zu %s %zu\n",positions[selection],kind,count);
}
'''
ANCHOR = "    // Verify the complete measured result after timing, without another GPU pass.\n"


def instrument(source):
    include="#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)\n"
    for anchor in (include,ANCHOR):
        if source.count(anchor)!=1:raise ValueError("expected one production anchor: "+anchor)
    return source.replace(include,include+HOOK).replace(
        ANCHOR,"    syr2k_test_fault(d_C,(size_t)N*N);\n"+ANCHOR)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source",type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument("--work-dir",type=Path,required=True,help="New, nonexistent output directory")
    ap.add_argument("--platform",choices=("cuda","rocm"),required=True)
    ap.add_argument("--compile-only",action="store_true")
    args=ap.parse_args();args.source,args.work_dir=args.source.resolve(),args.work_dir.resolve()
    try:args.work_dir.mkdir(parents=True,exist_ok=False)
    except FileExistsError:ap.error("--work-dir must name a new, nonexistent directory")
    src=args.source/"tier2/polybench_syr2k/hip";dst=args.work_dir/"tier2/polybench_syr2k/hip";dst.mkdir(parents=True)
    shutil.copytree(args.source/"common_hip",args.work_dir/"common_hip")
    for name in ("Makefile","params.json"):shutil.copy2(src/name,dst/name)
    for header in src.glob("*.h"):shutil.copy2(header,dst/header.name)
    source=(src/"polybench_syr2k.hip").read_text();(dst/"polybench_syr2k.hip").write_text(instrument(source))
    with (args.work_dir/"build.log").open("w") as log:
        subprocess.run(["make","PLATFORM="+args.platform],cwd=dst,stdout=log,stderr=subprocess.STDOUT,check=True)
    if args.compile_only:
        print("SYR2K regression compiled; GPU checks not run")
        return 0
    clean_env={k:v for k,v in os.environ.items() if not k.startswith(("BENCH_PARAM_","SYR2K_TEST_"))}
    coefficients=((1.5,1.2),(0,-.75),(2.25,0),(-.5,1.25))
    shapes=((1,1),(2,3),(15,17),(16,16),(17,15),(31,33),
            (32,32),(33,31),(64,256),(65,17),(97,65),(17,2048))
    cases=[(n,m,a,b,None,0,"false") for n,m in shapes for a,b in coefficients]
    cases += [(n,m,1.5,1.2,kind,at,"false") for n,m in ((17,19),(64,256))
              for kind in ("finite","nan","inf","negative_inf") for at in range(5)]
    cases += [(n,m,1.5,1.2,kind,4,"false") for n,m in ((1,1),(129,65))
              for kind in ("finite","nan","inf","negative_inf")]
    cases += [(17,19,1.5,1.2,"many",0,"false")]
    cases += [(17,19,a,b,"finite",4,"false") for a,b in coefficients[1:]]
    cases += [(64,256,1.5,1.2,None,0,"true")]
    results=[]

    def report(complete=False):
        data=dict(complete=complete,total=len(results),expected=len(cases),
                  passed=sum(r["passed"] for r in results),
                  source_sha256=hashlib.sha256(source.encode()).hexdigest(),results=results)
        (args.work_dir/"report.json").write_text(json.dumps(data,indent=2)+"\n")
        return data

    for n,m,a,b,kind,at,warm in cases:
        label=f"n{n}-m{m}-a{a}-b{b}-{kind or 'control'}-p{at}-{warm}"
        env=dict(clean_env,BENCH_PARAM_matrix_size=str(n),BENCH_PARAM_inner_size=str(m),
                 BENCH_PARAM_alpha=str(a),BENCH_PARAM_beta=str(b),
                 BENCH_PARAM_vector_add_warmup=warm)
        if kind:env.update(SYR2K_TEST_KIND=kind,SYR2K_TEST_LOCATION=str(at))
        start=time.monotonic()
        try:
            p=subprocess.run([str(dst/"polybench_syr2k")],env=env,capture_output=True,text=True,timeout=60)
            stdout,stderr,code=p.stdout,p.stderr,p.returncode
        except subprocess.TimeoutExpired as exc:
            stdout=(exc.stdout or b"").decode(errors="replace");stderr=(exc.stderr or b"").decode(errors="replace");code="timeout"
        elapsed=time.monotonic()-start
        (args.work_dir/(label+".stdout")).write_text(stdout);(args.work_dir/(label+".stderr")).write_text(stderr)
        errors=12 if kind=="many" else 1 if kind else 0
        passed=code==(1 if kind else 0) and elapsed<60
        passed &= f"SYR2K verification: {'FAIL' if kind else 'PASS'} ({n*n} elements, {errors} mismatches)" in stderr
        passed &= stderr.count("MISMATCH ")==min(errors,10)
        passed &= ("SYR2K_TEST_INJECT" in stderr)==bool(kind)
        if kind: passed &= "MISMATCH C[" in stderr
        try:
            kernel,summary=[json.loads(line) for line in stdout.splitlines()]
            passed &= kernel["type"]=="kernel" and kernel["name"]=="syr2k_kernel" and kernel["iter"]==0
            passed &= kernel["params"]==dict(matrix_size=n,inner_size=m,alpha=a,beta=b,verification="full_cpu_v1")
            passed &= all(math.isfinite(kernel[f]) and kernel[f]>=0 for f in ("time_ms","host_time_ms"))
            passed &= summary["type"]=="summary" and abs(summary["total_time_ms"]-kernel["time_ms"])<.000003
        except (ValueError,KeyError,TypeError):passed=False
        results.append(dict(label=label,n=n,m=m,alpha=a,beta=b,kind=kind,location=at,warm=warm,
                            exit_code=code,seconds=elapsed,passed=bool(passed)))
        report()
        if not passed:print("FAIL:",label,"see",args.work_dir,flush=True)
    final=report(complete=True)
    print(f"{final['passed']}/{final['total']} SYR2K cases passed")
    return 0 if final['passed']==len(cases) else 1


if __name__=="__main__":
    raise SystemExit(main())
