#!/usr/bin/env python3
"""Corrupt the actual measured GEMM output and verify ordinary-run failure."""
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
static void gemm_test_fault(float* output,int n) {
    const char* kind=std::getenv("GEMM_TEST_KIND");
    if(!kind)return;
    const char* location=std::getenv("GEMM_TEST_LOCATION");
    if(!location)std::abort();
    const int selection=std::atoi(location);
    if(selection<0 || selection>4)std::abort();
    const size_t elements=(size_t)n*n;
    const size_t positions[]={0,std::min(size_t(7),elements-1),
        std::min(size_t(8),elements-1),elements/2,elements-1};
    const bool many=!std::strcmp(kind,"many");
    const size_t count=many?12:1;
    if(count>elements)std::abort();
    for(size_t i=0;i<count;++i) {
        const size_t at=many?i:positions[selection];
        float value;
        if(!std::strcmp(kind,"nan"))value=std::numeric_limits<float>::quiet_NaN();
        else if(!std::strcmp(kind,"inf"))value=std::numeric_limits<float>::infinity();
        else if(!std::strcmp(kind,"negative_inf"))value=-std::numeric_limits<float>::infinity();
        else if(!std::strcmp(kind,"finite") || many) {
            HIP_CHECK(hipMemcpy(&value,output+at,sizeof(value),hipMemcpyDeviceToHost));
            value+=std::max(1000.0f,std::fabs(value)*0.25f);
        } else std::abort();
        HIP_CHECK(hipMemcpy(output+at,&value,sizeof(value),hipMemcpyHostToDevice));
    }
    std::fprintf(stderr,"GEMM_TEST_INJECT kind=%s count=%zu\n",kind,count);
}
'''
ANCHOR = "    // Check the complete measured C after timing, without another GPU pass.\n"
INPUT_ANCHOR = "    // Device allocations\n"
INPUT_HOOK = r'''
    if(const char* dataset=std::getenv("GEMM_TEST_DATASET")) {
        for(int i=0;i<N;++i)for(int j=0;j<N;++j) {
            const size_t q=(size_t)i*N+j;
            if(!std::strcmp(dataset,"zero"))h_A[q]=h_B[q]=h_C[q]=0;
            else if(!std::strcmp(dataset,"identity_a"))h_A[q]=i==j?1:0;
            else if(!std::strcmp(dataset,"identity_b"))h_B[q]=i==j?1:0;
            else if(!std::strcmp(dataset,"initial"))h_A[q]=h_B[q]=0;
            else if(!std::strcmp(dataset,"tail")) {
                h_A[q]=j==N-1?float(i+1)/8:0;
                h_B[q]=i==N-1?float(j+1)/8:0;
                h_C[q]=0;
            } else if(!std::strcmp(dataset,"signed")) {
                h_A[q]=float(int(q*7%31)-15)/8;
                h_B[q]=float(int(q*11%29)-14)/8;
                h_C[q]=float(int(q*13%23)-11)/8;
            } else if(std::strcmp(dataset,"random"))std::abort();
        }
    }
'''


def instrument(source):
    include="#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)\n"
    for anchor in (include,ANCHOR,INPUT_ANCHOR):
        if source.count(anchor)!=1:raise ValueError("expected one production anchor: "+anchor)
    return source.replace(include,include+HOOK).replace(
        ANCHOR,"    gemm_test_fault(d_C,N);\n"+ANCHOR).replace(INPUT_ANCHOR,INPUT_HOOK+INPUT_ANCHOR)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source",type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument("--work-dir",type=Path,required=True,help="New, nonexistent output directory")
    ap.add_argument("--platform",choices=("cuda","rocm"),required=True)
    ap.add_argument("--compile-only",action="store_true")
    args=ap.parse_args();args.source,args.work_dir=args.source.resolve(),args.work_dir.resolve()
    try:args.work_dir.mkdir(parents=True,exist_ok=False)
    except FileExistsError:ap.error("--work-dir must name a new, nonexistent directory")
    src=args.source/"tier2/polybench_gemm";dst=args.work_dir/"tier2/polybench_gemm";dst.mkdir(parents=True)
    shutil.copytree(args.source/"common",args.work_dir/"common")
    for name in ("Makefile","params.json"):shutil.copy2(src/name,dst/name)
    for header in src.glob("*.h"):shutil.copy2(header,dst/header.name)
    source=(src/"polybench_gemm.hip").read_text();(dst/"polybench_gemm.hip").write_text(instrument(source))
    with (args.work_dir/"build.log").open("w") as log:
        subprocess.run(["make","PLATFORM="+args.platform],cwd=dst,stdout=log,stderr=subprocess.STDOUT,check=True)
    if args.compile_only:
        print("GEMM regression compiled; GPU checks not run")
        return 0
    clean_env={k:v for k,v in os.environ.items() if not k.startswith(("BENCH_PARAM_","GEMM_TEST_"))}
    cases=[(n,None,0,"random","false") for n in (1,2,15,16,17,31,32,33,64,65,96,128,129,511,512,513)]
    datasets=("zero","identity_a","identity_b","initial","tail","signed")
    cases += [(n,None,0,dataset,"false") for n in (1,2,16,17,33) for dataset in datasets]
    cases += [(n,kind,at,"random","false") for n in (17,64)
              for kind in ("finite","nan","inf","negative_inf") for at in range(5)]
    cases += [(1,kind,4,"random","false") for kind in ("finite","nan","inf","negative_inf")]
    cases += [(17,"many",0,"random","false")]
    cases += [(17,"finite",at,dataset,"false") for dataset in datasets for at in range(5)]
    cases += [(n,"finite",4,"random","false") for n in (512,513)]
    cases += [(64,None,0,"random","true")]
    results=[]

    def report(complete=False):
        data=dict(complete=complete,total=len(results),expected=len(cases),passed=sum(r["passed"] for r in results),
                  source_sha256=hashlib.sha256(source.encode()).hexdigest(),results=results)
        (args.work_dir/"report.json").write_text(json.dumps(data,indent=2)+"\n")
        return data

    for n,kind,at,dataset,warm in cases:
        label=f"n{n}-{kind or 'control'}-p{at}-{dataset}-{warm}"
        env=dict(clean_env,BENCH_PARAM_size=str(n),BENCH_PARAM_vector_add_warmup=warm,GEMM_TEST_DATASET=dataset)
        if kind:env.update(GEMM_TEST_KIND=kind,GEMM_TEST_LOCATION=str(at))
        start=time.monotonic()
        try:
            p=subprocess.run([str(dst/"polybench_gemm")],env=env,capture_output=True,text=True,timeout=60)
            stdout,stderr,code=p.stdout,p.stderr,p.returncode
        except subprocess.TimeoutExpired as exc:
            stdout=(exc.stdout or b"").decode(errors="replace");stderr=(exc.stderr or b"").decode(errors="replace");code="timeout"
        elapsed=time.monotonic()-start
        (args.work_dir/(label+".stdout")).write_text(stdout);(args.work_dir/(label+".stderr")).write_text(stderr)
        errors=12 if kind=="many" else 1 if kind else 0
        passed=code==(1 if kind else 0) and elapsed<60
        passed &= f"GEMM verification: {'FAIL' if kind else 'PASS'} ({n*n} values, {errors} mismatches)" in stderr
        passed &= stderr.count("MISMATCH C[")==min(errors,10)
        passed &= ("GEMM_TEST_INJECT" in stderr)==bool(kind)
        try:
            kernel,summary=[json.loads(line) for line in stdout.splitlines()]
            passed &= kernel["type"]=="kernel" and kernel["name"]=="polybench_gemm_kernel" and kernel["iter"]==0
            passed &= kernel["params"]==dict(size=n,verification="full_cpu_v1")
            passed &= all(math.isfinite(kernel[f]) and kernel[f]>=0 for f in ("time_ms","host_time_ms"))
            passed &= summary["type"]=="summary" and abs(summary["total_time_ms"]-kernel["time_ms"])<.000002
        except (ValueError,KeyError,TypeError):passed=False
        results.append(dict(label=label,n=n,kind=kind,location=at,dataset=dataset,warm=warm,
                            exit_code=code,seconds=elapsed,passed=bool(passed)))
        report()
        if not passed:print("FAIL:",label,"see",args.work_dir,flush=True)
    final=report(complete=True)
    print(f"{final['passed']}/{final['total']} GEMM cases passed")
    return 0 if final['passed']==len(cases) else 1


if __name__=="__main__":
    raise SystemExit(main())
