#!/usr/bin/env python3
"""Corrupt each actual measured 3MM output and verify ordinary-run failure."""
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
static void mm3_test_fault(float* e,float* f,float* g,int n) {
    const char* kind=std::getenv("MM3_TEST_KIND");if(!kind)return;
    const char* text=std::getenv("MM3_TEST_STAGE");
    const char* location=std::getenv("MM3_TEST_LOCATION");
    if(!text || !location)std::abort();
    const int stage=std::atoi(text),selection=std::atoi(location);
    if(stage<0 || stage>3 || selection<0 || selection>4)std::abort();
    float* outputs[]={e,f,g};const size_t elements=(size_t)n*n;
    const size_t positions[]={0,std::min(size_t(31),elements-1),
        std::min(size_t(32),elements-1),elements/2,elements-1};
    const bool many=!std::strcmp(kind,"many");const size_t count=many?12:1;
    if(count>elements)std::abort();
    for(int s=0;s<3;++s) {
        if(stage!=3 && stage!=s)continue;
        for(size_t i=0;i<count;++i) {
            const size_t at=many?i:positions[selection];float value;
            if(!std::strcmp(kind,"nan"))value=std::numeric_limits<float>::quiet_NaN();
            else if(!std::strcmp(kind,"inf"))value=std::numeric_limits<float>::infinity();
            else if(!std::strcmp(kind,"negative_inf"))value=-std::numeric_limits<float>::infinity();
            else if(!std::strcmp(kind,"finite") || many) {
                HIP_CHECK(hipMemcpy(&value,outputs[s]+at,sizeof(value),hipMemcpyDeviceToHost));
                value+=std::max(1000.0f,std::fabs(value)*0.25f);
            } else std::abort();
            HIP_CHECK(hipMemcpy(outputs[s]+at,&value,sizeof(value),hipMemcpyHostToDevice));
        }
    }
    std::fprintf(stderr,"MM3_TEST_INJECT stage=%d kind=%s count=%zu\n",stage,kind,count);
}
'''
ANCHOR = "    // Verify all three measured products after timing, with no extra GPU pass.\n"
INPUT_ANCHOR = "    // Device allocations\n"
INPUT_HOOK = r'''
    if(const char* dataset=std::getenv("MM3_TEST_DATASET")) {
        for(int i=0;i<N;++i)for(int j=0;j<N;++j) {
            const size_t q=(size_t)i*N+j;
            if(!std::strcmp(dataset,"zero"))h_A[q]=h_B[q]=h_C[q]=h_D[q]=0;
            else if(!std::strcmp(dataset,"identity_left"))h_A[q]=h_C[q]=i==j?1:0;
            else if(!std::strcmp(dataset,"identity_right"))h_B[q]=h_D[q]=i==j?1:0;
            else if(!std::strcmp(dataset,"first_zero"))h_A[q]=0;
            else if(!std::strcmp(dataset,"second_zero"))h_C[q]=0;
            else if(!std::strcmp(dataset,"tail")) {
                h_A[q]=h_C[q]=j==N-1?float(i+1)/8:0;
                h_B[q]=h_D[q]=i==N-1?float(j+1)/8:0;
            } else if(!std::strcmp(dataset,"signed")) {
                h_A[q]=float(int(q*7%31)-15)/32;
                h_B[q]=float(int(q*11%29)-14)/32;
                h_C[q]=float(int(q*13%23)-11)/32;
                h_D[q]=float(int(q*17%19)-9)/32;
            } else if(std::strcmp(dataset,"random"))std::abort();
        }
    }
'''


def instrument(source):
    include="#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)\n"
    for anchor in (include,ANCHOR,INPUT_ANCHOR):
        if source.count(anchor)!=1:raise ValueError("expected one production anchor: "+anchor)
    return source.replace(include,include+HOOK).replace(
        ANCHOR,"    mm3_test_fault(d_E,d_F,d_G,N);\n"+ANCHOR).replace(INPUT_ANCHOR,INPUT_HOOK+INPUT_ANCHOR)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source",type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument("--work-dir",type=Path,required=True,help="New, nonexistent output directory")
    ap.add_argument("--platform",choices=("cuda","rocm"),required=True)
    ap.add_argument("--compile-only",action="store_true")
    args=ap.parse_args();args.source,args.work_dir=args.source.resolve(),args.work_dir.resolve()
    try:args.work_dir.mkdir(parents=True,exist_ok=False)
    except FileExistsError:ap.error("--work-dir must name a new, nonexistent directory")
    src=args.source/"tier2/polybench_3mm/hip";dst=args.work_dir/"tier2/polybench_3mm/hip";dst.mkdir(parents=True)
    shutil.copytree(args.source/"common_hip",args.work_dir/"common_hip")
    for name in ("Makefile","params.json"):shutil.copy2(src/name,dst/name)
    for header in src.glob("*.h"):shutil.copy2(header,dst/header.name)
    source=(src/"polybench_3mm.hip").read_text();(dst/"polybench_3mm.hip").write_text(instrument(source))
    with (args.work_dir/"build.log").open("w") as log:
        subprocess.run(["make","PLATFORM="+args.platform],cwd=dst,stdout=log,stderr=subprocess.STDOUT,check=True)
    if args.compile_only:
        print("MM3 regression compiled; GPU checks not run")
        return 0
    clean_env={k:v for k,v in os.environ.items() if not k.startswith(("BENCH_PARAM_","MM3_TEST_"))}
    cases=[(n,None,0,0,"random","false") for n in (1,2,15,16,17,31,32,33,64,65,96,128,129,511,512,513)]
    datasets=("zero","identity_left","identity_right","tail","first_zero","second_zero","signed")
    cases += [(n,None,0,0,dataset,"false") for n in (1,2,16,17,33) for dataset in datasets]
    cases += [(n,kind,stage,at,"random","false") for n in (17,64)
              for kind in ("finite","nan","inf","negative_inf") for stage in range(3) for at in range(5)]
    cases += [(1,kind,stage,4,"random","false") for kind in ("finite","nan","inf","negative_inf") for stage in range(3)]
    cases += [(17,"many",stage,0,"random","false") for stage in range(4)]
    cases += [(17,"finite",stage,4,dataset,"false") for dataset in datasets for stage in range(3)]
    cases += [(n,"finite",stage,4,"random","false") for n in (512,513) for stage in range(3)]
    cases += [(64,None,0,0,"random","true")]
    results=[]

    def report(complete=False):
        data=dict(complete=complete,total=len(results),expected=len(cases),passed=sum(r["passed"] for r in results),
                  source_sha256=hashlib.sha256(source.encode()).hexdigest(),results=results)
        (args.work_dir/"report.json").write_text(json.dumps(data,indent=2)+"\n")
        return data

    for n,kind,stage,at,dataset,warm in cases:
        label=f"n{n}-{kind or 'control'}-s{stage}-p{at}-{dataset}-{warm}"
        env=dict(clean_env,BENCH_PARAM_size=str(n),BENCH_PARAM_vector_add_warmup=warm,MM3_TEST_DATASET=dataset)
        if kind:env.update(MM3_TEST_KIND=kind,MM3_TEST_STAGE=str(stage),MM3_TEST_LOCATION=str(at))
        start=time.monotonic()
        try:
            p=subprocess.run([str(dst/"polybench_3mm")],env=env,capture_output=True,text=True,timeout=60)
            stdout,stderr,code=p.stdout,p.stderr,p.returncode
        except subprocess.TimeoutExpired as exc:
            stdout=(exc.stdout or b"").decode(errors="replace");stderr=(exc.stderr or b"").decode(errors="replace");code="timeout"
        elapsed=time.monotonic()-start
        (args.work_dir/(label+".stdout")).write_text(stdout);(args.work_dir/(label+".stderr")).write_text(stderr)
        errors=(12 if kind=="many" else 1)*(3 if stage==3 else 1) if kind else 0
        passed=code==(1 if kind else 0) and elapsed<60
        passed &= f"3MM verification: {'FAIL' if kind else 'PASS'} ({3*n*n} values, {errors} mismatches)" in stderr
        passed &= stderr.count("MISMATCH ")==min(errors,10)
        passed &= ("MM3_TEST_INJECT" in stderr)==bool(kind)
        if kind and stage!=3:
            names=("E","F","G")
            passed &= f"MISMATCH {names[stage]}[" in stderr
            passed &= all(f"MISMATCH {name}[" not in stderr for i,name in enumerate(names) if i!=stage)
        try:
            first,second,third,summary=[json.loads(line) for line in stdout.splitlines()]
            kernels=(first,second,third)
            for kernel,name in zip(kernels,("mm3_kernel1","mm3_kernel2","mm3_kernel3")):
                passed &= kernel["type"]=="kernel" and kernel["name"]==name and kernel["iter"]==0
                passed &= kernel["params"]==dict(size=n,verification="full_cpu_v1")
                passed &= all(math.isfinite(kernel[f]) and kernel[f]>=0 for f in ("time_ms","host_time_ms"))
            passed &= summary["type"]=="summary" and abs(summary["total_time_ms"]-sum(k["time_ms"] for k in kernels))<.000004
        except (ValueError,KeyError,TypeError):passed=False
        results.append(dict(label=label,n=n,kind=kind,stage=stage,location=at,dataset=dataset,warm=warm,
                            exit_code=code,seconds=elapsed,passed=bool(passed)))
        report()
        if not passed:print("FAIL:",label,"see",args.work_dir,flush=True)
    final=report(complete=True)
    print(f"{final['passed']}/{final['total']} MM3 cases passed")
    return 0 if final['passed']==len(cases) else 1


if __name__=="__main__":
    raise SystemExit(main())
