#!/usr/bin/env python3
"""Corrupt actual measured FDTD fields and verify ordinary-run failure."""
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
static void fdtd_test_fault(float* ex,float* ey,float* hz,int n) {
    const char* kind=std::getenv("FDTD_TEST_KIND");if(!kind)return;
    const char* text=std::getenv("FDTD_TEST_STAGE");
    const char* location=std::getenv("FDTD_TEST_LOCATION");
    if(!text || !location)std::abort();
    const int stage=std::atoi(text),selection=std::atoi(location);
    if(stage<0 || stage>3 || selection<0 || selection>5)std::abort();
    float* outputs[]={ex,ey,hz};const size_t elements=(size_t)n*n;
    const size_t positions[]={0,std::min(size_t(31),elements-1),
        std::min(size_t(32),elements-1),(size_t)(n/2)*n+n/2,elements-1,
        (size_t)std::max(0,n-2)*n+std::max(0,n-2)};
    const bool many=!std::strcmp(kind,"many"),ulp=!std::strcmp(kind,"boundary_ulp");
    const size_t count=many?12:1;if(count>elements)std::abort();
    for(int s=0;s<3;++s) {
        if(stage!=3 && stage!=s)continue;
        for(size_t i=0;i<count;++i) {
            size_t at=many?i:positions[selection];float value;
            if(ulp) {
                if(s==0)at=(size_t)(n/2)*n;
                else if(s==1)at=n/2;
                else at=selection==0?(size_t)(n-1)*n+n/2:(size_t)(n/2)*n+n-1;
            }
            if(!std::strcmp(kind,"nan"))value=std::numeric_limits<float>::quiet_NaN();
            else if(!std::strcmp(kind,"inf"))value=std::numeric_limits<float>::infinity();
            else if(!std::strcmp(kind,"negative_inf"))value=-std::numeric_limits<float>::infinity();
            else if(!std::strcmp(kind,"finite") || many || ulp) {
                HIP_CHECK(hipMemcpy(&value,outputs[s]+at,sizeof(value),hipMemcpyDeviceToHost));
                value=ulp?std::nextafter(value,std::numeric_limits<float>::infinity())
                         :value+std::max(1000.0f,std::fabs(value)*0.25f);
            } else std::abort();
            HIP_CHECK(hipMemcpy(outputs[s]+at,&value,sizeof(value),hipMemcpyHostToDevice));
        }
    }
    std::fprintf(stderr,"FDTD_TEST_INJECT stage=%d kind=%s count=%zu\n",stage,kind,count);
}
'''
ANCHOR = "    // Verify every measured final field after timing, without another GPU pass.\n"
INPUT_ANCHOR = "    // Device allocations\n"
INPUT_HOOK = r'''
    if(const char* dataset=std::getenv("FDTD_TEST_DATASET")) {
        for(int i=0;i<N;++i)for(int j=0;j<N;++j) {
            const size_t q=(size_t)i*N+j;
            if(!std::strcmp(dataset,"zero"))h_ex[q]=h_ey[q]=h_hz[q]=0;
            else if(!std::strcmp(dataset,"impulse")) {
                h_ex[q]=h_ey[q]=0;h_hz[q]=(i==N/2 && j==N/2)?1:0;
            } else if(!std::strcmp(dataset,"constant")) {
                h_ex[q]=1;h_ey[q]=2;h_hz[q]=3;
            } else if(!std::strcmp(dataset,"signed")) {
                h_ex[q]=float(int(q*7%31)-15)/8;
                h_ey[q]=float(int(q*11%29)-14)/8;
                h_hz[q]=float(int(q*13%23)-11)/8;
            } else if(std::strcmp(dataset,"random"))std::abort();
        }
    }
'''


def instrument(source):
    include="#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)\n"
    for anchor in (include,ANCHOR,INPUT_ANCHOR):
        if source.count(anchor)!=1:raise ValueError("expected one production anchor: "+anchor)
    return source.replace(include,include+HOOK).replace(
        ANCHOR,"    fdtd_test_fault(d_ex,d_ey,d_hz,N);\n"+ANCHOR).replace(INPUT_ANCHOR,INPUT_HOOK+INPUT_ANCHOR)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source",type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument("--work-dir",type=Path,required=True,help="New, nonexistent output directory")
    ap.add_argument("--platform",choices=("cuda","rocm"),required=True)
    ap.add_argument("--compile-only",action="store_true")
    args=ap.parse_args();args.source,args.work_dir=args.source.resolve(),args.work_dir.resolve()
    try:args.work_dir.mkdir(parents=True,exist_ok=False)
    except FileExistsError:ap.error("--work-dir must name a new, nonexistent directory")
    src=args.source/"tier2/polybench_fdtd2d/hip";dst=args.work_dir/"tier2/polybench_fdtd2d/hip";dst.mkdir(parents=True)
    shutil.copytree(args.source/"common_hip",args.work_dir/"common_hip")
    for name in ("Makefile","params.json"):shutil.copy2(src/name,dst/name)
    for header in src.glob("*.h"):shutil.copy2(header,dst/header.name)
    source=(src/"polybench_fdtd2d.hip").read_text();(dst/"polybench_fdtd2d.hip").write_text(instrument(source))
    with (args.work_dir/"build.log").open("w") as log:
        subprocess.run(["make","PLATFORM="+args.platform],cwd=dst,stdout=log,stderr=subprocess.STDOUT,check=True)
    if args.compile_only:
        print("FDTD regression compiled; GPU checks not run")
        return 0
    clean_env={k:v for k,v in os.environ.items() if not k.startswith(("BENCH_PARAM_","FDTD_TEST_"))}
    controls=((1,1),(2,1),(15,2),(16,10),(17,1),(17,10),(31,50),(32,200),(33,50),
              (64,10),(64,200),(65,50),(128,10),(129,50),(511,2),(512,2),(513,2))
    cases=[(n,t,None,0,0,"random","false") for n,t in controls]
    datasets=("zero","impulse","constant","signed")
    cases += [(n,t,None,0,0,dataset,"false") for n in (1,2,17,33) for t in (1,2,10) for dataset in datasets]
    cases += [(n,10,kind,stage,at,"random","false") for n in (17,64)
              for kind in ("finite","nan","inf","negative_inf") for stage in range(3) for at in range(6)]
    cases += [(1,1,kind,stage,4,"random","false") for kind in ("finite","nan","inf","negative_inf") for stage in range(3)]
    cases += [(17,10,"many",stage,0,"random","false") for stage in range(4)]
    cases += [(n,10,"boundary_ulp",stage,at,"random","false") for n in (1,17,64) for stage,at in ((0,0),(1,0),(2,0),(2,1))]
    cases += [(17,10,"finite",stage,4,dataset,"false") for dataset in datasets for stage in range(3)]
    cases += [(n,2,"finite",stage,4,"random","false") for n in (512,513) for stage in range(3)]
    cases += [(64,10,None,0,0,"random","true")]
    results=[]

    def report(complete=False):
        data=dict(complete=complete,total=len(results),expected=len(cases),passed=sum(r["passed"] for r in results),
                  source_sha256=hashlib.sha256(source.encode()).hexdigest(),results=results)
        (args.work_dir/"report.json").write_text(json.dumps(data,indent=2)+"\n")
        return data

    for n,t,kind,stage,at,dataset,warm in cases:
        label=f"n{n}-t{t}-{kind or 'control'}-s{stage}-p{at}-{dataset}-{warm}"
        env=dict(clean_env,BENCH_PARAM_size=str(n),BENCH_PARAM_tmax=str(t),BENCH_PARAM_vector_add_warmup=warm,FDTD_TEST_DATASET=dataset)
        if kind:env.update(FDTD_TEST_KIND=kind,FDTD_TEST_STAGE=str(stage),FDTD_TEST_LOCATION=str(at))
        start=time.monotonic()
        try:
            p=subprocess.run([str(dst/"polybench_fdtd2d")],env=env,capture_output=True,text=True,timeout=60)
            stdout,stderr,code=p.stdout,p.stderr,p.returncode
        except subprocess.TimeoutExpired as exc:
            stdout=(exc.stdout or b"").decode(errors="replace");stderr=(exc.stderr or b"").decode(errors="replace");code="timeout"
        elapsed=time.monotonic()-start
        (args.work_dir/(label+".stdout")).write_text(stdout);(args.work_dir/(label+".stderr")).write_text(stderr)
        errors=(12 if kind=="many" else 1)*(3 if stage==3 else 1) if kind else 0
        passed=code==(1 if kind else 0) and elapsed<60
        passed &= f"FDTD verification: {'FAIL' if kind else 'PASS'} ({3*n*n} values, {errors} mismatches)" in stderr
        passed &= stderr.count("MISMATCH ")==min(errors,10)
        passed &= ("FDTD_TEST_INJECT" in stderr)==bool(kind)
        if kind and stage!=3:
            names=("ex","ey","hz");passed &= f"MISMATCH {names[stage]}[" in stderr
            passed &= all(f"MISMATCH {name}[" not in stderr for i,name in enumerate(names) if i!=stage)
        try:
            first,second,third,summary=[json.loads(line) for line in stdout.splitlines()]
            kernels=(first,second,third)
            for kernel,name in zip(kernels,("fdtd_update_ex","fdtd_update_ey","fdtd_update_hz")):
                passed &= kernel["type"]=="kernel" and kernel["name"]==name and kernel["iter"]==0
                passed &= kernel["params"]==dict(size=n,tmax=t,verification="full_cpu_v1")
                passed &= all(math.isfinite(kernel[f]) and kernel[f]>=0 for f in ("time_ms","host_time_ms"))
            passed &= summary["type"]=="summary" and abs(summary["total_time_ms"]-sum(k["time_ms"] for k in kernels))<.000004
        except (ValueError,KeyError,TypeError):passed=False
        results.append(dict(label=label,n=n,tmax=t,kind=kind,stage=stage,location=at,dataset=dataset,warm=warm,
                            exit_code=code,seconds=elapsed,passed=bool(passed)))
        report()
        if not passed:print("FAIL:",label,"see",args.work_dir,flush=True)
    final=report(complete=True)
    print(f"{final['passed']}/{final['total']} FDTD cases passed")
    return 0 if final['passed']==len(cases) else 1


if __name__=="__main__":
    raise SystemExit(main())
