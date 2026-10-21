#!/usr/bin/env python3
"""Verify ordinary CORRELATION output by corrupting the actual measured GPU buffers."""
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
static void correlation_test_fault(float* mean,float* stddev,float* data,float* corr,int n) {
    const char* kind=std::getenv("CORRELATION_TEST_KIND");
    if (!kind) return;
    const char* stage_text=std::getenv("CORRELATION_TEST_STAGE");
    const char* location=std::getenv("CORRELATION_TEST_LOCATION");
    if (!stage_text || !location) std::abort();
    const int stage=std::atoi(stage_text),selection=std::atoi(location);
    if (stage<0 || stage>4 || selection<0 || selection>5) std::abort();
    float* outputs[]={mean,stddev,data,corr};
    const bool many=!std::strcmp(kind,"many"),ulp=!std::strcmp(kind,"ulp");
    const size_t count=many?12:1;
    for (int s=0;s<4;++s) {
        if (stage!=4 && s!=stage) continue;
        const size_t elements=s<2?(size_t)n:(size_t)n*n;
        if (count>elements || (ulp && s!=3)) std::abort();
        const size_t positions[]={0,std::min(size_t(7),elements-1),
            std::min(size_t(8),elements-1),elements/2,elements-1,elements>1?elements-2:0};
        for (size_t i=0;i<count;++i) {
            const size_t at=ulp?(selection==0?0:selection==1?(n>1?(size_t)n+1:0):elements-1):many?i:positions[selection];
            float value;
            if (!std::strcmp(kind,"nan")) value=std::numeric_limits<float>::quiet_NaN();
            else if (!std::strcmp(kind,"inf")) value=std::numeric_limits<float>::infinity();
            else if (!std::strcmp(kind,"negative_inf")) value=-std::numeric_limits<float>::infinity();
            else if (ulp) value=std::nextafter(1.0f,2.0f);
            else if (!std::strcmp(kind,"finite") || many) {
                HIP_CHECK(hipMemcpy(&value,outputs[s]+at,sizeof(value),hipMemcpyDeviceToHost));
                value+=std::max(1000.0f,std::fabs(value)*0.25f);
            } else std::abort();
            HIP_CHECK(hipMemcpy(outputs[s]+at,&value,sizeof(value),hipMemcpyHostToDevice));
        }
    }
    std::fprintf(stderr,"CORRELATION_TEST_INJECT stage=%d kind=%s count=%zu\n",stage,kind,count);
}
'''
ANCHOR = "    // Verify all four measured outputs after timing, with no extra GPU pass.\n"
INPUT_ANCHOR = "    // Device allocations\n"
INPUT_HOOK = r'''
    if (const char* dataset=std::getenv("CORRELATION_TEST_DATASET")) {
        for (int i=0;i<M;++i) for(int j=0;j<N;++j) {
            float& value=h_data[(size_t)i*N+j];
            if (!std::strcmp(dataset,"zero")) value=0;
            else if (!std::strcmp(dataset,"constant")) value=2;
            else if (!std::strcmp(dataset,"mixed")) value=j%3==0?1.0f:j%3==1?(float)i/8:-(float)i/8;
            else if (!std::strcmp(dataset,"threshold")) {
                const float eps=1e-12f;
                const float scale=j%3==0?std::nextafter(eps,0.0f):j%3==1?eps:std::nextafter(eps,1.0f);
                value=(i%2?1.0f:-1.0f)*scale;
            } else if (!std::strcmp(dataset,"threshold_upper")) {
                const float scale=j%2?std::nextafter(1e-12f,1.0f):1e-12f;
                value=(i%2?1.0f:-1.0f)*scale;
            } else if (!std::strcmp(dataset,"low_variance")) value=(i%2?1.0f:-1.0f)*(j%2?2.0f:0.5f)*1e-12f;
            else if (std::strcmp(dataset,"random")) std::abort();
        }
    }
'''


def instrument(source):
    include="#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)\n"
    for anchor in (include,ANCHOR,INPUT_ANCHOR):
        if source.count(anchor)!=1:raise ValueError("expected one production anchor: "+anchor)
    return source.replace(include,include+HOOK).replace(
        ANCHOR,"    correlation_test_fault(d_mean,d_stddev,d_data,d_corr,N);\n"+ANCHOR).replace(
        INPUT_ANCHOR,INPUT_HOOK+INPUT_ANCHOR)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source",type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument("--work-dir",type=Path,required=True,help="New, nonexistent output directory")
    ap.add_argument("--platform",choices=("cuda","rocm"),required=True)
    ap.add_argument("--compile-only",action="store_true")
    args=ap.parse_args();args.source,args.work_dir=args.source.resolve(),args.work_dir.resolve()
    try:args.work_dir.mkdir(parents=True,exist_ok=False)
    except FileExistsError:ap.error("--work-dir must name a new, nonexistent directory")
    src=args.source/"tier2/polybench_correlation/hip";dst=args.work_dir/"tier2/polybench_correlation/hip";dst.mkdir(parents=True)
    shutil.copytree(args.source/"common",args.work_dir/"common")
    for name in ("Makefile","params.json"):shutil.copy2(src/name,dst/name)
    for header in src.glob("*.h"):shutil.copy2(header,dst/header.name)
    source=(src/"polybench_correlation.hip").read_text();(dst/"polybench_correlation.hip").write_text(instrument(source))
    with (args.work_dir/"build.log").open("w") as log:
        subprocess.run(["make","PLATFORM="+args.platform],cwd=dst,stdout=log,stderr=subprocess.STDOUT,check=True)
    if args.compile_only:
        print("CORRELATION regression compiled; GPU checks not run")
        return 0
    clean_env={k:v for k,v in os.environ.items() if not k.startswith(("BENCH_PARAM_","CORRELATION_TEST_"))}
    cases=[(n,None,0,0,"random","false") for n in (1,2,15,16,17,31,32,33,64,65,96,128,129)]
    cases += [(n,None,0,0,dataset,"false") for n in (1,2,16,17,32)
              for dataset in ("zero","constant","mixed")]
    cases += [(2,None,0,0,"threshold","false"),(2,None,0,0,"threshold_upper","false"),(16,None,0,0,"low_variance","false")]
    cases += [(n,kind,stage,at,"random","false") for n in (17,64)
              for kind in ("finite","nan","inf","negative_inf") for stage in range(4) for at in range(5)]
    cases += [(n,kind,3,5,"random","false") for n in (17,64) for kind in ("finite","nan","inf","negative_inf")]
    cases += [(1,kind,stage,4,"random","false") for kind in ("finite","nan","inf","negative_inf") for stage in range(4)]
    cases += [(17,"many",stage,0,"random","false") for stage in range(5)]
    cases += [(n,"ulp",3,at,"random","false") for n in (17,64) for at in (0,1,4)]
    cases += [(1,"ulp",3,0,"random","false")]
    cases += [(17,"finite",stage,4,dataset,"false") for stage in range(4) for dataset in ("zero","constant","mixed")]
    cases += [(64,None,0,0,"random","true")]
    results=[]

    def report(complete=False):
        data=dict(complete=complete,total=len(results),expected=len(cases),
                  passed=sum(r["passed"] for r in results),
                  source_sha256=hashlib.sha256(source.encode()).hexdigest(),results=results)
        (args.work_dir/"report.json").write_text(json.dumps(data,indent=2)+"\n")
        return data

    for n,kind,stage,at,dataset,warm in cases:
        label=f"n{n}-{kind or 'control'}-s{stage}-p{at}-{dataset}-{warm}"
        env=dict(clean_env,BENCH_PARAM_size=str(n),BENCH_PARAM_block_size="256",
                 BENCH_PARAM_vector_add_warmup=warm,CORRELATION_TEST_DATASET=dataset)
        if kind:env.update(CORRELATION_TEST_KIND=kind,CORRELATION_TEST_STAGE=str(stage),CORRELATION_TEST_LOCATION=str(at))
        start=time.monotonic()
        try:
            p=subprocess.run([str(dst/"polybench_correlation")],env=env,capture_output=True,text=True,timeout=60)
            stdout,stderr,code=p.stdout,p.stderr,p.returncode
        except subprocess.TimeoutExpired as exc:
            stdout=(exc.stdout or b"").decode(errors="replace");stderr=(exc.stderr or b"").decode(errors="replace");code="timeout"
        elapsed=time.monotonic()-start
        (args.work_dir/(label+".stdout")).write_text(stdout);(args.work_dir/(label+".stderr")).write_text(stderr)
        errors=(12 if kind=="many" else 1)*(4 if stage==4 else 1) if kind else 0
        passed=code==(1 if kind else 0) and elapsed<60
        passed &= f"Correlation verification: {'FAIL' if kind else 'PASS'} ({2*n+2*n*n} values, {errors} mismatches)" in stderr
        passed &= stderr.count("MISMATCH ")==min(errors,10)
        passed &= ("CORRELATION_TEST_INJECT" in stderr)==bool(kind)
        if kind and stage!=4:
            names=("mean","stddev","normalized","correlation")
            passed &= f"MISMATCH {names[stage]}[" in stderr
            passed &= all(f"MISMATCH {name}[" not in stderr for i,name in enumerate(names) if i!=stage)
        try:
            first,second,third,fourth,summary=[json.loads(line) for line in stdout.splitlines()]
            kernels=(first,second,third,fourth)
            for kernel,name in zip(kernels,("mean_kernel","stddev_kernel","normalize_kernel","correlation_kernel")):
                passed &= kernel["type"]=="kernel" and kernel["name"]==name and kernel["iter"]==0
                passed &= kernel["params"]==dict(size=n,block_size=256,verification="full_cpu_v1")
                passed &= all(math.isfinite(kernel[f]) and kernel[f]>=0 for f in ("time_ms","host_time_ms"))
            passed &= summary["type"]=="summary" and abs(summary["total_time_ms"]-sum(k["time_ms"] for k in kernels))<.000005
        except (ValueError,KeyError,TypeError):passed=False
        results.append(dict(label=label,n=n,kind=kind,stage=stage,location=at,dataset=dataset,warm=warm,
                            exit_code=code,seconds=elapsed,passed=bool(passed)))
        report()
        if not passed:print("FAIL:",label,"see",args.work_dir,flush=True)
    final=report(complete=True)
    print(f"{final['passed']}/{final['total']} CORRELATION cases passed")
    return 0 if final['passed']==len(cases) else 1


if __name__=="__main__":
    raise SystemExit(main())
