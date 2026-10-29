#!/usr/bin/env python3
"""Verify both ordinary 2MM products by corrupting the actual measured GPU buffers."""
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
#include <algorithm>
static void two_mm_test_fault(float* tmp, float* output, size_t elements) {
    const char* kind=std::getenv("TWOMM_TEST_KIND");
    if (!kind) return;
    const char* stage_text=std::getenv("TWOMM_TEST_STAGE");
    const char* location=std::getenv("TWOMM_TEST_LOCATION");
    if (!stage_text || !location) std::abort();
    const int stage=std::atoi(stage_text), selection=std::atoi(location);
    if (stage<0 || stage>2 || selection<0 || selection>4) std::abort();
    const size_t positions[]={0,std::min(size_t(7),elements-1),
        std::min(size_t(8),elements-1),elements/2,elements-1};
    const bool many=!std::strcmp(kind,"many");
    const size_t count=many?12:1;
    if (count>elements) std::abort();
    for (int s=0;s<2;++s) {
        if (stage!=2 && s!=stage) continue;
        float* buffer=s?output:tmp;
        for (size_t i=0;i<count;++i) {
            const size_t at=many?i:positions[selection];
            float value;
            if (!std::strcmp(kind,"nan")) value=std::numeric_limits<float>::quiet_NaN();
            else if (!std::strcmp(kind,"inf")) value=std::numeric_limits<float>::infinity();
            else if (!std::strcmp(kind,"negative_inf")) value=-std::numeric_limits<float>::infinity();
            else if (!std::strcmp(kind,"finite") || many) {
                HIP_CHECK(hipMemcpy(&value,buffer+at,sizeof(value),hipMemcpyDeviceToHost));
                value+=std::max(1000.0f,std::fabs(value)*0.25f);
            } else std::abort();
            HIP_CHECK(hipMemcpy(buffer+at,&value,sizeof(value),hipMemcpyHostToDevice));
        }
    }
    std::fprintf(stderr,"TWOMM_TEST_INJECT %d %zu %s %zu\n",stage,positions[selection],kind,count);
}
'''
ANCHOR = "    // Verify both measured products after timing, with no extra GPU pass.\n"


def instrument(source):
    include="#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)\n"
    for anchor in (include,ANCHOR):
        if source.count(anchor)!=1:raise ValueError("expected one production anchor: "+anchor)
    return source.replace(include,include+HOOK).replace(
        ANCHOR,"    two_mm_test_fault(d_tmp,d_D,elements);\n"+ANCHOR)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source",type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument("--work-dir",type=Path,required=True,help="New, nonexistent output directory")
    ap.add_argument("--platform",choices=("cuda","rocm"),required=True)
    ap.add_argument("--compile-only",action="store_true")
    args=ap.parse_args();args.source,args.work_dir=args.source.resolve(),args.work_dir.resolve()
    try:args.work_dir.mkdir(parents=True,exist_ok=False)
    except FileExistsError:ap.error("--work-dir must name a new, nonexistent directory")
    src=args.source/"tier2/polybench_2mm/hip";dst=args.work_dir/"tier2/polybench_2mm/hip";dst.mkdir(parents=True)
    shutil.copytree(args.source/"common_hip",args.work_dir/"common_hip")
    for name in ("Makefile","params.json"):shutil.copy2(src/name,dst/name)
    for header in src.glob("*.h"):shutil.copy2(header,dst/header.name)
    source=(src/"polybench_2mm.hip").read_text();(dst/"polybench_2mm.hip").write_text(instrument(source))
    with (args.work_dir/"build.log").open("w") as log:
        subprocess.run(["make","PLATFORM="+args.platform],cwd=dst,stdout=log,stderr=subprocess.STDOUT,check=True)
    if args.compile_only:
        print("2MM regression compiled; GPU checks not run")
        return 0
    clean_env={k:v for k,v in os.environ.items() if not k.startswith(("BENCH_PARAM_","TWOMM_TEST_"))}
    coefficients=((1.5,1.2),(0,-.75),(2.25,0),(-.5,1.25))
    cases=[(n,a,b,None,0,0,"false") for n in (1,2,15,16,17,31,32,33,64,65,96,128,129)
           for a,b in coefficients]
    cases += [(n,1.5,1.2,kind,stage,at,"false") for n in (17,64)
              for kind in ("finite","nan","inf","negative_inf") for stage in (0,1) for at in range(5)]
    cases += [(n,1.5,1.2,kind,stage,4,"false") for n in (1,129)
              for kind in ("finite","nan","inf","negative_inf") for stage in (0,1)]
    cases += [(17,1.5,1.2,"many",stage,0,"false") for stage in (0,1,2)]
    cases += [(17,a,b,"finite",stage,4,"false") for a,b in coefficients[1:] for stage in (0,1)]
    cases += [(64,1.5,1.2,None,0,0,"true")]
    results=[]

    def report(complete=False):
        data=dict(complete=complete,total=len(results),expected=len(cases),
                  passed=sum(r["passed"] for r in results),
                  source_sha256=hashlib.sha256(source.encode()).hexdigest(),results=results)
        (args.work_dir/"report.json").write_text(json.dumps(data,indent=2)+"\n")
        return data

    for n,a,b,kind,stage,at,warm in cases:
        label=f"n{n}-a{a}-b{b}-{kind or 'control'}-s{stage}-p{at}-{warm}"
        env=dict(clean_env,BENCH_PARAM_size=str(n),BENCH_PARAM_alpha=str(a),BENCH_PARAM_beta=str(b),
                 BENCH_PARAM_vector_add_warmup=warm)
        if kind:env.update(TWOMM_TEST_KIND=kind,TWOMM_TEST_STAGE=str(stage),TWOMM_TEST_LOCATION=str(at))
        start=time.monotonic()
        try:
            p=subprocess.run([str(dst/"polybench_2mm")],env=env,capture_output=True,text=True,timeout=60)
            stdout,stderr,code=p.stdout,p.stderr,p.returncode
        except subprocess.TimeoutExpired as exc:
            stdout=(exc.stdout or b"").decode(errors="replace");stderr=(exc.stderr or b"").decode(errors="replace");code="timeout"
        elapsed=time.monotonic()-start
        (args.work_dir/(label+".stdout")).write_text(stdout);(args.work_dir/(label+".stderr")).write_text(stderr)
        errors=(24 if stage==2 else 12) if kind=="many" else 1 if kind else 0
        passed=code==(1 if kind else 0) and elapsed<60
        passed &= f"2MM verification: {'FAIL' if kind else 'PASS'} ({n*n} elements per matrix, {errors} mismatches)" in stderr
        passed &= stderr.count("MISMATCH ")==min(errors,10)
        passed &= ("TWOMM_TEST_INJECT" in stderr)==bool(kind)
        if kind and stage!=2:
            passed &= ("MISMATCH D[" if stage else "MISMATCH tmp[") in stderr
            passed &= ("MISMATCH tmp[" if stage else "MISMATCH D[") not in stderr
        try:
            first,second,summary=[json.loads(line) for line in stdout.splitlines()]
            for k,name in zip((first,second),("mm_kernel_1","mm_kernel_2")):
                passed &= k["type"]=="kernel" and k["name"]==name and k["iter"]==0
                passed &= k["params"]==dict(size=n,alpha=a,beta=b,verification="full_cpu_v1")
                passed &= all(math.isfinite(k[f]) and k[f]>=0 for f in ("time_ms","host_time_ms"))
            passed &= summary["type"]=="summary" and abs(summary["total_time_ms"]-first["time_ms"]-second["time_ms"])<.000003
        except (ValueError,KeyError,TypeError):passed=False
        results.append(dict(label=label,n=n,alpha=a,beta=b,kind=kind,stage=stage,location=at,warm=warm,
                            exit_code=code,seconds=elapsed,passed=bool(passed)))
        report()
        if not passed:print("FAIL:",label,"see",args.work_dir,flush=True)
    final=report(complete=True)
    print(f"{final['passed']}/{final['total']} 2MM cases passed")
    return 0 if final['passed']==len(cases) else 1


if __name__=="__main__":
    raise SystemExit(main())
