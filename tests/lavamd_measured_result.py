#!/usr/bin/env python3
"""Verify ordinary LAVAMD output by corrupting the actual measured GPU buffers."""
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
static void lavamd_test_fault(FOUR_VECTOR* output, size_t particles) {
    const char* kind=std::getenv("LAVAMD_TEST_KIND");
    if (!kind) return;
    const char* location=std::getenv("LAVAMD_TEST_LOCATION");
    const char* channel=std::getenv("LAVAMD_TEST_COMPONENT");
    if (!location || !channel) std::abort();
    const int selection=std::atoi(location),component=std::atoi(channel);
    if (selection<0 || selection>4 || component<0 || component>4) std::abort();
    const size_t positions[]={0,std::min(size_t(99),particles-1),
        std::min(size_t(100),particles-1),particles/2,particles-1};
    const bool many=!std::strcmp(kind,"many");
    const size_t count=many?12:1;
    if (count>particles) std::abort();
    for (size_t i=0;i<count;++i) {
        const size_t at=many?i:positions[selection];
        double* addresses[]={&output[at].v,&output[at].x,&output[at].y,&output[at].z};
        for (int c=0;c<4;++c) {
            if (component!=4 && c!=component) continue;
            double value;
            if (!std::strcmp(kind,"nan")) value=std::numeric_limits<double>::quiet_NaN();
            else if (!std::strcmp(kind,"inf")) value=std::numeric_limits<double>::infinity();
            else if (!std::strcmp(kind,"negative_inf")) value=-std::numeric_limits<double>::infinity();
            else if (!std::strcmp(kind,"finite") || many) {
                HIP_CHECK(hipMemcpy(&value,addresses[c],sizeof(value),hipMemcpyDeviceToHost));
                value+=std::max(1000.0,std::fabs(value)*0.25);
            } else std::abort();
            HIP_CHECK(hipMemcpy(addresses[c],&value,sizeof(value),hipMemcpyHostToDevice));
        }
    }
    std::fprintf(stderr,"LAVAMD_TEST_INJECT %zu %s %zu component=%d\n",positions[selection],kind,count,component);
}
'''
ANCHOR = "    // Verify every measured energy/force component after timing, with no extra GPU pass.\n"
INPUT_ANCHOR = "    // Device buffers\n"
INPUT_HOOK = r'''
    if (const char* dataset=std::getenv("LAVAMD_TEST_DATASET")) {
        for (long i=0;i<dim_cpu.space_elem;++i) {
            if (!std::strcmp(dataset,"zero_charge")) qv_cpu[i]=0;
            else if (!std::strcmp(dataset,"uniform")) {
                rv_cpu[i]=FOUR_VECTOR{};qv_cpu[i]=1;
            } else if (!std::strcmp(dataset,"signed_charge")) {
                if (i%2) qv_cpu[i]=-qv_cpu[i];
            } else if (std::strcmp(dataset,"random")) std::abort();
        }
    }
'''


def instrument(source):
    include="// KERNEL — verbatim from Rodinia lavaMD/kernel/kernel_gpu_cuda.cu\n"
    for anchor in (include,ANCHOR,INPUT_ANCHOR):
        if source.count(anchor)!=1:raise ValueError("expected one production anchor: "+anchor)
    return source.replace(include,HOOK+include).replace(
        ANCHOR,"    lavamd_test_fault(d_fv_gpu,dim_cpu.space_elem);\n"+ANCHOR).replace(
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
    src=args.source/"tier2/rodinia_lavamd/hip";dst=args.work_dir/"tier2/rodinia_lavamd/hip";dst.mkdir(parents=True)
    shutil.copytree(args.source/"common",args.work_dir/"common")
    for name in ("Makefile","params.json"):shutil.copy2(src/name,dst/name)
    for header in src.glob("*.h"):shutil.copy2(header,dst/header.name)
    source=(src/"rodinia_lavamd.hip").read_text();(dst/"rodinia_lavamd.hip").write_text(instrument(source))
    with (args.work_dir/"build.log").open("w") as log:
        subprocess.run(["make","PLATFORM="+args.platform],cwd=dst,stdout=log,stderr=subprocess.STDOUT,check=True)
    if args.compile_only:
        print("LAVAMD regression compiled; GPU checks not run")
        return 0
    clean_env={k:v for k,v in os.environ.items() if not k.startswith(("BENCH_PARAM_","LAVAMD_TEST_"))}
    cases=[(n,None,0,0,dataset,"false") for n in (1,2,3)
           for dataset in ("random","zero_charge","uniform","signed_charge")]
    cases += [(n,None,0,0,"random","false") for n in (4,5)]
    cases += [(n,kind,component,at,"random","false") for n in (2,3)
              for kind in ("finite","nan","inf","negative_inf") for component in range(4) for at in range(5)]
    cases += [(1,kind,component,4,"random","false")
              for kind in ("finite","nan","inf","negative_inf") for component in range(4)]
    cases += [(2,"many",component,0,"random","false") for component in (0,4)]
    cases += [(2,"finite",4,4,"random","false")]
    cases += [(2,"finite",component,4,dataset,"false") for component in range(4)
              for dataset in ("zero_charge","uniform")]
    cases += [(2,"finite",1,4,"signed_charge","false")]
    cases += [(2,None,0,0,"random","true")]
    results=[]

    def report(complete=False):
        data=dict(complete=complete,total=len(results),expected=len(cases),
                  passed=sum(r["passed"] for r in results),
                  source_sha256=hashlib.sha256(source.encode()).hexdigest(),results=results)
        (args.work_dir/"report.json").write_text(json.dumps(data,indent=2)+"\n")
        return data

    for n,kind,component,at,dataset,warm in cases:
        label=f"n{n}-{kind or 'control'}-c{component}-p{at}-{dataset}-{warm}"
        env=dict(clean_env,BENCH_PARAM_boxes1d=str(n),BENCH_PARAM_vector_add_warmup=warm,
                 LAVAMD_TEST_DATASET=dataset)
        if kind:env.update(LAVAMD_TEST_KIND=kind,LAVAMD_TEST_COMPONENT=str(component),LAVAMD_TEST_LOCATION=str(at))
        start=time.monotonic()
        try:
            p=subprocess.run([str(dst/"rodinia_lavamd")],env=env,capture_output=True,text=True,timeout=60)
            stdout,stderr,code=p.stdout,p.stderr,p.returncode
        except subprocess.TimeoutExpired as exc:
            stdout=(exc.stdout or b"").decode(errors="replace");stderr=(exc.stderr or b"").decode(errors="replace");code="timeout"
        elapsed=time.monotonic()-start
        (args.work_dir/(label+".stdout")).write_text(stdout);(args.work_dir/(label+".stderr")).write_text(stderr)
        particles=n*n*n*100
        errors=(12 if kind=="many" else 1)*(4 if component==4 else 1) if kind else 0
        passed=code==(1 if kind else 0) and elapsed<60
        passed &= f"LavaMD verification: {'FAIL' if kind else 'PASS'} ({particles} particles, {4*particles} components, {errors} mismatches)" in stderr
        passed &= stderr.count("MISMATCH ")==min(errors,10)
        passed &= ("LAVAMD_TEST_INJECT" in stderr)==bool(kind)
        if kind:
            passed &= "MISMATCH force[" in stderr
            if component!=4:passed &= f"].{('v','x','y','z')[component]}:" in stderr
        try:
            kernel,summary=[json.loads(line) for line in stdout.splitlines()]
            passed &= kernel["type"]=="kernel" and kernel["name"]=="kernel_gpu_cuda" and kernel["iter"]==0
            passed &= kernel["params"]==dict(boxes1d=n,particles_per_box=100,number_threads=128,verification="full_cpu_v1")
            passed &= all(math.isfinite(kernel[f]) and kernel[f]>=0 for f in ("time_ms","host_time_ms"))
            passed &= summary["type"]=="summary" and abs(summary["total_time_ms"]-kernel["time_ms"])<.000003
        except (ValueError,KeyError,TypeError):passed=False
        results.append(dict(label=label,n=n,kind=kind,component=component,location=at,dataset=dataset,warm=warm,
                            exit_code=code,seconds=elapsed,passed=bool(passed)))
        report()
        if not passed:print("FAIL:",label,"see",args.work_dir,flush=True)
    final=report(complete=True)
    print(f"{final['passed']}/{final['total']} LAVAMD cases passed")
    return 0 if final['passed']==len(cases) else 1


if __name__=="__main__":
    raise SystemExit(main())
