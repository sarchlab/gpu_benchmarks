#!/usr/bin/env python3
"""Corrupt actual measured SRAD fields and verify ordinary-run failure."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

HOOK = r'''
#include <limits>
static void srad_test_fault(float* J,float* C,float* E,float* W,float* S,float* N,
                            int n,int iteration,int iterations) {
    const char* kind=std::getenv("SRAD_TEST_KIND");if(!kind)return;
    const bool early=!std::strcmp(kind,"early");
    if(iteration!=(early?0:iterations-1))return;
    if(early && iterations<2)std::abort();
    const char* text=std::getenv("SRAD_TEST_STAGE");
    const char* location=std::getenv("SRAD_TEST_LOCATION");
    if(!text || !location)std::abort();
    const int stage=std::atoi(text),selection=std::atoi(location);
    if(stage<0 || stage>6 || selection<0 || selection>5 || (early && stage!=0))std::abort();
    float* outputs[]={J,C,E,W,S,N};const size_t elements=(size_t)n*n;
    const size_t positions[]={0,std::min(size_t(31),elements-1),
        std::min(size_t(32),elements-1),(size_t)(n/2)*n+n/2,elements-1,
        (size_t)(n-2)*n+n-2};
    const bool many=!std::strcmp(kind,"many"),ulp=!std::strcmp(kind,"ulp");
    const size_t count=many?12:1;if(count>elements)std::abort();
    for(int s=0;s<6;++s) {
        if(stage!=6 && stage!=s)continue;
        for(size_t i=0;i<count;++i) {
            const size_t at=many?i:positions[selection];float value;
            if(!std::strcmp(kind,"nan"))value=std::numeric_limits<float>::quiet_NaN();
            else if(!std::strcmp(kind,"inf"))value=std::numeric_limits<float>::infinity();
            else if(!std::strcmp(kind,"negative_inf"))value=-std::numeric_limits<float>::infinity();
            else if(!std::strcmp(kind,"below_range"))value=std::nextafter(0.0f,-1.0f);
            else if(!std::strcmp(kind,"above_range"))value=std::nextafter(1.0f,2.0f);
            else if(!std::strcmp(kind,"finite") || many || ulp || early) {
                HIP_CHECK(hipMemcpy(&value,outputs[s]+at,sizeof(value),hipMemcpyDeviceToHost));
                if(ulp)value=std::nextafter(value,std::numeric_limits<float>::infinity());
                else if(s==1)value=value>=.5f?value-.25f:value+.25f;
                else value+=std::max(1.0f,std::fabs(value)*.25f);
            } else std::abort();
            HIP_CHECK(hipMemcpy(outputs[s]+at,&value,sizeof(value),hipMemcpyHostToDevice));
        }
    }
    std::fprintf(stderr,"SRAD_TEST_INJECT stage=%d kind=%s count=%zu iteration=%d\n",stage,kind,count,iteration);
}
'''
ANCHOR = "            HIP_CHECK(hipMemcpy(iter == niter - 1 ? h_I : h_J, J_cuda, size_I * sizeof(float), hipMemcpyDeviceToHost));"
CALL = "            srad_test_fault(J_cuda,C_cuda,E_C,W_C,S_C,N_C,rows,iter,niter);\n"


def instrument(source):
    include = "#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)\n"
    for anchor in (include, ANCHOR):
        if source.count(anchor) != 1:
            raise ValueError("expected one production anchor: " + anchor)
    return source.replace(include, include + HOOK).replace(ANCHOR, CALL + ANCHOR)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--work-dir", type=Path, required=True, help="New, nonexistent output directory")
    ap.add_argument("--platform", choices=("cuda", "rocm"), required=True)
    ap.add_argument("--compile-only", action="store_true")
    args = ap.parse_args()
    args.source, args.work_dir = args.source.resolve(), args.work_dir.resolve()
    try:
        args.work_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        ap.error("--work-dir must name a new, nonexistent directory")
    src = args.source / "tier2/rodinia_srad/hip"
    dst = args.work_dir / "tier2/rodinia_srad/hip"
    dst.mkdir(parents=True)
    shutil.copytree(args.source / "common_hip", args.work_dir / "common_hip")
    for name in ("Makefile", "params.json"):
        shutil.copy2(src / name, dst / name)
    for header in src.glob("*.h"):
        shutil.copy2(header, dst / header.name)
    source = (src / "rodinia_srad.hip").read_text()
    (dst / "rodinia_srad.hip").write_text(instrument(source))
    with (args.work_dir / "build.log").open("w") as log:
        subprocess.run(["make", "PLATFORM=" + args.platform], cwd=dst,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    if args.compile_only:
        print("SRAD regression compiled; GPU checks not run")
        return 0
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith(("BENCH_PARAM_", "SRAD_TEST_"))}
    controls = ((16,1),(16,2),(16,10),(17,2),(32,10),(48,50),(64,1),(64,2),
                (64,10),(64,50),(64,100),(64,200),(128,10),(496,2),(512,2),(528,2))
    cases = [(n,t,None,0,0,.5,"false") for n,t in controls]
    cases += [(n,t,None,0,0,lam,"false") for n in (16,64,512) for t in (1,10) for lam in (.25,.75)]
    cases += [(n,10,kind,stage,at,.5,"false") for n in (16,64)
              for kind in ("finite","nan","inf","negative_inf") for stage in range(6) for at in range(6)]
    cases += [(64,10,"many",stage,0,.5,"false") for stage in range(7)]
    cases += [(n,10,"ulp",stage,at,.5,"false") for n in (16,64) for stage in range(2,6) for at in (0,4)]
    cases += [(64,10,kind,1,at,.5,"false") for kind in ("below_range","above_range") for at in (0,4)]
    cases += [(n,2,"early",0,at,.5,"false") for n in (16,64,512) for at in (0,3,4)]
    cases += [(n,2,"finite",stage,5,.5,"false") for n in (512,528) for stage in range(6)]
    cases += [(64,10,None,0,0,.5,"true")]
    results = []

    def report(complete=False):
        data = dict(complete=complete,total=len(results),expected=len(cases),
                    passed=sum(r["passed"] for r in results),
                    source_sha256=hashlib.sha256(source.encode()).hexdigest(),results=results)
        (args.work_dir / "report.json").write_text(json.dumps(data,indent=2) + "\n")
        return data

    for requested,t,kind,stage,at,lam,warm in cases:
        n = (requested+15)//16*16
        label = f"n{requested}-t{t}-{kind or 'control'}-s{stage}-p{at}-l{lam}-{warm}"
        env = dict(clean_env,BENCH_PARAM_image_size=str(requested),BENCH_PARAM_num_iterations=str(t),
                   BENCH_PARAM_lambda=str(lam),BENCH_PARAM_vector_add_warmup=warm)
        if kind:
            env.update(SRAD_TEST_KIND=kind,SRAD_TEST_STAGE=str(stage),SRAD_TEST_LOCATION=str(at))
        start = time.monotonic()
        try:
            p = subprocess.run([str(dst / "rodinia_srad")],env=env,capture_output=True,text=True,timeout=60)
            stdout,stderr,code = p.stdout,p.stderr,p.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or b"").decode(errors="replace")
            stderr = (exc.stderr or b"").decode(errors="replace")
            code = "timeout"
        elapsed = time.monotonic()-start
        (args.work_dir / (label+".stdout")).write_text(stdout)
        (args.work_dir / (label+".stderr")).write_text(stderr)
        errors = (12 if kind=="many" else 1)*(6 if stage==6 else 1) if kind else 0
        passed = code==(1 if kind else 0) and elapsed<60
        match = re.search(r"SRAD verification: (PASS|FAIL) \((\d+) values, (\d+) comparisons, (\d+) mismatches\)",stderr)
        if match:
            status,values,comparisons,mismatches = match.groups()
            mismatches = int(mismatches)
            passed &= status==("FAIL" if kind else "PASS") and int(values)==6*n*n and int(comparisons)==7*n*n
            passed &= mismatches>0 if kind=="early" else mismatches==errors
            passed &= stderr.count("MISMATCH ")==min(mismatches,10)
        else:
            passed = False
        passed &= ("SRAD_TEST_INJECT" in stderr)==bool(kind)
        if kind and stage!=6:
            names = ("J","C","E","W","S","N")
            passed &= f"MISMATCH {names[stage]}[" in stderr
            passed &= all(f"MISMATCH {name}[" not in stderr for i,name in enumerate(names) if i!=stage)
        try:
            first,second,summary = [json.loads(line) for line in stdout.splitlines()]
            kernels = (first,second)
            for kernel,name in zip(kernels,("srad_cuda_1","srad_cuda_2")):
                passed &= kernel["type"]=="kernel" and kernel["name"]==name and kernel["iter"]==0
                passed &= kernel["params"]=={
                    "image_size":n,"num_iterations":t,"block_size":16,"lambda":lam,
                    "verification":"full_cpu_trajectory_and_step_v1"}
                passed &= all(math.isfinite(kernel[f]) and kernel[f]>=0 for f in ("time_ms","host_time_ms"))
            passed &= summary["type"]=="summary" and abs(summary["total_time_ms"]-sum(k["time_ms"] for k in kernels))<.000004
        except (ValueError,KeyError,TypeError):
            passed = False
        results.append(dict(label=label,requested=requested,n=n,iterations=t,kind=kind,stage=stage,location=at,
                            lambda_value=lam,warm=warm,exit_code=code,seconds=elapsed,passed=bool(passed)))
        report()
        if not passed:
            print("FAIL:",label,"see",args.work_dir,flush=True)
    final = report(complete=True)
    print(f"{final['passed']}/{final['total']} SRAD cases passed")
    return 0 if final['passed']==len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
