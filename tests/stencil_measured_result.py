#!/usr/bin/env python3
"""Check ordinary Parboil stencil verification by corrupting its final GPU output."""
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
#include <cmath>
#include <limits>
static void stencil_test_fault(float* latest, int n) {
    const char* kind=std::getenv("STENCIL_TEST_KIND");
    if (!kind) return;
    const char* location=std::getenv("STENCIL_TEST_LOCATION");
    if (!location) std::abort();
    const int selection=std::atoi(location);
    if (selection<0 || selection>8) std::abort();
    const size_t first=((size_t)n+1)*n+1;
    const size_t last=(((size_t)n-2)*n+n-2)*n+n-2;
    const size_t positions[]={first,last,0,(size_t)n*n*n-1,
        ((size_t)n/2*n+n/2)*n, ((size_t)n/2*n+n/2)*n+n-1,
        ((size_t)n/2*n)*n+n/2, ((size_t)n/2*n+n-1)*n+n/2,
        ((size_t)n/2)*n+n/2};
    const bool many=!std::strcmp(kind,"many");
    const int count=many?12:1;
    for (int i=0;i<count;++i) {
        const size_t at=many?(size_t)i:positions[selection];
        float value=1000;
        if (!std::strcmp(kind,"nan")) value=std::numeric_limits<float>::quiet_NaN();
        else if (!std::strcmp(kind,"inf")) value=std::numeric_limits<float>::infinity();
        else if (!std::strcmp(kind,"negative_inf")) value=-std::numeric_limits<float>::infinity();
        else if (!std::strcmp(kind,"ulp")) {
            HIP_CHECK(hipMemcpy(&value,latest+at,sizeof(value),hipMemcpyDeviceToHost));
            value=std::nextafter(value,std::numeric_limits<float>::infinity());
        } else if (std::strcmp(kind,"finite") && !many) std::abort();
        HIP_CHECK(hipMemcpy(latest+at,&value,sizeof(value),hipMemcpyHostToDevice));
    }
    std::fprintf(stderr,"STENCIL_TEST_INJECT %zu %s %d\n",positions[selection],kind,count);
}
'''
ANCHOR = "    // Verify the final measured state after timing, with no extra GPU pass.\n"


def instrument(source):
    include = "#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)\n"
    for anchor in (include, ANCHOR):
        if source.count(anchor) != 1:
            raise ValueError("expected one production anchor: " + anchor)
    return source.replace(include, include+HOOK).replace(
        ANCHOR, "    stencil_test_fault(latest,grid_dim);\n"+ANCHOR)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source",type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument("--work-dir",type=Path,required=True,help="New, nonexistent output directory")
    ap.add_argument("--platform",choices=("cuda","rocm"),required=True)
    ap.add_argument("--compile-only",action="store_true")
    args=ap.parse_args()
    args.source,args.work_dir=args.source.resolve(),args.work_dir.resolve()
    try:args.work_dir.mkdir(parents=True,exist_ok=False)
    except FileExistsError:ap.error("--work-dir must name a new, nonexistent directory")
    src=args.source/"tier2/parboil_stencil/hip";dst=args.work_dir/"tier2/parboil_stencil/hip"
    dst.mkdir(parents=True)
    shutil.copytree(args.source/"common",args.work_dir/"common")
    for name in ("Makefile","params.json"):
        shutil.copy2(src/name,dst/name)
    for header in src.glob("*.h"):
        shutil.copy2(header,dst/header.name)
    source=(src/"parboil_stencil.hip").read_text()
    (dst/"parboil_stencil.hip").write_text(instrument(source))
    with (args.work_dir/"build.log").open("w") as log:
        subprocess.run(["make","PLATFORM="+args.platform],cwd=dst,stdout=log,stderr=subprocess.STDOUT,check=True)
    if args.compile_only:
        print("Stencil regression compiled; GPU checks not run")
        return 0
    clean_env={k:v for k,v in os.environ.items() if not k.startswith(("BENCH_PARAM_","STENCIL_TEST_"))}
    cases=[(n,t,None,0,"false") for n in (3,17,64) for t in (1,2,5,10,100)]
    cases += [(17,t,kind,at,"false") for t in (1,2,5,10,100)
              for kind in ("finite","nan","inf","negative_inf") for at in range(9)]
    cases += [(17,t,"ulp",at,"false") for t in (5,10) for at in range(2,9)]
    cases += [(17,t,"many",0,"false") for t in (5,10)]
    cases += [(64,10,None,0,"true")]
    results=[]

    def report(complete=False):
        data=dict(complete=complete,total=len(results),expected=len(cases),
                  passed=sum(r["passed"] for r in results),
                  source_sha256=hashlib.sha256(source.encode()).hexdigest(),results=results)
        (args.work_dir/"report.json").write_text(json.dumps(data,indent=2)+"\n")
        return data

    for n,t,kind,at,warm in cases:
        label=f"n{n}-t{t}-{kind or 'control'}-{at}-{warm}"
        env=dict(clean_env,BENCH_PARAM_grid_dim=str(n),BENCH_PARAM_num_timesteps=str(t),
                 BENCH_PARAM_vector_add_warmup=warm)
        if kind:env.update(STENCIL_TEST_KIND=kind,STENCIL_TEST_LOCATION=str(at))
        start=time.monotonic()
        try:
            p=subprocess.run([str(dst/"parboil_stencil")],env=env,capture_output=True,text=True,timeout=60)
            stdout,stderr,code=p.stdout,p.stderr,p.returncode
        except subprocess.TimeoutExpired as exc:
            stdout=(exc.stdout or b"").decode(errors="replace")
            stderr=(exc.stderr or b"").decode(errors="replace");code="timeout"
        elapsed=time.monotonic()-start
        (args.work_dir/(label+".stdout")).write_text(stdout)
        (args.work_dir/(label+".stderr")).write_text(stderr)
        errors=12 if kind=="many" else 1 if kind else 0
        passed=code==(1 if kind else 0) and elapsed<60
        passed &= f"Stencil verification: {'FAIL' if kind else 'PASS'} ({n**3} cells, {errors} mismatches)" in stderr
        passed &= stderr.count("MISMATCH stencil[")==min(errors,10)
        passed &= ("STENCIL_TEST_INJECT" in stderr)==bool(kind)
        try:
            records=[json.loads(line) for line in stdout.splitlines()]
            k,summary=records
            passed &= k["type"]=="kernel" and k["name"]=="stencil3d" and k["iter"]==0
            passed &= k["params"]==dict(grid_dim=n,num_timesteps=t,verification="full_cpu_v1")
            passed &= all(math.isfinite(k[f]) and k[f]>=0 for f in ("time_ms","host_time_ms"))
            passed &= summary["type"]=="summary" and abs(summary["total_time_ms"]-k["time_ms"])<.000002
        except (ValueError,KeyError,TypeError):passed=False
        results.append(dict(label=label,n=n,steps=t,kind=kind,location=at,warm=warm,
                            exit_code=code,seconds=elapsed,passed=bool(passed)))
        report()
        if not passed:print("FAIL:",label,"see",args.work_dir,flush=True)
    final=report(complete=True)
    print(f"{final['passed']}/{final['total']} stencil cases passed")
    return 0 if final['passed']==len(cases) else 1


if __name__=="__main__":
    raise SystemExit(main())
