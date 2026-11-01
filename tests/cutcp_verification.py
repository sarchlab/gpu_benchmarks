#!/usr/bin/env python3
"""Verify every actual CUTCP lattice value, including device-tail corruption."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

HOOK = r'''
    if(const char* text=std::getenv("CUTCP_TEST_INDEX")) {
        size_t at=std::strtoull(text,nullptr,10),count=std::getenv("CUTCP_TEST_MANY")?12:1;
        const size_t total=size_t(nx)*ny*nz;
        if(count>total || at>total-count)std::abort();
        for(size_t v=at;v<at+count;++v) {
            size_t i=v%nx,j=(v/nx)%ny,k=v/(size_t(nx)*ny);
            size_t offset=(((k/8)*yRegionDim+j/8)*xRegionDim+i/8)*REGION_SIZE
                + ((k%8)*8+j%8)*8+i%8;
            float value;
            HIP_CHECK(hipMemcpy(&value,regionZeroCuda+offset,sizeof(float),hipMemcpyDeviceToHost));
            value=std::getenv("CUTCP_TEST_NAN")?NAN:value+1234.f;
            HIP_CHECK(hipMemcpy(regionZeroCuda+offset,&value,sizeof(float),hipMemcpyHostToDevice));
        }
        std::fprintf(stderr,"CUTCP_TEST_INJECT %zu %zu\n",at,count);
    }
'''


def instrument(source):
    anchor="    // ---- copy regions back + transpose into lattice (verbatim, untimed) ----"
    if source.count(anchor)!=1:raise ValueError("expected one measured-output download")
    return source.replace(anchor,HOOK+anchor)


def host_tests(root, work, compiler):
    relative=Path("tier2/parboil_cutcp/hip/cpu_reference.h")
    original=(root/relative).read_text()
    for label,fail_at in (("normal",None),("no-threads",1),("partial-threads",2)):
        path=work/label
        (path/relative.parent).mkdir(parents=True)
        (path/"tests").mkdir()
        header=original
        if fail_at:
            old="threads.emplace_back(part, worker);"
            if header.count(old)!=1:raise ValueError("thread fallback hook missing")
            header=header.replace(old,"{ if (worker == "+str(fail_at)+") throw std::runtime_error(\"test thread failure\"); threads.emplace_back(part, worker); }")
        (path/relative).write_text(header)
        shutil.copy2(root/"tests/cutcp_reference_test.cpp",path/"tests/cutcp_reference_test.cpp")
        binary=path/"reference"
        subprocess.run([compiler,"-O2","-std=c++14","-pthread","-ffp-contract=off","-Wall","-Wextra","-Werror",
                        "-fsanitize=address,undefined","-fno-sanitize-recover=all",
                        str(path/"tests/cutcp_reference_test.cpp"),"-o",str(binary)],check=True)
        subprocess.run([str(binary)],check=True,timeout=60)
        print("PASS host "+label,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source",type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument("--work-dir",type=Path,required=True,help="New, nonexistent directory")
    ap.add_argument("--host-only",action="store_true")
    ap.add_argument("--compiler",default="c++")
    ap.add_argument("--platform",choices=("cuda","rocm"))
    ap.add_argument("--compile-only",action="store_true")
    ap.add_argument("--sweep",action="store_true")
    args=ap.parse_args()
    if not args.host_only and not args.platform:ap.error("--platform required")
    args.source,args.work_dir=args.source.resolve(),args.work_dir.resolve()
    args.work_dir.mkdir(parents=True,exist_ok=False)
    if args.host_only:
        host_tests(args.source,args.work_dir,args.compiler)
        return 0
    relative=Path("tier2/parboil_cutcp/hip")
    src=args.source/relative
    source=(src/"parboil_cutcp.hip").read_text()
    inputs=[relative/n for n in ("parboil_cutcp.hip","cpu_reference.h","Makefile","params.json")]
    inputs += [Path("common_hip")/n for n in ("bench_timing.h","bench_warmup.h","hip_cuda_compat.h")]
    hashes={str(p):hashlib.sha256((args.source/p).read_bytes()).hexdigest() for p in inputs}
    binaries={}
    for variant in ("production","faults"):
        root=args.work_dir/variant;dst=root/relative;dst.mkdir(parents=True)
        shutil.copytree(args.source/"common_hip",root/"common_hip")
        for f in ("Makefile","params.json","cpu_reference.h"):shutil.copy2(src/f,dst/f)
        (dst/"parboil_cutcp.hip").write_text(source if variant=="production" else instrument(source))
        if variant=="faults":
            p=dst/"cpu_reference.h";s=p.read_text();anchor="    std::vector<float> result(count, 0.f);"
            if s.count(anchor)!=1:raise ValueError("reference allocation hook missing")
            s=s.replace(anchor,'    if(std::getenv("CUTCP_TEST_ALLOC_FAIL")) { std::fprintf(stderr,"CUTCP_TEST_ALLOC_FAIL\\n"); throw std::bad_alloc(); }\n'+anchor)
            p.write_text(s)
        with (root/"build.log").open("w") as log:
            subprocess.run(["make","PLATFORM="+args.platform],cwd=dst,stdout=log,stderr=subprocess.STDOUT,check=True)
        binaries[variant]=dst/"parboil_cutcp"
    if args.compile_only:
        print("CUTCP production and device-fault variants compiled; hardware checks not run")
        return 0
    env={k:v for k,v in os.environ.items() if not k.startswith(("BENCH_PARAM_","CUTCP_TEST_"))}
    results=[]
    def record():
        report=dict(source_sha256=hashes,binary_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in binaries.items()},
                    total=len(results),passed=sum(x["passed"] for x in results),cases=results)
        (args.work_dir/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    def run(label,n,d,h,c,warm,inject=None,many=False,nan=False,allocation=False,variant="faults"):
        e=dict(env,BENCH_PARAM_num_atoms=str(n),BENCH_PARAM_lattice_dim=str(d),BENCH_PARAM_grid_spacing=str(h),
               BENCH_PARAM_cutoff_radius=str(c),BENCH_PARAM_vector_add_warmup=warm)
        if inject is not None:e["CUTCP_TEST_INDEX"]=str(inject)
        if many:e["CUTCP_TEST_MANY"]="1"
        if nan:e["CUTCP_TEST_NAN"]="1"
        if allocation:e["CUTCP_TEST_ALLOC_FAIL"]="1"
        start=time.monotonic()
        try:
            p=subprocess.run([str(binaries[variant])],env=e,capture_output=True,text=True,timeout=60)
            stdout,stderr,code=p.stdout,p.stderr,p.returncode
        except subprocess.TimeoutExpired as exc:
            stdout=(exc.stdout or b"").decode(errors="replace");stderr=(exc.stderr or b"").decode(errors="replace");code="timeout"
        seconds=time.monotonic()-start
        (args.work_dir/(label+".stdout")).write_text(stdout);(args.work_dir/(label+".stderr")).write_text(stderr)
        bad=inject is not None or allocation
        passed=code==(1 if bad else 0) and seconds<60
        if allocation:passed &= "CUTCP_TEST_ALLOC_FAIL" in stderr and "FAIL: complete CPU reference unavailable:" in stderr
        elif inject is not None:
            count=12 if many else 1
            passed &= f"CUTCP_TEST_INJECT {inject} {count}" in stderr and f"FAIL: {count} mismatches in {d**3} points" in stderr
            passed &= stderr.count("Mismatch (")==min(10,count)
        else:passed &= f"PASS ({d**3} points verified)" in stderr
        if bad:passed &= "PASS (" not in stderr
        try:
            k,s=[json.loads(line) for line in stdout.splitlines()]
            params=k["params"]
            passed &= k["type"]=="kernel" and k["name"]=="cuda_cutoff_potential_lattice6overlap" and k["iter"]==0
            passed &= all(params[key]==val for key,val in dict(num_atoms=n,lattice_dim=d,grid_spacing=h,cutoff_radius=c,bin_depth=16,verification="full_cpu_v1").items())
            passed &= isinstance(params["nbrlist_len"],int) and params["nbrlist_len"]>0
            passed &= all(math.isfinite(k[f]) and k[f]>=0 for f in ("time_ms","host_time_ms"))
            passed &= s["type"]=="summary" and math.isfinite(s["total_time_ms"]) and abs(s["total_time_ms"]-k["time_ms"])<.000002
        except (ValueError,KeyError,TypeError):passed=False
        results.append(dict(label=label,atoms=n,dim=d,spacing=h,cutoff=c,warm=warm,inject=inject,many=many,nan=nan,
                            allocation=allocation,variant=variant,exit=code,seconds=seconds,passed=bool(passed)))
        record();print(f"{'PASS' if passed else 'FAIL'} {label}: {seconds:.3f}s",flush=True)
    for warm in ("false","true"):
        for n,d,h,c in ((8,9,.5,8),(8,16,1.,12),(5000,144,.5,12),(45000,256,.5,16)):
            tag=f"{n}-{d}-{h}-{c}-{warm}"
            run("healthy-"+tag,n,d,h,c,warm)
            run("tail-"+tag,n,d,h,c,warm,inject=d**3-1)
            run("nan-"+tag,n,d,h,c,warm,inject=d**3-1,nan=True)
        run("many-"+warm,5000,144,.5,12,warm,inject=144**3-12,many=True)
        run("unavailable-reference-"+warm,5000,144,.5,12,warm,allocation=True)
        # Cross padded region boundaries; faults must map to logical outputs.
        for index in (0,7,8,9*8,9**2*8):
            run(f"boundary-{index}-{warm}",8,9,.5,8,warm,inject=index)
    if args.sweep:
        m=json.loads((src/"params.json").read_text());p={x["name"]:x["values"] for x in m["non_scaling_params"]}
        for n,d,h,c in itertools.product(m["scaling_param"]["values"],p["lattice_dim"],p["grid_spacing"],p["cutoff_radius"]):
            run(f"production-{n}-{d}-{h}-{c}",n,d,h,c,"true",variant="production")
    record();passed=sum(x["passed"] for x in results)
    print(f"{passed}/{len(results)} CUTCP measured-result cases passed")
    return 0 if passed==len(results) else 1


if __name__=="__main__":
    raise SystemExit(main())
