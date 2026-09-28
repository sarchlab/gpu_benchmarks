#!/usr/bin/env python3
"""Check complete measured CC labels, capped trajectories and final buffers."""
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
#include <cstring>
static void cc_test_fixture(int n, std::vector<int>& rows, std::vector<int>& cols) {
    if (n!=9) std::abort();
    // Path0--1--2--3--4--5, pair6--7, isolated8.
    rows={0,1,3,5,7,9,10,11,12,12};
    cols={1,0,2,1,3,2,4,3,5,4,7,6};
}
static void cc_test_fault(int* labels, int n) {
    const char* kind=std::getenv("CC_TEST_KIND");
    if (!kind) return;
    const char* index=std::getenv("CC_TEST_INDEX");
    if (!index) std::abort();
    int at=std::atoi(index), count=!std::strcmp(kind,"many")?12:1;
    if(at<0 || at>n-count) std::abort();
    for(int i=at;i<at+count;++i) {
        int value;
        HIP_CHECK(hipMemcpy(&value,labels+i,sizeof(value),hipMemcpyDeviceToHost));
        value+=1;
        HIP_CHECK(hipMemcpy(labels+i,&value,sizeof(value),hipMemcpyHostToDevice));
    }
    std::fprintf(stderr,"CC_TEST_INJECT %d %d\n",at,count);
}
'''


def instrument(source):
    anchors = ["#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)\n",
               "    generate_csr_graph(N, avg_degree, 42, row_offsets, col_indices);",
               "        std::vector<int> h_gpu_labels(N);",
               "    if (cur != d_labels) {"]
    for anchor in anchors:
        if source.count(anchor) != 1:
            raise ValueError("expected one production anchor: " + anchor)
    source = source.replace(anchors[0], anchors[0] + HOOK)
    source = source.replace(anchors[1], '    if (std::getenv("CC_TEST_FIXTURE")) cc_test_fixture(N,row_offsets,col_indices);\n    else\n'+anchors[1])
    source = source.replace(anchors[2], '        cc_test_fault(d_labels,N);\n'+anchors[2])
    # Test-only omission of the ordinary odd-step final-buffer copy.
    return source.replace(anchors[3], '    if (cur != d_labels && !std::getenv("CC_TEST_OLD_BUFFER")) {')


def host_reference(source, work, compiler):
    start=source.index("static void cc_cpu(")
    end=source.index("// ---------------------------------------------------------------------------",start)
    program="#include <vector>\n"+source[start:end]+r'''
int main() {
    const std::vector<int> rows={0,1,3,5,7,9,10,11,12,12};
    const std::vector<int> cols={1,0,2,1,3,2,4,3,5,4,7,6};
    const int expected[7][9]={
      {0,1,2,3,4,5,6,7,8},{0,0,1,2,3,4,6,6,8},
      {0,0,0,1,2,3,6,6,8},{0,0,0,0,1,2,6,6,8},
      {0,0,0,0,0,1,6,6,8},{0,0,0,0,0,0,6,6,8},
      {0,0,0,0,0,0,6,6,8}};
    for(int steps=0;steps<=7;++steps) {
      std::vector<int> actual;
      cc_cpu(9,steps,rows,cols,actual);
      if(actual.size()!=9) return 1;
      for(int i=0;i<9;++i) if(actual[i]!=expected[steps<6?steps:6][i]) return 2;
    }
    std::vector<int> actual;
    cc_cpu(1,100,std::vector<int>{0,0},std::vector<int>{},actual);
    return actual==std::vector<int>{0}?0:3;
}
'''
    path=work/"reference.cpp";path.write_text(program)
    binary=work/"reference"
    subprocess.run([compiler,"-O2","-std=c++14","-Wall","-Wextra","-Werror",
                    "-fsanitize=address,undefined","-fno-sanitize-recover=all",str(path),"-o",str(binary)],check=True)
    subprocess.run([str(binary)],check=True,timeout=20)
    print("PASS: production CPU Jacobi labels, eight exact trajectories and isolated vertex",flush=True)


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
    if not args.host_only and not args.platform: ap.error("--platform is required for GPU builds")
    args.source,args.work_dir=args.source.resolve(),args.work_dir.resolve()
    args.work_dir.mkdir(parents=True,exist_ok=False)
    relative=Path("tier2/graph_cc");src=args.source/relative
    source=(src/"graph_cc.hip").read_text()
    if args.host_only:
        host_reference(source,args.work_dir,args.compiler)
        return 0
    inputs=[relative/name for name in ("graph_cc.hip","Makefile","params.json")]
    inputs += [Path("common")/name for name in ("bench_timing.h","bench_warmup.h","hip_cuda_compat.h")]
    hashes={str(p):hashlib.sha256((args.source/p).read_bytes()).hexdigest() for p in inputs}
    binaries={}
    for variant in ("production","faults"):
        root=args.work_dir/variant;dst=root/relative;dst.mkdir(parents=True)
        shutil.copytree(args.source/"common",root/"common")
        for name in ("Makefile","params.json"): shutil.copy2(src/name,dst/name)
        (dst/"graph_cc.hip").write_text(source if variant=="production" else instrument(source))
        with (root/"build.log").open("w") as log:
            subprocess.run(["make","PLATFORM="+args.platform],cwd=dst,check=True,stdout=log,stderr=subprocess.STDOUT)
        binaries[variant]=dst/"graph_cc"
    if args.compile_only:
        print("Production and measured-label fault binaries compiled; GPU tests not run")
        return 0
    clean_env={k:v for k,v in os.environ.items() if not k.startswith(("BENCH_PARAM_","CC_TEST_"))}
    # variant,n,degree,cap,warm,fixture,kind,index,expected errors,expected iterations
    cases=[("production",n,4,100,"false",False,None,0,0,None) for n in (1,2,9,1023,1024,1025,2049)]
    cases += [("faults",9,4,t,"false",True,None,0,0,min(t,6)) for t in (1,2,3,4,5,6,7,100)]
    cases += [("faults",2049,16,100,"false",False,"one",i,1,None) for i in (0,1023,1024,1536,2048)]
    cases += [("faults",2049,16,100,"false",False,"many",1024,12,None)]
    cases += [("faults",9,4,t,"false",True,"old",0,errors,t) for t,errors in ((1,6),(3,3),(5,1))]
    cases += [("faults",9,4,5,"true",True,None,0,0,5),
              ("faults",9,4,6,"true",True,None,0,0,6),
              ("faults",9,4,100,"true",True,None,0,0,6),
              ("faults",2049,16,100,"true",False,"one",2048,1,None)]
    if args.sweep:
        manifest=json.loads((src/"params.json").read_text())
        degrees=next(x["values"] for x in manifest["non_scaling_params"] if x["name"]=="avg_degree")
        cases += [("production",n,d,100,"true",False,None,0,0,None)
                  for n in manifest["scaling_param"]["values"] for d in degrees]
    results=[]

    def report(complete=False):
        data=dict(complete=complete,source_sha256=hashes,
                  binary_sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in binaries.items()},
                  expected=len(cases),total=len(results),passed=sum(x["passed"] for x in results),cases=results)
        (args.work_dir/"report.json").write_text(json.dumps(data,indent=2)+"\n")
        return data

    for variant,n,degree,cap,warm,fixture,kind,index,errors,iters in cases:
        label=f"{variant}-{n}-{degree}-cap{cap}-{warm}-fixture{fixture}-{kind or 'control'}-{index}"
        env=dict(clean_env,BENCH_PARAM_num_vertices=str(n),BENCH_PARAM_avg_degree=str(degree),
                 BENCH_PARAM_max_iterations=str(cap),BENCH_PARAM_block_size="256",BENCH_PARAM_vector_add_warmup=warm)
        if fixture: env['CC_TEST_FIXTURE']='1'
        if kind=='old': env['CC_TEST_OLD_BUFFER']='1'
        elif kind: env.update(CC_TEST_KIND=kind,CC_TEST_INDEX=str(index))
        begin=time.monotonic()
        try:
            p=subprocess.run([str(binaries[variant])],env=env,capture_output=True,text=True,timeout=60)
            stdout,stderr,code=p.stdout,p.stderr,p.returncode
        except subprocess.TimeoutExpired as exc:
            stdout=(exc.stdout or b'').decode(errors='replace');stderr=(exc.stderr or b'').decode(errors='replace');code='timeout'
        seconds=time.monotonic()-begin
        (args.work_dir/(label+'.stdout')).write_text(stdout);(args.work_dir/(label+'.stderr')).write_text(stderr)
        passed=code==(1 if errors else 0) and seconds<60
        passed &= f"CC verification: {'FAIL' if errors else 'PASS'} ({n} vertices, {errors} mismatches)" in stderr
        passed &= stderr.count('Mismatch at vertex ')==min(errors,10)
        if kind in ('one','many'):
            passed &= f'CC_TEST_INJECT {index} {errors}' in stderr
            passed &= f'Mismatch at vertex {index}:' in stderr
        if iters is not None: passed &= f'({iters} CC iterations)' in stderr
        try:
            k,s=[json.loads(line) for line in stdout.splitlines()]
            passed &= k['type']=='kernel' and k['name']=='cc_propagate_kernel' and k['iter']==0
            performed=k['params']['iterations_performed']
            converged=k['params']['converged']
            passed &= type(performed) is int and 1<=performed<=cap
            passed &= type(converged) is bool and converged==(not fixture or cap>=6)
            if iters is not None: passed &= performed==iters
            passed &= f'({performed} CC iterations) '+('converged' if converged else 'iteration cap reached') in stderr
            passed &= k['params']==dict(num_vertices=n,avg_degree=degree,max_iterations=cap,block_size=256,
                                       verification='full_cpu_v1',iterations_performed=performed,converged=converged)
            passed &= all(math.isfinite(k[f]) and k[f]>=0 for f in ('time_ms','host_time_ms'))
            passed &= s['type']=='summary' and math.isfinite(s['total_time_ms']) and abs(s['total_time_ms']-k['time_ms'])<0.000002
        except (ValueError,KeyError,TypeError): passed=False
        results.append(dict(label=label,variant=variant,num_vertices=n,avg_degree=degree,cap=cap,warm=warm,
                            fixture=fixture,kind=kind,index=index,exit_code=code,seconds=seconds,passed=bool(passed)))
        report();print(f"{'PASS' if passed else 'FAIL'} {label}: {seconds:.3f}s",flush=True)
    final=report(complete=True)
    print(f"{final['passed']}/{final['total']} CC cases passed")
    return 0 if final['passed']==len(cases) else 1


if __name__=='__main__':
    raise SystemExit(main())
