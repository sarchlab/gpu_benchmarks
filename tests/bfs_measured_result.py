#!/usr/bin/env python3
"""Check measured BFS distances with structured graphs and device-output faults."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

GRAPHS = ("isolated", "chain", "reverse", "diamond", "cycle", "duplicates", "shortcuts", "disconnected")
HOOK = r'''
#include <limits>
#include <vector>
static void bfs_test_graph(int n, int* row, int** col, int* count) {
    const char* name = std::getenv("BFS_TEST_GRAPH");
    if (!name) return;
    if (n < 4) std::abort();
    std::vector<std::vector<int>> edges(n);
    if (!std::strcmp(name,"isolated")) {}
    else if (!std::strcmp(name,"chain") || !std::strcmp(name,"shortcuts")) {
        for (int u=0;u+1<n;++u) edges[u].push_back(u+1);
        if (!std::strcmp(name,"shortcuts")) { edges[0].push_back(n-1); edges[0].push_back(n/2); }
    } else if (!std::strcmp(name,"reverse")) {
        for (int u=1;u<n;++u) edges[u].push_back(u-1);
    } else if (!std::strcmp(name,"diamond")) {
        edges[0]={1,2};edges[1]={3};edges[2]={3};
    } else if (!std::strcmp(name,"cycle") || !std::strcmp(name,"duplicates")) {
        for (int u=0;u<n;++u) {
            edges[u].push_back((u+1)%n);
            if (!std::strcmp(name,"duplicates")) {
                edges[u].push_back((u+1)%n);edges[u].push_back(u);edges[u].push_back((u+1)%n);
            }
        }
    } else if (!std::strcmp(name,"disconnected")) {
        for (int u=0;u<n;++u) edges[u].push_back(u+1==n/2?0:u+1==n?n/2:u+1);
    } else std::abort();
    std::vector<int> values;
    row[0]=0;
    for (int u=0;u<n;++u) {
        values.insert(values.end(),edges[u].begin(),edges[u].end());
        row[u+1]=static_cast<int>(values.size());
    }
    std::free(*col);
    // Keep a valid allocation for zero-edge fixtures and zero-byte GPU copies.
    *col=static_cast<int*>(std::malloc((values.empty()?1:values.size())*sizeof(int)));
    if (!*col) std::abort();
    for (size_t i=0;i<values.size();++i) (*col)[i]=values[i];
    *count=static_cast<int>(values.size());
    std::fprintf(stderr,"BFS_TEST_GRAPH %s %d %d\n",name,n,*count);
}
static void bfs_test_fault(int* cost, int n) {
    const char* kind=std::getenv("BFS_TEST_KIND");
    if (!kind) return;
    const char* location=std::getenv("BFS_TEST_LOCATION");
    if (!location) std::abort();
    const int selection=std::atoi(location);
    if (selection<0 || selection>2) std::abort();
    const int at=selection==0?0:selection==1?n/2:n-1;
    int value;
    HIP_CHECK(hipMemcpy(&value,cost+at,sizeof(value),hipMemcpyDeviceToHost));
    if (!std::strcmp(kind,"plus1")) ++value;
    else if (!std::strcmp(kind,"minus1")) --value;
    else if (!std::strcmp(kind,"far")) value=std::numeric_limits<int>::max();
    else std::abort();
    HIP_CHECK(hipMemcpy(cost+at,&value,sizeof(value),hipMemcpyHostToDevice));
    std::fprintf(stderr,"BFS_TEST_INJECT %d %s\n",at,kind);
}
'''


def instrument(source):
    anchors = ("#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)\n",
               "    // Host arrays for BFS state\n",
               "    // Verify every distance produced by the measured traversal, after timing.\n")
    for anchor in anchors:
        if source.count(anchor) != 1:
            raise ValueError("expected one production anchor: " + anchor)
    return source.replace(anchors[0], anchors[0]+HOOK).replace(
        anchors[1], "    bfs_test_graph(num_nodes,h_row_offsets,&h_col_indices,&num_edges);\n"+anchors[1]).replace(
        anchors[2], "    bfs_test_fault(d_cost,num_nodes);\n"+anchors[2])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source",type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument("--work-dir",type=Path,required=True,help="New, nonexistent output directory")
    ap.add_argument("--platform",choices=("cuda","rocm"),required=True)
    ap.add_argument("--compile-only",action="store_true")
    args=ap.parse_args();args.source,args.work_dir=args.source.resolve(),args.work_dir.resolve()
    try:args.work_dir.mkdir(parents=True,exist_ok=False)
    except FileExistsError:ap.error("--work-dir must name a new, nonexistent directory")
    src=args.source/"tier2/rodinia_bfs/hip";dst=args.work_dir/"tier2/rodinia_bfs/hip";dst.mkdir(parents=True)
    shutil.copytree(args.source/"common_hip",args.work_dir/"common_hip")
    for name in ("Makefile","params.json"):
        shutil.copy2(src/name,dst/name)
    for header in src.glob("*.h"):
        shutil.copy2(header,dst/header.name)
    source=(src/"rodinia_bfs.hip").read_text()
    (dst/"rodinia_bfs.hip").write_text(instrument(source))
    with (args.work_dir/"build.log").open("w") as log:
        subprocess.run(["make","PLATFORM="+args.platform],cwd=dst,stdout=log,stderr=subprocess.STDOUT,check=True)
    if args.compile_only:
        print("BFS regression compiled; GPU checks not run")
        return 0
    clean_env={k:v for k,v in os.environ.items() if not k.startswith(("BENCH_PARAM_","BFS_TEST_"))}
    results=[]

    def report(complete=False):
        data=dict(complete=complete,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  cases=results,total=len(results),passed=sum(r["passed"] for r in results))
        (args.work_dir/"report.json").write_text(json.dumps(data,indent=2)+"\n")
        return data

    def run(n,degree=20,graph="random",kind=None,location=0,warm="false"):
        label=f"n{n}-d{degree}-{graph}-{kind}-{location}-warm{warm}"
        env=dict(clean_env,BENCH_PARAM_num_nodes=str(n),BENCH_PARAM_edges_per_node=str(degree),
                 BENCH_PARAM_block_size="256",BENCH_PARAM_vector_add_warmup=warm)
        if graph!="random":env["BFS_TEST_GRAPH"]=graph
        if kind:env.update(BFS_TEST_KIND=kind,BFS_TEST_LOCATION=str(location))
        begin=time.monotonic()
        try:
            proc=subprocess.run([str(dst/"rodinia_bfs")],cwd=dst,env=env,text=True,capture_output=True,timeout=60)
        except subprocess.TimeoutExpired as exc:
            for suffix,value in (("stdout",exc.stdout),("stderr",exc.stderr)):
                if isinstance(value,bytes):value=value.decode(errors="replace")
                (args.work_dir/(label+"."+suffix)).write_text(value or "")
            results.append(dict(case=label,passed=False,errors=["timeout"],exit_code=None,seconds=time.monotonic()-begin));report()
            raise SystemExit("BFS test timed out; partial evidence retained in "+str(args.work_dir))
        for suffix,value in (("stdout",proc.stdout),("stderr",proc.stderr)):
            (args.work_dir/(label+"."+suffix)).write_text(value)
        errors=[];expected_exit=1 if kind else 0
        if proc.returncode!=expected_exit:errors.append(f"exit {proc.returncode}, expected {expected_exit}")
        diagnostic=f"BFS verification: {'FAIL' if kind else 'PASS'} ({n} distances, {1 if kind else 0} mismatches)"
        if diagnostic not in proc.stderr:errors.append("missing exact verification diagnostic")
        if graph!="random" and f"BFS_TEST_GRAPH {graph} {n} " not in proc.stderr:errors.append("graph fixture not installed")
        if kind:
            at=0 if location==0 else n//2 if location==1 else n-1
            if f"BFS_TEST_INJECT {at} {kind}" not in proc.stderr or f"MISMATCH cost[{at}]" not in proc.stderr:
                errors.append("missing injected-output mismatch")
        try:
            records=[json.loads(line) for line in proc.stdout.splitlines()]
            if len(records)!=3 or records[-1].get("type")!="summary":raise ValueError("expected two kernels and summary")
            params=dict(num_nodes=n,edges_per_node=degree,block_size=256,verification="full_cpu_v1",
                        discovery_policy="level_visited_atomic_v1")
            work=records[0]["params"]
            for key in ("bfs_levels","frontier_edges","discovery_atomics","promotion_atomics"):
                if not isinstance(work.get(key),int) or work[key]<(1 if key=="bfs_levels" else 0):
                    raise ValueError("invalid logical work descriptor")
                params[key]=work[key]
            for kernel,name in zip(records[:2],("bfs_kernel","bfs_update")):
                if kernel.get("type")!="kernel" or kernel.get("name")!=name or kernel.get("iter")!=0 or kernel.get("params")!=params:
                    raise ValueError("incorrect kernel/workload/verification identity")
                if any(not math.isfinite(kernel[k]) or kernel[k]<0 for k in ("time_ms","host_time_ms")):raise ValueError("invalid timing")
            total=records[-1]["total_time_ms"]
            if not math.isfinite(total) or not math.isclose(total,sum(k["time_ms"] for k in records[:2]),rel_tol=1e-5,abs_tol=2e-6):raise ValueError("invalid timing sum")
        except (ValueError,KeyError,TypeError) as exc:errors.append("invalid output: "+str(exc))
        results.append(dict(case=label,passed=not errors,errors=errors,exit_code=proc.returncode,seconds=time.monotonic()-begin));report()
        print(("FAIL " if errors else "PASS ")+label+(": "+"; ".join(errors) if errors else ""),flush=True)

    for n in (1,2,3,7,8,9,17,33):run(n)
    for n,degree,graph in [(1024,d,"random") for d in (5,20,80)]+[(9,20,g) for g in GRAPHS]:
        run(n,degree,graph)
        for kind in ("plus1","minus1","far"):
            for location in (0,1,2):run(n,degree,graph,kind,location)
    run(65536)
    run(1024,warm="true")
    final=report(complete=True)
    print(f"{final['passed']}/{final['total']} BFS cases passed")
    return 0 if final['passed']==final['total']==120 else 1


if __name__=="__main__":
    raise SystemExit(main())
