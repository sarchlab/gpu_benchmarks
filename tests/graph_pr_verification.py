#!/usr/bin/env python3
"""Check all measured PageRank values, balanced faults and final buffers."""
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
static void pr_test_fixture(int n, std::vector<int>& columns,
                            std::vector<int>& rows, std::vector<int>& degrees) {
    if(n!=4) std::abort();
    // 0->1,0->2,1->2,2->0,3->2. No dangling vertex, but vertex3 has no incoming edge.
    columns={0,1,2,5,5}; rows={2,0,0,1,3}; degrees={2,1,1,1};
}
static void pr_test_fault(float* output, int n) {
    const char* kind=std::getenv("PR_TEST_KIND");
    if(!kind) return;
    const char* index=std::getenv("PR_TEST_INDEX");
    if(!index) std::abort();
    int at=std::atoi(index);
    bool balanced=!std::strcmp(kind,"balanced"), many=!std::strcmp(kind,"many");
    int count=many?12:balanced?2:1;
    if(at<0 || at>=n || (many && at>n-count)) std::abort();
    int other=at==n-1?1024:n-1;
    if(balanced && (other<0 || other>=n || other==at)) std::abort();
    for(int j=0;j<count;++j) {
        int pos=balanced?(j==0?at:other):at+j;
        float value;
        HIP_CHECK(hipMemcpy(&value,output+pos,sizeof(value),hipMemcpyDeviceToHost));
        if(balanced || many) value+=(j%2==0?0.125f:-0.125f);
        else if(!std::strcmp(kind,"nan")) value=std::numeric_limits<float>::quiet_NaN();
        else if(!std::strcmp(kind,"pinf")) value=std::numeric_limits<float>::infinity();
        else if(!std::strcmp(kind,"ninf")) value=-std::numeric_limits<float>::infinity();
        else std::abort();
        HIP_CHECK(hipMemcpy(output+pos,&value,sizeof(value),hipMemcpyHostToDevice));
    }
    std::fprintf(stderr,"PR_TEST_INJECT %d %s %d\n",at,kind,count);
}
'''


def instrument(source):
    anchors = ["#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)\n",
               "    generate_csc_graph(N, avg_deg, 42, col_offsets, row_indices, out_degree);",
               "        HIP_CHECK(hipMemcpy(h_gpu_rank.data(), d_rank_a, N * sizeof(float), hipMemcpyDeviceToHost));",
               "        if (cur != d_rank_a) {"]
    for anchor in anchors:
        if source.count(anchor) != 1:
            raise ValueError("expected one production anchor: " + anchor)
    source = source.replace(anchors[0], anchors[0] + HOOK)
    source = source.replace(anchors[1], '    if(std::getenv("PR_TEST_FIXTURE")) pr_test_fixture(N,col_offsets,row_indices,out_degree);\n    else\n' + anchors[1])
    source = source.replace(anchors[2], "        pr_test_fault(d_rank_a,N);\n" + anchors[2])
    return source.replace(anchors[3], '        if (cur != d_rank_a && !std::getenv("PR_TEST_OLD_BUFFER")) {')


def host_reference(source, work, compiler):
    start = source.index("static void pagerank_cpu(")
    end = source.index("// ---------------------------------------------------------------------------", start)
    program = "#include <vector>\n" + source[start:end] + r'''
int main() {
    const std::vector<int> columns={0,1,2,5,5}, rows={2,0,0,1,3}, degrees={2,1,1,1};
    // Independently evaluated dyadic recurrence with damping1/2. This checks
    // edge direction, out-degree division, the base term and odd/even state.
    const float expected[6][4]={
      {0.25f,0.25f,0.25f,0.25f},
      {0.25f,0.1875f,0.4375f,0.125f},
      {0.34375f,0.1875f,0.34375f,0.125f},
      {0.296875f,0.2109375f,0.3671875f,0.125f},
      {0.30859375f,0.19921875f,0.3671875f,0.125f},
      {0.30859375f,0.2021484375f,0.3642578125f,0.125f}};
    for(int steps=0;steps<6;++steps) {
      std::vector<float> actual;
      pagerank_cpu(4,steps,0.5f,columns,rows,degrees,actual);
      if(actual.size()!=4) return 1;
      float sum=0;
      for(int i=0;i<4;++i){if(actual[i]!=expected[steps][i]) return 2;sum+=actual[i];}
      if(sum!=1) return 3;
    }
    // Duplicated 0->1 edges still preserve the uniform stationary vector.
    std::vector<float> actual;
    pagerank_cpu(4,9,0.5f,std::vector<int>{0,1,3,4,5},
                 std::vector<int>{3,0,0,1,2},degrees,actual);
    for(float value:actual) if(value!=0.25f) return 4;
    pagerank_cpu(1,9,0.5f,std::vector<int>{0,1},std::vector<int>{0},std::vector<int>{1},actual);
    return actual==std::vector<float>{1}?0:5;
}
'''
    path = work / "reference.cpp"
    path.write_text(program)
    binary = work / "reference"
    subprocess.run([compiler, "-O2", "-std=c++14", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                    str(path), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=20)
    print("PASS: production CPU PageRank exact trajectories, duplicate edges, mass and single vertex", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--work-dir", type=Path, required=True, help="New, nonexistent directory")
    ap.add_argument("--host-only", action="store_true")
    ap.add_argument("--compiler", default="c++")
    ap.add_argument("--platform", choices=("cuda", "rocm"))
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    if not args.host_only and not args.platform:
        ap.error("--platform is required for GPU builds")
    args.source, args.work_dir = args.source.resolve(), args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=False)
    relative = Path("tier2/graph_pr/hip")
    src = args.source / relative
    source = (src / "graph_pr.hip").read_text()
    if args.host_only:
        host_reference(source, args.work_dir, args.compiler)
        return 0
    inputs = [relative / name for name in ("graph_pr.hip", "Makefile", "params.json")]
    inputs += [Path("common") / name for name in ("bench_timing.h", "bench_warmup.h", "hip_cuda_compat.h")]
    hashes = {str(p): hashlib.sha256((args.source / p).read_bytes()).hexdigest() for p in inputs}
    binaries = {}
    for variant in ("production", "faults"):
        root = args.work_dir / variant
        dst = root / relative
        dst.mkdir(parents=True)
        shutil.copytree(args.source / "common", root / "common")
        for name in ("Makefile", "params.json"):
            shutil.copy2(src / name, dst / name)
        (dst / "graph_pr.hip").write_text(source if variant == "production" else instrument(source))
        with (root / "build.log").open("w") as log:
            subprocess.run(["make", "PLATFORM=" + args.platform], cwd=dst, check=True,
                           stdout=log, stderr=subprocess.STDOUT)
        binaries[variant] = dst / "graph_pr"
    if args.compile_only:
        print("Production and measured-rank fault binaries compiled; GPU tests not run")
        return 0
    clean_env = {k:v for k,v in os.environ.items() if not k.startswith(("BENCH_PARAM_", "PR_TEST_"))}
    # variant,n,degree,steps,warm,fixture,kind,index,expected per-vertex errors
    cases = [("production",n,16,5,"false",False,None,0,0) for n in (1,2,9,1023,1024,1025,2049)]
    cases += [("faults",4,16,t,"false",True,None,0,0) for t in (1,2,3,4,5,8)]
    cases += [("faults",2049,16,5,"false",False,"balanced",i,2) for i in (0,1023,1024,1536,2048)]
    cases += [("faults",2049,16,5,"false",False,kind,i,1)
              for kind in ("nan","pinf","ninf") for i in (0,1024,2048)]
    cases += [("faults",2049,16,5,"false",False,"many",1024,12)]
    # For the fixture at the production damping0.85, stale odd-step buffers
    # disagree at3,3,2 vertices; even steps already end in d_rank_a.
    cases += [("faults",4,16,t,"false",True,"old",0,e) for t,e in ((1,3),(2,0),(3,3),(4,0),(5,2))]
    cases += [("faults",4,16,5,"true",True,None,0,0),
              ("faults",2049,16,5,"true",False,"balanced",1024,2)]
    if args.sweep:
        manifest = json.loads((src / "params.json").read_text())
        degrees = next(x["values"] for x in manifest["non_scaling_params"] if x["name"] == "degree")
        cases += [("production",n,d,5,"true",False,None,0,0)
                  for n in manifest["scaling_param"]["values"] for d in degrees]
    results = []

    def report(complete=False):
        data = dict(complete=complete, source_sha256=hashes,
                    binary_sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in binaries.items()},
                    expected=len(cases), total=len(results), passed=sum(x["passed"] for x in results), cases=results)
        (args.work_dir / "report.json").write_text(json.dumps(data, indent=2) + "\n")
        return data

    for variant,n,degree,steps,warm,fixture,kind,index,errors in cases:
        label = f"{variant}-{n}-{degree}-steps{steps}-warm{warm}-fixture{fixture}-{kind or 'control'}-{index}"
        env = dict(clean_env, BENCH_PARAM_vertices=str(n), BENCH_PARAM_degree=str(degree),
                   BENCH_PARAM_pr_iterations=str(steps), BENCH_PARAM_vector_add_warmup=warm)
        if fixture:
            env["PR_TEST_FIXTURE"] = "1"
        if kind == "old":
            env["PR_TEST_OLD_BUFFER"] = "1"
        elif kind:
            env.update(PR_TEST_KIND=kind, PR_TEST_INDEX=str(index))
        start = time.monotonic()
        try:
            p = subprocess.run([str(binaries[variant])], env=env, capture_output=True, text=True, timeout=60)
            stdout,stderr,code = p.stdout,p.stderr,p.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or b"").decode(errors="replace")
            stderr = (exc.stderr or b"").decode(errors="replace")
            code = "timeout"
        seconds = time.monotonic() - start
        (args.work_dir / (label + ".stdout")).write_text(stdout)
        (args.work_dir / (label + ".stderr")).write_text(stderr)
        mass_bad = kind in ("nan", "pinf", "ninf")
        passed = code == (1 if errors else 0) and seconds < 60
        suffix = ", rank mass not conserved" if mass_bad else ""
        passed &= f"PageRank verification: {'FAIL' if errors else 'PASS'} ({n} vertices, {errors} mismatches{suffix})" in stderr
        passed &= f"Max error ({n} vertices):" in stderr
        passed &= stderr.count("Mismatch at vertex ") == min(errors,10)
        passed &= ("Mass conservation violated:" in stderr) == mass_bad
        if kind and kind != "old":
            passed &= f"PR_TEST_INJECT {index} {kind} {errors}" in stderr
            passed &= f"Mismatch at vertex {index}:" in stderr
        try:
            k,s = [json.loads(line) for line in stdout.splitlines()]
            passed &= k["type"] == "kernel" and k["name"] == "pagerank_kernel" and k["iter"] == 0
            passed &= k["params"] == dict(vertices=n,degree=degree,pr_iterations=steps,verification="full_cpu_v1")
            passed &= all(math.isfinite(k[f]) and k[f] >= 0 for f in ("time_ms", "host_time_ms"))
            passed &= s["type"] == "summary" and math.isfinite(s["total_time_ms"])
            passed &= abs(s["total_time_ms"] - k["time_ms"]) < 0.000002
        except (ValueError,KeyError,TypeError):
            passed = False
        results.append(dict(label=label,variant=variant,vertices=n,degree=degree,steps=steps,
                            warm=warm,fixture=fixture,kind=kind,index=index,exit_code=code,seconds=seconds,passed=bool(passed)))
        report()
        print(f"{'PASS' if passed else 'FAIL'} {label}: {seconds:.3f}s", flush=True)
    final = report(complete=True)
    print(f"{final['passed']}/{final['total']} PageRank cases passed")
    return 0 if final["passed"] == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
