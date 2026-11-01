#!/usr/bin/env python3
"""Check CUTCP metadata ownership with hardware sanitizers and full small outputs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time


def instrument(source):
    anchor = "    // ---- verification: complete actual measured lattice, untimed ----"
    if anchor not in source:  # Historical pre-race baseline.
        anchor = "    // ---- verification: CPU reference over the binned atoms. With the"
    if source.count(anchor) != 1:
        raise ValueError("expected one measured-lattice anchor")
    source = source.replace(anchor, r'''
    if(const char* path=std::getenv("CUTCP_TEST_DUMP")) {
        FILE* file=std::fopen(path,"wb");if(!file)std::abort();
        if(std::fwrite(lattice.data(),sizeof(float),lattice.size(),file)!=lattice.size())std::abort();
        if(std::fclose(file))std::abort();
        std::fprintf(stderr,"CUTCP_TEST_DUMP %zu values\n",lattice.size());
    }
''' + anchor)
    return source


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--work-dir", type=Path, required=True, help="New, nonexistent directory")
    ap.add_argument("--platform", choices=("cuda", "rocm"), required=True)
    ap.add_argument("--baseline-source", type=Path, help="Pre-repair parboil_cutcp.hip for regression comparison")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    if not args.compile_only and args.platform != "cuda":
        ap.error("runtime sanitizer checks currently require CUDA Compute Sanitizer")
    args.source,args.work_dir = args.source.resolve(),args.work_dir.resolve()
    args.work_dir.mkdir(parents=True,exist_ok=False)
    relative = Path("tier2/parboil_cutcp/hip")
    src = args.source / relative
    source = (src / "parboil_cutcp.hip").read_text()
    inputs = [relative / n for n in ("parboil_cutcp.hip","cpu_reference.h","Makefile","params.json")]
    inputs += [Path("common_hip") / n for n in ("bench_timing.h","bench_warmup.h","hip_cuda_compat.h")]
    hashes = {str(p):hashlib.sha256((args.source/p).read_bytes()).hexdigest() for p in inputs}
    variants = dict(production=source,dump=instrument(source))
    if args.baseline_source:
        variants["baseline"] = instrument(args.baseline_source.read_text())
    binaries = {}
    for name,text in variants.items():
        root = args.work_dir / name
        dst = root / relative
        dst.mkdir(parents=True)
        shutil.copytree(args.source/"common_hip",root/"common_hip")
        for f in ("Makefile","params.json","cpu_reference.h"):
            shutil.copy2(src/f,dst/f)
        (dst/"parboil_cutcp.hip").write_text(text)
        with (root/"build.log").open("w") as log:
            subprocess.run(["make","PLATFORM="+args.platform],cwd=dst,stdout=log,stderr=subprocess.STDOUT,check=True)
        binaries[name] = dst / "parboil_cutcp"
    if args.compile_only:
        print("CUTCP production and complete-small-output variants compiled; hardware checks not run")
        return 0
    env = {k:v for k,v in os.environ.items() if not k.startswith(("BENCH_PARAM_","CUTCP_TEST_"))}
    results = []

    def record():
        report = dict(source_sha256=hashes,binary_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in binaries.items()},
                      total=len(results),passed=sum(x["passed"] for x in results),cases=results)
        (args.work_dir/"report.json").write_text(json.dumps(report,indent=2)+"\n")

    def run(label,variant,atoms,dim,h,cut,warm,tool=None,expected_exit=0,dump=None):
        e = dict(env,BENCH_PARAM_num_atoms=str(atoms),BENCH_PARAM_lattice_dim=str(dim),
                 BENCH_PARAM_grid_spacing=str(h),BENCH_PARAM_cutoff_radius=str(cut),BENCH_PARAM_vector_add_warmup=warm)
        if dump:
            e["CUTCP_TEST_DUMP"] = str(dump)
        cmd = [str(binaries[variant])]
        if tool:
            cmd = ["compute-sanitizer","--tool",tool,"--error-exitcode","91"] + cmd
        start = time.monotonic()
        try:
            p = subprocess.run(cmd,env=e,capture_output=True,text=True,timeout=60)
            stdout,stderr,code = p.stdout,p.stderr,p.returncode
        except subprocess.TimeoutExpired as exc:
            stdout=(exc.stdout or b"").decode(errors="replace")
            stderr=(exc.stderr or b"").decode(errors="replace")
            code="timeout"
        seconds = time.monotonic()-start
        (args.work_dir/(label+".stdout")).write_text(stdout)
        (args.work_dir/(label+".stderr")).write_text(stderr)
        count = 64 if variant=="baseline" else dim**3
        passed = code==expected_exit and seconds<60 and f"PASS ({count} points verified)" in stderr
        if tool and expected_exit==0:
            passed &= "ERROR SUMMARY: 0 errors" in stdout+stderr or "RACECHECK SUMMARY: 0 hazards" in stdout+stderr
        elif tool:
            passed &= "hazard" in (stdout+stderr).lower() and "ERROR SUMMARY: 0 errors" not in stdout+stderr
        # Sanitizer diagnostics may also use stdout; normal executions retain
        # the exact two-record benchmark protocol.
        if not tool:
            try:
                k,s = [json.loads(line) for line in stdout.splitlines()]
                params=k["params"]
                passed &= k["type"]=="kernel" and k["name"]=="cuda_cutoff_potential_lattice6overlap" and k["iter"]==0
                passed &= all(params[key]==val for key,val in dict(num_atoms=atoms,lattice_dim=dim,grid_spacing=h,cutoff_radius=cut,bin_depth=16).items())
                passed &= isinstance(params["nbrlist_len"],int) and params["nbrlist_len"]>0
                passed &= all(math.isfinite(k[f]) and k[f]>=0 for f in ("time_ms","host_time_ms"))
                passed &= s["type"]=="summary" and math.isfinite(s["total_time_ms"])
                passed &= abs(s["total_time_ms"]-k["time_ms"])<0.000002
            except (ValueError,KeyError,TypeError):
                passed=False
        if dump:
            passed &= dump.exists() and dump.stat().st_size==dim**3*4
            passed &= f"CUTCP_TEST_DUMP {dim**3} values" in stderr
        results.append(dict(label=label,variant=variant,atoms=atoms,dim=dim,spacing=h,cutoff=cut,warm=warm,
                            tool=tool,exit=code,seconds=seconds,passed=bool(passed)))
        record()
        print(f"{'PASS' if passed else 'FAIL'} {label}: {seconds:.3f}s",flush=True)
        return bool(passed)

    small = [(1,8,.5,8.,"false"),(8,9,.5,8.,"false"),(8,16,.5,12.,"false"),
             (16,8,.5,8.,"false"),(64,32,.5,16.,"false"),(8,16,1.,12.,"false"),
             (128,32,.5,12.,"true"),(1,8,.5,8.,"true")]
    for i,c in enumerate(small):
        run(f"full-small-{i}","production",*c)
    for i in (0,2,4):
        run(f"racecheck-{i}","production",*small[i],tool="racecheck")
    for tool in ("memcheck","initcheck","synccheck"):
        run(tool,"production",*small[4],tool=tool)
    if args.baseline_source:
        for i in (0,2):
            run(f"baseline-racecheck-{i}","baseline",*small[i],tool="racecheck",expected_exit=91)
        pairs = small[:3]+[(5000,144,.5,8.,"true"),(8000,160,.5,16.,"true"),
                          (40000,192,.5,12.,"true"),(45000,256,.5,16.,"true")]
        for i,c in enumerate(pairs):
            a,b = args.work_dir/f"pair-{i}-old.bin",args.work_dir/f"pair-{i}-new.bin"
            ok_old=run(f"pair-{i}-baseline","baseline",*c,dump=a)
            ok_new=run(f"pair-{i}-current","dump",*c,dump=b)
            same = ok_old and ok_new and a.read_bytes()==b.read_bytes()
            results.append(dict(label=f"pair-{i}-full-lattice-equality",passed=same,
                                sha256=hashlib.sha256(b.read_bytes()).hexdigest() if b.exists() else None))
            record()
            # These are test-created dumps; retain hashes/logs without keeping
            # duplicate multi-megabyte output arrays for every comparison.
            for path in (a,b):
                if path.exists():path.unlink()
    if args.sweep:
        m=json.loads((src/"params.json").read_text())
        p={x["name"]:x["values"] for x in m["non_scaling_params"]}
        for n in m["scaling_param"]["values"]:
            for d in p["lattice_dim"]:
                for h in p["grid_spacing"]:
                    for c in p["cutoff_radius"]:
                        run(f"production-{n}-{d}-{h}-{c}","production",n,d,h,c,"true")
    record()
    passed=sum(x["passed"] for x in results)
    print(f"{passed}/{len(results)} CUTCP checks passed")
    return 0 if passed==len(results) else 1


if __name__=="__main__":
    raise SystemExit(main())
