#!/usr/bin/env python3
"""Exercise actual Tier 2 verification paths with test-only source injection.

Requires Python 3.8+, make, and a CUDA or ROCm compiler/GPU. Original sources
are never edited; isolated copies receive hooks immediately before comparison.
NaN/+Inf/-Inf affect one compared sample; the incorrect finite control affects
all compared samples to exceed Barnes-Hut's existing approximation allowance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

# (anchor before error calculation, output/reference pairs). Exact anchors fail
# closed when verification code moves; these are injection sites, not copies of
# the verifier or its tolerance logic.
SITES = {
    "altis_cfd": [("float diff_rho =", [("rho", "gpu_flux_rho[i]", "cpu_f_rho[i]"), ("mx", "gpu_flux_mx[i]", "cpu_f_mx[i]")])],
    "cuda_convolution_separable": [("float diff = fabsf(h_output[idx]", [("output", "h_output[idx]", "ref[idx]")])],
    "cuda_nbody": [("const bool position_ok =", [("pos_" + c, f"h_pos[i].{c}", f"h_pos_orig[i].{c}") for c in "xyzw"] + [("vel_" + c, f"h_vel[i].{c}", f"h_vel_orig[i].{c}") for c in "xyzw"])],
    "cuda_transpose": [("int naive_errors = 0;", [("naive", "h_odata[0]", "h_ref[0]")]), ("int opt_errors = 0;", [("optimized", "h_odata[0]", "h_ref[0]")])],
    "graph_pr": [("float err = fabsf(h_gpu_rank[i]", [("rank", "h_gpu_rank[i]", "h_cpu_rank[i]")])],
    "heteromark_fir": [("float diff = fabsf(h_output[i]", [("output", "h_output[i]", "h_ref[i]")])],
    "heteromark_pagerank": [("float diff = fabsf(h_pr[v]", [("rank", "h_pr[v]", "refPr[v]")])],
    "lonestar_bh": [("float ref_mag = sqrtf(ref_ax[i]", [("x", "gpu_ax[i]", "ref_ax[i]"), ("y", "gpu_ay[i]", "ref_ay[i]")])],
    "npb_cg": [("float rel_res = res_norm /", [("residual", "res_norm", "b_norm")])],
    "npb_mg": [("float rel_res = res_norm /", [("residual", "res_norm", "rhs_norm")])],
    "parboil_cutcp": [("float tol = 1e-3f * (fabsf(ref)", [("potential", "got", "ref")])],
    "parboil_sgemm": [("float diff = fabsf(h_C[i]", [("C", "h_C[i]", "h_ref[i]")])],
    "parboil_spmv": [("float thresh = 1e-4f *", [("selftest", "got[r]", "ref[r]")]), ("float thresh = 1e-3f *", [("y", "h_y[r]", "ref[r]")])],
    "polybench_2dconv": [
        ("double tolerance = interior ?", [("output", "actual", "expected")]),
        ("const bool verified = verify_convolution(",
         [("measured_tail", "h_B[(N-2)*N+N-2]", "h_A[(N-2)*N+N-2]")]),
    ],
    "polybench_3dconv": [
        ("double tolerance = interior ?", [("output", "actual", "expected")]),
        ("const bool verified = verify_convolution3d(",
         [("measured_tail", "h_output[((N-2)*N+N-2)*N+N-2]", "h_input[((N-2)*N+N-2)*N+N-2]")]),
    ],
    "polybench_atax": [("float rel = fabsf(h_y[j]", [("y", "h_y[j]", "y_ref[j]")])],
    "polybench_bicg": [("float rel = fabsf(h_s[j]", [("s", "h_s[j]", "ref")]), ("float rel = fabsf(h_q[i]", [("q", "h_q[i]", "ref")])],
    "polybench_gramschmidt": [("float expected = (ci == cj) ? 1.0f : 0.0f;", [("dot", "dot", "expected")])],
    "polybench_jacobi2d": [("float rel = fabsf(gpu - ref)", [("grid", "gpu", "ref")])],
    "polybench_mvt": [("float rel1 =", [("x1", "h_x1[i]", "x1_ref[i]")]), ("float rel2 =", [("x2", "h_x2[i]", "x2_ref[i]")])],
    "rodinia_backprop": [
        ("double tolerance = std::max(absolute_tolerance,", [("value", "actual", "expected")]),
        ("return verify_backprop_epoch(initial,net,laps)",
         [("partial_tail", "laps.partials.back()", "initial.weights.back()"),
          ("weight_tail", "net.weights.back()", "initial.previous_weights.back()"),
          ("momentum_tail", "net.previous_weights.back()", "initial.previous_weights.back()"),
          ("output", "net.output", "initial.target")]),
    ],
    "shoc_fft": [("double dx = (double)result[i].x", [(c, f"result[i].{c}", f"source[i].{c}") for c in "xy"])],
    "shoc_gemm": [("float diff = fabsf(h_C[i]", [("C", "h_C[i]", "h_ref[i]")])],
    "shoc_reduction": [("double rel_err = fabs(", [("sum", "gpu_result", "cpu_result")])],
    "shoc_scan": [("float diff = fabsf(h_output[i]", [("output", "h_output[i]", "h_ref[i]")])],
    "shoc_spmv": [("float rel = fabsf(h_y[i]", [("y", "h_y[i]", "ref")])],
    "shoc_stencil2d": [("float rel = fabsf(gpu - ref)", [("grid", "gpu", "ref")])],
    "shoc_triad": [("float diff = fabsf(h_a[i]", [("a", "h_a[i]", "h_ref[i]")])],
    "tango_binomial_options": [("float tol = 1e-2f *", [("price", "h_prices[i]", "ref")])],
    "tango_blackscholes": [("float tol = 1e-3f *", [("call", "h_call[i]", "ref_call[i]"), ("put", "h_put[i]", "ref_put[i]")])],
}

HOOK = r'''
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
template<class T> static void r04_poison(T& value, const char* kind) {
    if (!strcmp(kind, "nan")) value = std::numeric_limits<T>::quiet_NaN();
    else if (!strcmp(kind, "pinf")) value = std::numeric_limits<T>::infinity();
    else if (!strcmp(kind, "ninf")) value = -std::numeric_limits<T>::infinity();
    else value = static_cast<T>(1000.0 * (std::fabs(static_cast<double>(value)) + 1.0));
}
template<class G, class R> static void r04_inject(const char* site, G& got, R& ref) {
    const char* target = std::getenv("R04_SITE");
    if (!target || std::strcmp(target, site)) return;
    const char* kind = std::getenv("R04_KIND");
    const char* side = std::getenv("R04_SIDE");
    if (!kind || !side) std::abort();
    static bool injected = false;
    if (injected && std::strcmp(kind, "bad")) return;
    if (!injected) std::fprintf(stderr, "R04_INJECT %s %s %s\n", site, side, kind);
    if (!std::strcmp(side, "got")) r04_poison(got, kind);
    else r04_poison(ref, kind);
    injected = true;
}
'''


def instrument(source, sites):
    for anchor, pairs in sites:
        assert source.count(anchor) == 1, (anchor, source.count(anchor))
        pos = source.index(anchor)
        # The Gram-Schmidt hook needs both dot and the constant reference.
        if anchor.startswith("float expected ="):
            pos = source.index("\n", pos) + 1
        else:
            pos = source.rfind("\n", 0, pos) + 1
        calls = "".join(f'    r04_inject("{label}", {got}, {ref});\n' for label, got, ref in pairs)
        source = source[:pos] + calls + source[pos:]
    return HOOK + source


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--work-dir", type=Path, required=True, help="new directory for copies, binaries, logs, report")
    ap.add_argument("--platform", choices=("cuda", "rocm"), required=True)
    ap.add_argument("--benchmark", action="append", choices=sorted(SITES), help="repeat to select a subset")
    ap.add_argument("--compile-only", action="store_true", help="prepare and compile; do not claim runtime verification")
    args = ap.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.source / "common", args.work_dir / "common")
    results = []
    for name in args.benchmark or sorted(SITES):
        src = args.source / "tier2" / name
        dest = args.work_dir / "tier2" / name
        dest.mkdir(parents=True)
        for filename in ("Makefile", "params.json"):
            shutil.copy2(src / filename, dest / filename)
        for header in src.glob("*.h"):
            shutil.copy2(header, dest / header.name)
        original = (src / f"{name}.hip").read_text()
        (dest / f"{name}.hip").write_text(instrument(original, SITES[name]))
        with (dest / "build.log").open("w") as log:
            subprocess.run(["make", f"PLATFORM={args.platform}"], cwd=dest, stdout=log, stderr=subprocess.STDOUT, check=True)
        if args.compile_only:
            print(f"{name}: compiled (GPU checks not run)", flush=True)
            continue
        manifest = json.loads((src / "params.json").read_text())
        env = {k: v for k, v in os.environ.items() if not k.startswith(("BENCH_PARAM_", "R04_"))}
        scaling = manifest["scaling_param"]
        env["BENCH_PARAM_" + scaling["name"]] = str(scaling["values"][0])
        for param in manifest.get("non_scaling_params", []):
            env["BENCH_PARAM_" + param["name"]] = str(param["default"])
        env.update(BENCH_PARAM_vector_add_warmup="false", BENCH_PARAM_verify="true")
        for precision in (["float", "double"] if name == "shoc_fft" else [None]):
            if precision:
                env["BENCH_PARAM_precision"] = precision
            sites = [label for _, pairs in SITES[name] for label, _, _ in pairs]
            # One normal execution per code path, not one per injection site.
            controls = [([], "control")]
            if name == "parboil_spmv":
                controls.append((["--selftest"], "control-selftest"))
            cases = [(label, cli, {}) for cli, label in controls]
            for site in sites:
                cli = ["--selftest"] if name == "parboil_spmv" and site == "selftest" else []
                for side in ("got", "ref"):
                    for kind in ("nan", "pinf", "ninf"):
                        cases.append((f"{site}-{side}-{kind}", cli, dict(R04_SITE=site, R04_SIDE=side, R04_KIND=kind)))
                cases.append((f"{site}-got-bad", cli, dict(R04_SITE=site, R04_SIDE="got", R04_KIND="bad")))
            for label, cli, injection in cases:
                case_name = "-".join(filter(None, [name, precision, label]))
                run = subprocess.run([str((dest / name).resolve())] + cli, env={**env, **injection}, capture_output=True, text=True, timeout=120)
                (dest / f"{case_name}.stdout").write_text(run.stdout)
                (dest / f"{case_name}.stderr").write_text(run.stderr)
                # Explicit diagnostic + normal failure exit prevents a crash or
                # unexecuted hook from masquerading as verification rejection.
                if injection:
                    reached = f"R04_INJECT {injection['R04_SITE']} {injection['R04_SIDE']} {injection['R04_KIND']}" in run.stderr
                    rejected = any(word in run.stderr.lower() for word in ("fail", "mismatch"))
                    if name == "cuda_transpose":
                        # A poisoned naive reference survives into the optimized
                        # check: require the selected comparator itself to fail.
                        diagnostic = "Naive mismatch" if injection["R04_SITE"] == "naive" else "Optimized mismatch"
                        rejected = diagnostic in run.stderr
                    runtime_error = "HIP error" in run.stderr
                    passed = reached and rejected and not runtime_error and run.returncode == 1
                else:
                    passed = run.returncode == 0 and "PASS" in run.stderr
                results.append(dict(case=case_name, passed=passed, exit=run.returncode, source_sha256=hashlib.sha256(original.encode()).hexdigest()))
                if not passed:
                    print(f"FAIL {case_name}: exit {run.returncode}; see {dest}", flush=True)
        print(f"{name}: completed", flush=True)
        (args.work_dir / "report.json").write_text(json.dumps(results, indent=2) + "\n")
    if args.compile_only:
        return 0
    failures = sum(not r["passed"] for r in results)
    print(f"{len(results) - failures}/{len(results)} verification cases passed", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
