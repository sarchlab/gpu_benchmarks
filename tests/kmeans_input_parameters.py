#!/usr/bin/env python3
"""Check k-means parameter admission without allocating boundary-sized inputs."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time


NAMES = ("points", "features", "clusters", "max_iter")
DEFAULTS = (100000, 34, 5, 10)


def env_case(label, n=16, d=34, k=5, t=1, cli=()):
    values = (n, d, k, t)
    return label, list(cli), dict(zip(NAMES, map(str, values))), values


def expected(values):
    n, d, k, t = values
    if min(values) < 1:
        return 1, "", f"Invalid k-means parameters: points={n} features={d} clusters={k} max_iter={t}; all must be positive\n"
    if k*d > 1088:
        return 1, "", f"clusters*features ({k*d}) exceeds constant-memory capacity (1088)\n"
    if k > n:
        return 1, "", f"Invalid k-means parameters: clusters={k} exceeds points={n}\n"
    if n*d > (1 << 31)-1:
        return 1, "", f"Invalid k-means parameters: points*features={n*d} exceeds signed-int indexing\n"
    # Python integer square root and unlimited integers independently determine
    # the smallest squared grid and its padded point-index range.
    blocks = (n+255)//256
    side = math.isqrt(blocks-1)+1
    padded = side*side*256
    if padded-1 > (1 << 31)-1:
        return 1, "", f"Invalid k-means parameters: padded point count={padded} exceeds signed-int indexing\n"
    return 0, f"{n} {d} {k} {t} {side*side} 256 {side}\n", ""


def run(binary, case, clean, no_device=False, warm=False):
    label, cli, params, values = case
    env = dict(clean)
    env.update({"BENCH_PARAM_"+k:v for k,v in params.items()})
    if not warm:
        env["BENCH_PARAM_vector_add_warmup"] = "false"
    if no_device:
        env.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="")
    start = time.monotonic()
    try:
        result = subprocess.run([str(binary), *cli], env=env, capture_output=True, text=True, timeout=60)
        row = dict(exit=result.returncode, stdout=result.stdout, stderr=result.stderr, timed_out=False)
    except subprocess.TimeoutExpired as exc:
        def decoded(output):
            return output.decode(errors="replace") if isinstance(output, bytes) else (output or "")
        row = dict(exit=None, stdout=decoded(exc.stdout), stderr=decoded(exc.stderr), timed_out=True)
    row.update(label=label, seconds=time.monotonic()-start, values=values)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "tier2/rodinia_kmeans/hip/rodinia_kmeans.hip")
    parser.add_argument("--compiler", default="c++")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--gpu-controls", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.gpu_controls and not args.binary:
        parser.error("--gpu-controls requires --binary")
    source = args.source.read_text()
    parse_start = source.index("static int parseIntParam(")
    parse_end = source.index("// ---------------------------------------------------------------------------", parse_start)
    main_start = source.index("int main(")
    main_end = source.index("    int device_id = 0;", main_start)
    macro = re.findall(r"^#define ASSUMED_NR_CLUSTERS [^\n]+", source, re.MULTILINE)
    assert len(macro) == 1
    host_source = ("#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n#include <climits>\n#include <cmath>\n" +
                   macro[0]+"\n"+source[parse_start:parse_end]+source[main_start:main_end]+
                   '    printf("%d %d %d %d %d %d %d\\n", npoints, nfeatures, nclusters, max_iter, num_blocks, num_threads, num_blocks_perdim);\n    return 0;\n}\n')
    cases = [("default", [], {}, DEFAULTS)]
    for axis in range(4):
        for value in (0, -1, -(1 << 31)):
            values = [16, 34, 5, 1]; values[axis] = value
            cases.append(env_case(f"{NAMES[axis]}-{value}", *values))
        cli = ["--"+NAMES[axis], "0"]
        cases.append((f"cli-{NAMES[axis]}-fallback", cli, {}, DEFAULTS))
    cases += [env_case("single-point", 1, 1, 1), env_case("centers-equal-points", 5, 34, 5),
              env_case("features-capacity", 16, 1088, 1), env_case("clusters-capacity", 1088, 1, 1088),
              env_case("constant-capacity", 32, 34, 32), env_case("constant-over-capacity", 33, 34, 33),
              env_case("constant-overflow", 65536, 65536, 65536),
              env_case("constant-largest-product", (1 << 31)-1, (1 << 31)-1, (1 << 31)-1),
              env_case("centers-exceed-points", 1, 34, 5), env_case("centers-one-over", 32, 1, 33),
              env_case("padded-last-safe", 2147024896, 1, 1),
              env_case("padded-one-over", 2147024897, 1, 1),
              env_case("padded-largest-int", (1 << 31)-1, 1, 1)]
    for d in (2, 34, 1088):
        last = ((1 << 31)-1)//d
        cases += [env_case(f"feature-product-{d}-below", last-1, d, 1),
                  env_case(f"feature-product-{d}-last", last, d, 1),
                  env_case(f"feature-product-{d}-over", last+1, d, 1)]
    for n in (16, 17, 255, 256, 257, 1024, 1025, 2304, 2305, 4096, 4097):
        cases.append(env_case(f"grid-{n}", n))
    cases += [env_case("invalid-env-overrides-cli", 0, cli=("--points", "64")),
              env_case("valid-env-overrides-cli", 17, 3, 2, 2,
                       ("--points", "64", "--features", "7", "--clusters", "4", "--max_iter", "3"))]
    clean = {k:v for k,v in os.environ.items() if not k.startswith(("BENCH_PARAM_", "KMEANS_TEST_"))}
    results = []
    with tempfile.TemporaryDirectory(prefix="kmeans-parameters-") as temporary:
        temporary = Path(temporary)
        cpp, host = temporary / "parameters.cpp", temporary / "parameters"
        cpp.write_text(host_source)
        subprocess.run([args.compiler, "-O1", "-std=c++14", "-fsanitize=address,undefined",
                        "-fno-sanitize-recover=all", str(cpp), "-o", str(host)], check=True, timeout=60)
        for case in cases:
            row = run(host, case, clean, no_device=True)
            row["passed"] = (row["exit"], row["stdout"], row["stderr"]) == expected(case[3])
            row["label"] = "host-prefix-"+row["label"]
            results.append(row)
        if args.binary:
            for case in cases:
                if expected(case[3])[0] == 0:
                    continue
                row = run(args.binary.resolve(), case, clean, no_device=True)
                row["passed"] = (row["exit"], row["stdout"], row["stderr"]) == expected(case[3])
                row["label"] = "gpu-binary-no-device-"+row["label"]
                results.append(row)
        if args.gpu_controls:
            manifest = json.loads((args.source.parent / "params.json").read_text())
            controls = [env_case(f"declared-{n}", n, 34, 5, 10) for n in manifest["scaling_param"]["values"]]
            selected = {"default", "single-point", "centers-equal-points", "features-capacity",
                        "clusters-capacity", "constant-capacity", "grid-17", "grid-256", "grid-257",
                        "valid-env-overrides-cli"}
            controls += [case for case in cases if case[0] in selected or case[0].endswith("-fallback")]
            for case in controls:
                row = run(args.binary.resolve(), case, clean,
                          warm=case[0]=="default" or case[0].startswith("declared-"))
                passed = row["exit"] == 0 and row["seconds"] < 60
                n,d,k,t = case[3]
                passed &= f"K-means verification: PASS ({n*d+n*t+k*d} values, 0 mismatches)" in row["stderr"]
                try:
                    first, second, summary = [json.loads(line) for line in row["stdout"].splitlines()]
                    params = dict(zip(NAMES, case[3]), threads_per_block=256,
                                  verification="full_cpu_fma_trajectory_v1")
                    for kernel, name in zip((first, second), ("invert_mapping", "kmeansPoint")):
                        passed &= kernel["type"]=="kernel" and kernel["name"]==name and kernel["iter"]==0
                        passed &= kernel["params"]==params
                        passed &= all(math.isfinite(kernel[f]) and kernel[f]>=0 for f in ("time_ms", "host_time_ms"))
                    passed &= summary["type"]=="summary" and abs(summary["total_time_ms"]-first["time_ms"]-second["time_ms"])<.000004
                except (ValueError, KeyError, TypeError):
                    passed = False
                row["passed"] = bool(passed); row["label"] = "gpu-control-"+row["label"]
                results.append(row)
    report = dict(total=len(results), passed=sum(row["passed"] for row in results), results=results)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2)+"\n")
    for row in results:
        if not row["passed"]:
            print("FAIL", row["label"], row)
    print(f"{report['passed']}/{report['total']} k-means input cases passed")
    return 0 if report["passed"]==report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
