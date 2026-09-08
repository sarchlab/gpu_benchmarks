#!/usr/bin/env python3
"""Check the exact SRAD parameter prefix without allocating boundary-size images."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

from srad_invalid_parameters import run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path(__file__).resolve().parents[1] / "tier2/rodinia_srad/rodinia_srad.hip")
    parser.add_argument("--compiler", default="c++")
    parser.add_argument("--binary", type=Path, help="Also test oversized inputs against the real GPU binary")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    source = args.source.read_text()
    parse_start = source.index("static int parseIntParam(")
    parse_end = source.index("// ---------------------------------------------------------------------------", parse_start)
    main_start = source.index("int main(")
    main_end = source.index("    int device_id = 0;", main_start)
    block = re.findall(r"^#define BLOCK_SIZE [^\n]+", source, re.MULTILINE)
    assert len(block) == 1
    # Compile the production parser and entire main prefix through the point
    # immediately before the first HIP runtime call. GPU kernels are untouched.
    host_source = ("#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n#include <climits>\n" +
                   block[0] + "\n" + source[parse_start:parse_end] + source[main_start:main_end] +
                   '    printf("%d %d %d\\n", rows, cols, size_I);\n    return 0;\n}\n')
    clean = {k:v for k,v in os.environ.items() if not k.startswith(("BENCH_PARAM_", "SRAD_TEST_"))}
    # Expected values use Python's unbounded integers, not the C++ expression.
    int_max = (1 << 31) - 1
    last = math.isqrt(int_max) // 16 * 16
    assert last == 46336
    valid = [1, 2, 15, 16, 17, 31, 32, 33, 511, 512, 513, 8191, 8192, 8193,
             last-17, last-16, last-15, last-1, last]
    invalid = [last+1, 46340, 46341, 46351, 46352, 65535, 65536, 65537,
               1 << 20, 1 << 30, int_max-15, int_max-14, int_max-1, int_max]
    results = []
    with tempfile.TemporaryDirectory(prefix="srad-index-range-") as temporary:
        temporary = Path(temporary)
        cpp, host = temporary / "parameters.cpp", temporary / "parameters"
        cpp.write_text(host_source)
        subprocess.run([args.compiler, "-O1", "-std=c++14", "-fsanitize=address,undefined",
                        "-fno-sanitize-recover=all", str(cpp), "-o", str(host)], check=True, timeout=60)
        targets = [("host-prefix", host, valid + invalid)]
        if args.binary:
            targets.append(("gpu-binary-no-device", args.binary.resolve(), invalid))
        for target, binary, sizes in targets:
            for requested in sizes:
                rounded = ((requested + 15) // 16) * 16
                env = dict(clean, BENCH_PARAM_image_size=str(requested), BENCH_PARAM_num_iterations="1",
                           BENCH_PARAM_vector_add_warmup="false", CUDA_VISIBLE_DEVICES="",
                           HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="")
                result = run_case(binary, [], env, f"{target}-{requested}")
                if requested <= last:
                    expected = (0, f"{rounded} {rounded} {rounded*rounded}\n", "")
                else:
                    expected = (1, "", f"Invalid SRAD image_size={requested}: rounded dimension {rounded} exceeds signed-int indexing\n")
                result["passed"] = (result["exit"], result["stdout"], result["stderr"]) == expected
                results.append(result)
    report = dict(total=len(results), passed=sum(r["passed"] for r in results), results=results)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    for result in results:
        if not result["passed"]:
            print("FAIL", result["label"], result)
    print(f"{report['passed']}/{report['total']} SRAD index-range cases passed")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
