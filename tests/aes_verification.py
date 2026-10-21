#!/usr/bin/env python3
"""Check complete measured AES ciphertext and NIST AES-256 known answers."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

# NIST SP 800-38A, F.1.5 ECB-AES256.Encrypt (printed page 26).
# https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-38a.pdf
VECTORS = r'''
static const char* kat_key="603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4";
static const char* kat_plain[]={"6bc1bee22e409f96e93d7e117393172a","ae2d8a571e03ac9c9eb76fac45af8e51",
 "30c81c46a35ce411e5fbc1191a0a52ef","f69f2445df4f9b17ad2b417be66c3710"};
static const char* kat_cipher[]={"f3eed1bdb5d2a03c064b5a7e3db181f8","591ccb10d410ed26dc5ba74a31362870",
 "b6ed21b99ca6f4f9f153e7b1beafed1d","23304b7a39f9f3ff067d8d8f9e24ecc7"};
static void unhex(const char* s,unsigned char* dst,int n) {
 for(int i=0;i<n;++i){unsigned x;if(std::sscanf(s+2*i,"%2x",&x)!=1)std::abort();dst[i]=(unsigned char)x;}
}
'''
HOOK = r'''
#include <vector>
static void aes_test_result(unsigned char* data,int n) {
    if(std::getenv("AES_TEST_KAT")) {
        std::vector<unsigned char> actual((size_t)n*16);
        HIP_CHECK(hipMemcpy(actual.data(),data,actual.size(),hipMemcpyDeviceToHost));
        for(int b=0;b<n;++b) {
            unsigned char expected[16];unhex(kat_cipher[b%4],expected,16);
            if(std::memcmp(actual.data()+(size_t)b*16,expected,16)) std::abort();
        }
        std::fprintf(stderr,"AES_TEST_KAT_PASS %d blocks\n",n);
    }
    const char* index=std::getenv("AES_TEST_INDEX");
    if(!index) return;
    size_t at=std::strtoull(index,nullptr,10), bytes=(size_t)n*16;
    size_t count=std::getenv("AES_TEST_MANY")?12:1;
    if(count>bytes || at>bytes-count) std::abort();
    for(size_t j=0;j<count;++j) {
        unsigned char value;
        HIP_CHECK(hipMemcpy(&value,data+at+j,1,hipMemcpyDeviceToHost));
        value^=1;
        HIP_CHECK(hipMemcpy(data+at+j,&value,1,hipMemcpyHostToDevice));
    }
    std::fprintf(stderr,"AES_TEST_INJECT %zu %zu\n",at,count);
}
'''


def instrument(source):
    anchors = ["#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)",
               "    // Key expansion\n", "    // Device allocation\n",
               "        HIP_CHECK(hipMemcpy(h_data, d_data, data_bytes, hipMemcpyDeviceToHost));"]
    for anchor in anchors:
        if source.count(anchor) != 1:
            raise ValueError("expected one production anchor: " + anchor)
    source = source.replace(anchors[0], anchors[0] + VECTORS + HOOK)
    source = source.replace(anchors[1], '    if(std::getenv("AES_TEST_KAT")) unhex(kat_key,key,32);\n' + anchors[1])
    source = source.replace(anchors[2], '''    if(std::getenv("AES_TEST_KAT")) {
        for(int b=0;b<N;++b) unhex(kat_plain[b%4],h_orig+(size_t)b*16,16);
        std::memcpy(h_data,h_orig,data_bytes);
    }
''' + anchors[2])
    return source.replace(anchors[3], "        aes_test_result(d_data,N);\n" + anchors[3])


def host_reference(source, work, compiler):
    def section(start, end):
        first = source.index(start)
        return source[first:source.index(end, first)]
    program = "#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n"
    program += section("static const unsigned char h_sbox[", "// AES-256 round constants")
    program += section("static const unsigned char h_rcon[", "// ---------------------------------------------------------------------------")
    program += section("static void key_expansion_256(", "// ---------------------------------------------------------------------------")
    program += section("static unsigned char h_xtime(", "// ---------------------------------------------------------------------------")
    program += VECTORS + r'''
int main() {
    unsigned char key[32],saved_key[32],expanded[242];
    unhex(kat_key,key,32);std::memcpy(saved_key,key,32);
    expanded[0]=0xa5;expanded[241]=0x5a;
    key_expansion_256(key,expanded+1);
    if(expanded[0]!=0xa5 || expanded[241]!=0x5a || std::memcmp(key,saved_key,32)) return 1;
    unsigned char saved_expanded[242];std::memcpy(saved_expanded,expanded,242);
    for(int i=0;i<4;++i) {
        unsigned char block[18],expected[16];block[0]=0xa5;block[17]=0x5a;
        unhex(kat_plain[i],block+1,16);unhex(kat_cipher[i],expected,16);
        aes256_encrypt_cpu(block+1,expanded+1);
        if(std::memcmp(block+1,expected,16) || block[0]!=0xa5 || block[17]!=0x5a) return 2;
        if(std::memcmp(expanded,saved_expanded,242)) return 3;
    }
    return 0;
}
'''
    path = work / "reference.cpp"
    path.write_text(program)
    binary = work / "reference"
    subprocess.run([compiler, "-O2", "-std=c++14", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                    str(path), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=20)
    print("PASS: four NIST AES-256 known answers, key/input guards and immutable round keys", flush=True)


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
    relative = Path("tier2/heteromark_aes/hip")
    src = args.source / relative
    source = (src / "heteromark_aes.hip").read_text()
    if args.host_only:
        host_reference(source, args.work_dir, args.compiler)
        return 0
    inputs = [relative / name for name in ("heteromark_aes.hip", "Makefile", "params.json")]
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
        (dst / "heteromark_aes.hip").write_text(source if variant == "production" else instrument(source))
        with (root / "build.log").open("w") as log:
            subprocess.run(["make", "PLATFORM=" + args.platform], cwd=dst, check=True,
                           stdout=log, stderr=subprocess.STDOUT)
        binaries[variant] = dst / "heteromark_aes"
    if args.compile_only:
        print("Production and measured-ciphertext fault binaries compiled; GPU tests not run")
        return 0
    clean_env = {k:v for k,v in os.environ.items() if not k.startswith(("BENCH_PARAM_", "AES_TEST_"))}
    # variant,blocks,threads,warm,NIST fixture,first corrupted byte,error count
    cases = [("production",n,256,"false",False,None,0) for n in (1,2,4,255,256,257,1025)]
    cases += [("faults",n,256,warm,True,None,0) for n in (4,257) for warm in ("false","true")]
    cases += [("faults",1025,256,"false",False,at,1) for at in (0,4095,4096,8192,16399)]
    cases += [("faults",1025,256,"false",False,4096+j,1) for j in range(1,16)]
    cases += [("faults",1025,256,"false",False,4096,12),
              ("faults",1025,256,"true",False,16399,1),
              ("faults",257,256,"true",True,4111,1)]
    if args.sweep:
        manifest = json.loads((src / "params.json").read_text())
        blocks = next(x["values"] for x in manifest["non_scaling_params"] if x["name"] == "block_size")
        cases += [("production",n,b,"true",False,None,0)
                  for n in manifest["scaling_param"]["values"] for b in blocks]
    results = []

    def report(complete=False):
        data = dict(complete=complete, source_sha256=hashes,
                    binary_sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in binaries.items()},
                    expected=len(cases), total=len(results), passed=sum(x["passed"] for x in results), cases=results)
        (args.work_dir / "report.json").write_text(json.dumps(data, indent=2) + "\n")
        return data

    for variant,n,threads,warm,kat,at,errors in cases:
        label = f"{variant}-{n}-{threads}-warm{warm}-kat{kat}-at{at}-errors{errors}"
        env = dict(clean_env, BENCH_PARAM_size=str(n), BENCH_PARAM_block_size=str(threads),
                   BENCH_PARAM_vector_add_warmup=warm)
        if kat:
            env["AES_TEST_KAT"] = "1"
        if at is not None:
            env["AES_TEST_INDEX"] = str(at)
        if errors == 12:
            env["AES_TEST_MANY"] = "1"
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
        passed = code == (1 if errors else 0) and seconds < 60
        passed &= f"AES verification: {'FAIL' if errors else 'PASS'} ({n} blocks, {n*16} bytes, {errors} byte mismatches)" in stderr
        passed &= stderr.count("Mismatch at block ") == min(errors,10)
        if kat:
            passed &= f"AES_TEST_KAT_PASS {n} blocks" in stderr
        if at is not None:
            passed &= f"AES_TEST_INJECT {at} {errors}" in stderr
            passed &= f"Mismatch at block {at//16} byte {at%16}:" in stderr
        try:
            k,s = [json.loads(line) for line in stdout.splitlines()]
            passed &= k["type"] == "kernel" and k["name"] == "aes256_encrypt_kernel" and k["iter"] == 0
            passed &= k["params"] == dict(size=n,block_size=threads,verification="full_cpu_v1")
            passed &= all(math.isfinite(k[f]) and k[f] >= 0 for f in ("time_ms", "host_time_ms"))
            passed &= s["type"] == "summary" and math.isfinite(s["total_time_ms"])
            passed &= abs(s["total_time_ms"] - k["time_ms"]) < 0.000002
        except (ValueError,KeyError,TypeError):
            passed = False
        results.append(dict(label=label,variant=variant,blocks=n,threads=threads,warm=warm,
                            kat=kat,index=at,errors=errors,exit_code=code,seconds=seconds,passed=bool(passed)))
        report()
        print(f"{'PASS' if passed else 'FAIL'} {label}: {seconds:.3f}s", flush=True)
    final = report(complete=True)
    print(f"{final['passed']}/{final['total']} AES cases passed")
    return 0 if final["passed"] == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
