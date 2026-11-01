#!/usr/bin/env python3
"""Verify all measured Parboil SGEMM outputs with ordered full CPU references."""
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
    if(const char* text=std::getenv("PARBOIL_SGEMM_TEST_INDEX")) {
        size_t at=std::strtoull(text,nullptr,10),count=std::getenv("PARBOIL_SGEMM_TEST_MANY")?12:1;
        const size_t total=(size_t)N*N;
        if(count>total || at>total-count) std::abort();
        for(size_t i=at;i<at+count;++i) {
            float value;
            HIP_CHECK(hipMemcpy(&value,d_C+i,sizeof(float),hipMemcpyDeviceToHost));
            value=std::getenv("PARBOIL_SGEMM_TEST_NAN")?NAN:value+123456.f;
            HIP_CHECK(hipMemcpy(d_C+i,&value,sizeof(float),hipMemcpyHostToDevice));
        }
        std::fprintf(stderr,"PARBOIL_SGEMM_TEST_INJECT %zu %zu\n",at,count);
    }
'''


def instrument(source):
    anchor = '    // Verify every measured output after timing; no skipped sizes or GPU rerun.'
    reference = '        gemm_cpu(h_A, h_B, h_ref, N, N, N, alpha, beta);'
    if source.count(anchor) != 1 or source.count(reference) != 1:
        raise ValueError('measured verification anchors missing')
    source = source.replace(anchor, HOOK + anchor)
    return source.replace(reference, '''        if(std::getenv("PARBOIL_SGEMM_TEST_REFERENCE_FAIL")) {
            std::fprintf(stderr,"PARBOIL_SGEMM_TEST_REFERENCE_FAIL\\n"); throw std::bad_alloc();
        }
''' + reference)

def host_tests(root, work, compiler):
    relative=Path("tier2/parboil_sgemm/hip/cpu_reference.h")
    original=(root/relative).read_text()
    for label,fail_at in (("normal",None),("no-threads",1),("partial-threads",2)):
        path=work/label
        (path/relative.parent).mkdir(parents=True)
        (path/"tests").mkdir()
        header=original
        if fail_at:
            old="threads.emplace_back(compute, worker);"
            if header.count(old)!=1:raise ValueError("thread fallback hook missing")
            header=header.replace(old,"{ if (worker == "+str(fail_at)+") throw std::runtime_error(\"test thread failure\"); threads.emplace_back(compute, worker); }")
        (path/relative).write_text(header)
        shutil.copy2(root/"tests/parboil_sgemm_reference_test.cpp",path/"tests/parboil_sgemm_reference_test.cpp")
        binary=path/"reference"
        subprocess.run([compiler,"-O3","-std=c++14","-pthread","-Wall","-Wextra","-Werror",
                        "-fsanitize=address,undefined","-fno-sanitize-recover=all",
                        str(path/"tests/parboil_sgemm_reference_test.cpp"),"-o",str(binary)],check=True)
        subprocess.run([str(binary)],check=True,timeout=60)
        print("PASS host "+label,flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--work-dir', type=Path, required=True, help='New, nonexistent directory')
    ap.add_argument('--host-only', action='store_true')
    ap.add_argument('--compiler', default='c++')
    ap.add_argument('--platform', choices=('cuda', 'rocm'))
    ap.add_argument('--compile-only', action='store_true')
    ap.add_argument('--sweep', action='store_true')
    args = ap.parse_args()
    if not args.host_only and not args.platform:
        ap.error('--platform required')
    args.source, args.work_dir = args.source.resolve(), args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=False)
    if args.host_only:
        host_tests(args.source, args.work_dir, args.compiler)
        return 0
    relative = Path('tier2/parboil_sgemm/hip')
    src = args.source / relative
    source = (src / 'parboil_sgemm.hip').read_text()
    inputs = [relative / n for n in ('parboil_sgemm.hip', 'cpu_reference.h', 'Makefile', 'params.json')]
    inputs += [Path('common_hip') / n for n in ('bench_timing.h', 'bench_warmup.h', 'hip_cuda_compat.h')]
    hashes = {str(p):hashlib.sha256((args.source / p).read_bytes()).hexdigest() for p in inputs}
    binaries = {}
    for variant in ('production', 'faults'):
        root = args.work_dir / variant
        dst = root / relative
        dst.mkdir(parents=True)
        shutil.copytree(args.source / 'common_hip', root / 'common_hip')
        for n in ('Makefile', 'params.json', 'cpu_reference.h'):
            shutil.copy2(src / n, dst / n)
        (dst / 'parboil_sgemm.hip').write_text(source if variant == 'production' else instrument(source))
        with (root / 'build.log').open('w') as log:
            subprocess.run(['make', 'PLATFORM=' + args.platform], cwd=dst, stdout=log, stderr=subprocess.STDOUT, check=True)
        binaries[variant] = dst / 'parboil_sgemm'
    if args.compile_only:
        print('Production and measured-device fault variants compiled; GPU checks not run')
        return 0
    clean = {k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_', 'PARBOIL_SGEMM_TEST_'))}
    # variant,size,warm,first corruption,count,NaN,unavailable reference
    cases = []
    for warm in ('false', 'true'):
        for n in (1, 17, 33, 257, 640):
            cases.append(('production',n,warm,None,0,False,False))
        for index in (0, 256*640, 640**2-1):
            cases.append(('faults',640,warm,index,1,False,False))
        cases += [('faults',640,warm,640**2-1,1,True,False),
                  ('faults',640,warm,640**2-12,12,False,False),
                  ('faults',640,warm,None,0,False,True)]
    cases.append(('production',8192,'false',None,0,False,False))
    if args.sweep:
        manifest = json.loads((src / 'params.json').read_text())
        cases += [('production',n,'true',None,0,False,False) for n in reversed(manifest['scaling_param']['values'])]
    results = []

    def report(complete=False):
        result = dict(complete=complete, source_sha256=hashes,
                      binary_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in binaries.items()},
                      expected=len(cases), total=len(results), passed=sum(x['passed'] for x in results), cases=results)
        (args.work_dir / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
        return result

    for variant,n,warm,index,count,nan,unavailable in cases:
        label = f'{len(results):03d}-{variant}-{n}-{warm}-{index}-{count}-{nan}-{unavailable}'
        env = dict(clean, BENCH_PARAM_size=str(n), BENCH_PARAM_vector_add_warmup=warm)
        if index is not None:
            env['PARBOIL_SGEMM_TEST_INDEX'] = str(index)
        if count == 12:
            env['PARBOIL_SGEMM_TEST_MANY'] = '1'
        if nan:
            env['PARBOIL_SGEMM_TEST_NAN'] = '1'
        if unavailable:
            env['PARBOIL_SGEMM_TEST_REFERENCE_FAIL'] = '1'
        start = time.monotonic()
        try:
            p = subprocess.run([str(binaries[variant])], env=env, capture_output=True, text=True, timeout=60)
            out,err,code = p.stdout,p.stderr,p.returncode
        except subprocess.TimeoutExpired as ex:
            out = (ex.stdout or b'').decode(errors='replace')
            err = (ex.stderr or b'').decode(errors='replace')
            code = 'timeout'
        seconds = time.monotonic() - start
        (args.work_dir / (label + '.stdout')).write_text(out)
        (args.work_dir / (label + '.stderr')).write_text(err)
        bad = bool(count or unavailable)
        passed = code == (1 if bad else 0) and seconds < 60
        if unavailable:
            passed &= 'PARBOIL_SGEMM_TEST_REFERENCE_FAIL' in err and 'FAIL: complete CPU reference unavailable:' in err
        else:
            passed &= f"{'FAIL' if count else 'PASS'}: {count} mismatches in {n*n} values" in err
            passed &= err.count('Mismatch at ') == min(count,10)
        if count:
            passed &= f'PARBOIL_SGEMM_TEST_INJECT {index} {count}' in err and f'Mismatch at {index}:' in err
        if bad:
            passed &= 'PASS:' not in err
        try:
            kernel,summary = [json.loads(line) for line in out.splitlines()]
            passed &= kernel['type']=='kernel' and kernel['name']=='sgemm_kernel' and kernel['iter']==0
            passed &= kernel['params']==dict(size=n,verification='full_cpu_v1')
            passed &= all(math.isfinite(kernel[f]) and kernel[f]>=0 for f in ('time_ms','host_time_ms'))
            passed &= summary['type']=='summary' and math.isfinite(summary['total_time_ms'])
            passed &= abs(summary['total_time_ms']-kernel['time_ms'])<.000002
        except (ValueError,KeyError,TypeError):
            passed = False
        results.append(dict(label=label,variant=variant,size=n,warm=warm,index=index,count=count,nan=nan,
                            unavailable=unavailable,exit=code,seconds=seconds,passed=bool(passed)))
        report()
        print(f"{'PASS' if passed else 'FAIL'} {label}: {seconds:.3f}s", flush=True)
    final = report(complete=True)
    print(f"{final['passed']}/{final['total']} Parboil SGEMM complete measured-result cases passed")
    return 0 if final['passed']==len(cases) else 1


if __name__ == '__main__':
    raise SystemExit(main())
