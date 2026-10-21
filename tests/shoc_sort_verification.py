#!/usr/bin/env python3
"""Check full measured SHOC sort key/value conservation and ordering."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

HOOK = r'''
    if (const char* kind=std::getenv("SORT_TEST_KIND")) {
        const size_t at=std::strtoull(std::getenv("SORT_TEST_INDEX"),nullptr,10);
        if (at>=(size_t)size) std::abort();
        const uint invalid=~0u;
        if (!std::strcmp(kind,"zero")) HIP_CHECK(hipMemset(dKeys,0,bytes));
        else if (!std::strcmp(kind,"key")) HIP_CHECK(hipMemcpy(dKeys+at,&invalid,sizeof(uint),hipMemcpyHostToDevice));
        else if (!std::strcmp(kind,"value")) HIP_CHECK(hipMemcpy(dVals+at,&invalid,sizeof(uint),hipMemcpyHostToDevice));
        else if (!std::strcmp(kind,"many")) {
            for (size_t i=0;i<(size_t)size;++i) HIP_CHECK(hipMemcpy(dVals+i,&invalid,sizeof(uint),hipMemcpyHostToDevice));
        } else if (!std::strcmp(kind,"duplicate")) {
            const size_t other=at+1<(size_t)size ? at+1 : at-1;
            HIP_CHECK(hipMemcpy(dKeys+at,dKeys+other,sizeof(uint),hipMemcpyDeviceToDevice));
            HIP_CHECK(hipMemcpy(dVals+at,dVals+other,sizeof(uint),hipMemcpyDeviceToDevice));
        } else if (!std::strcmp(kind,"association")) {
            const size_t other=at+1<(size_t)size ? at+1 : at-1;
            uint keys[2], values[2];
            HIP_CHECK(hipMemcpy(keys,dKeys+at,sizeof(uint),hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(keys+1,dKeys+other,sizeof(uint),hipMemcpyDeviceToHost));
            if (keys[0]==keys[1]) std::abort(); // require an observable wrong association
            HIP_CHECK(hipMemcpy(values,dVals+at,sizeof(uint),hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(values+1,dVals+other,sizeof(uint),hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(dVals+at,values+1,sizeof(uint),hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(dVals+other,values,sizeof(uint),hipMemcpyHostToDevice));
        } else std::abort();
        std::fprintf(stderr,"SORT_TEST_INJECT %s %zu\n",kind,at);
    }
'''


def instrument(source):
    replacements={
        "    // Correctness check on the last pass's result.": HOOK+"    // Correctness check on the last pass's result.",
        '        for (int bit = 0; bit < SORT_BITS; bit += 4)': '        if (!std::getenv("SORT_TEST_SKIP"))\n        for (int bit = 0; bit < SORT_BITS; bit += 4)',
        '(uint*)malloc(bytes);\n    bool sorted_ok': '(std::getenv("SORT_TEST_NO_KEYS") ? nullptr : (uint*)malloc(bytes));\n    bool sorted_ok',
        '(unsigned char*)calloc(size, sizeof(unsigned char));': '(std::getenv("SORT_TEST_NO_SEEN") ? nullptr : (unsigned char*)calloc(size, sizeof(unsigned char)));',
    }
    for old,new in replacements.items():
        if source.count(old)!=1: raise ValueError('Sort measured/checker anchor missing: '+old)
        source=source.replace(old,new)
    return source


def host_checks(root,work,compiler):
    source=(root/'tier2/shoc_sort/hip/shoc_sort.hip').read_text()
    start='static bool verifySort('; end='// ---------------------------------------------------------------------------\n// Main'
    if source.count(start)!=1 or source.count(end)!=1: raise ValueError('Sort checker anchors missing')
    prefix='''#include <cstdio>
#include <cstdlib>
typedef unsigned int uint;
static bool fail_seen=false;
static void* test_calloc(size_t n,size_t size) { return fail_seen ? nullptr : std::calloc(n,size); }
#define calloc test_calloc
'''
    text=prefix+start+source.split(start)[1].split(end)[0]+'\n#undef calloc\n'
    text+=(root/'tests/shoc_sort_verification_test.cpp').read_text()
    path=work/'checker.cpp';path.write_text(text);binary=work/'checker'
    subprocess.run([compiler,'-O3','-std=c++14','-Wall','-Wextra','-Werror','-fsanitize=address,undefined',
                    '-fno-sanitize-recover=all',str(path),'-o',str(binary)],check=True)
    p=subprocess.run([str(binary)],capture_output=True,text=True,timeout=60)
    (work/'host.stdout').write_text(p.stdout);(work/'host.stderr').write_text(p.stderr)
    assert p.returncode==0,p.stderr
    records=re.split(r'SORT_CASE (\d+) (\d+) (-?\d+)\n',p.stderr)
    assert not records[0] and len(records)>1
    for i in range(1,len(records),4):
        case,n,expected,err=records[i:i+4];expected=int(expected)
        if expected==-2:
            assert 'Sort verification unavailable' in err and 'invalid pairs' not in err
        else:
            m=re.search(r'Sort verification: (\d+) invalid pairs in (\d+) outputs',err)
            assert m and m[2]==n and (expected<0 or int(m[1])==expected),(case,err)
            assert err.count('Sort mismatch at ')==min(int(m[1]),10),(case,err)
    count=int(re.search(r'PASS: (\d+)',p.stdout)[1]);assert count==(len(records)-1)//4
    print(p.stdout,end='')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--work-dir',type=Path,required=True,help='New, nonexistent directory')
    ap.add_argument('--host-only',action='store_true');ap.add_argument('--compiler',default='c++')
    ap.add_argument('--platform',choices=('cuda','rocm'));ap.add_argument('--compile-only',action='store_true')
    ap.add_argument('--sweep',action='store_true');args=ap.parse_args()
    if not args.host_only and not args.platform:ap.error('--platform required')
    root,work=args.source.resolve(),args.work_dir.resolve();work.mkdir(parents=True,exist_ok=False)
    if args.host_only:host_checks(root,work,args.compiler);return 0
    rel=Path('tier2/shoc_sort/hip');src=root/rel;source=(src/'shoc_sort.hip').read_text()
    inputs=[rel/n for n in ('shoc_sort.hip','Makefile','params.json')]
    inputs += [Path('common')/n for n in ('bench_timing.h','bench_warmup.h','hip_cuda_compat.h')]
    hashes={str(p):hashlib.sha256((root/p).read_bytes()).hexdigest() for p in inputs};binaries={}
    for variant in ('production','faults'):
        dst=work/variant/rel;dst.mkdir(parents=True);shutil.copytree(root/'common',work/variant/'common')
        for name in ('Makefile','params.json'):shutil.copy2(src/name,dst/name)
        (dst/'shoc_sort.hip').write_text(source if variant=='production' else instrument(source))
        with (work/variant/'build.log').open('w') as log:
            subprocess.run(['make','PLATFORM='+args.platform],cwd=dst,stdout=log,stderr=subprocess.STDOUT,check=True)
        binaries[variant]=dst/'shoc_sort'
    if args.compile_only:print('Compiled production and measured-output faults; GPU tests not run');return 0
    clean={k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','SORT_TEST_'))}
    cases=[]
    for warm in ('false','true'):
        cases += [('production',n,warm,'healthy',0) for n in (1024,2048,3072,32768)]
        cases += [('faults',32768,warm,kind,at) for kind,at in itertools.product(
            ('key','value','duplicate','association'),(0,16384,32767))]
        cases += [('faults',1024,warm,kind,0) for kind in ('zero','many','skip','no_keys','no_seen','healthy')]
    if args.sweep:
        sizes=json.loads((src/'params.json').read_text())['scaling_param']['values']
        cases += [('production',n,warm,'healthy',0) for n,warm in itertools.product(sizes,('false','true'))]
    results=[]
    def report(complete=False):
        data=dict(complete=complete,expected=len(cases),total=len(results),passed=sum(x['passed'] for x in results),
                  source_sha256=hashes,binary_sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in binaries.items()},cases=results)
        (work/'report.json').write_text(json.dumps(data,indent=2)+'\n');return data
    for variant,n,warm,kind,at in cases:
        label=f'{len(results):03d}-{variant}-{n}-{warm}-{kind}-{at}'
        env=dict(clean,BENCH_PARAM_size=str(n),BENCH_PARAM_vector_add_warmup=warm)
        if kind in ('key','value','duplicate','association','zero','many'):env.update(SORT_TEST_KIND=kind,SORT_TEST_INDEX=str(at))
        if kind in ('skip','no_keys','no_seen'):env['SORT_TEST_'+kind.upper()]='1'
        start=time.monotonic()
        try:
            p=subprocess.run([str(binaries[variant])],env=env,capture_output=True,text=True,timeout=60)
            out,err,code=p.stdout,p.stderr,p.returncode
        except subprocess.TimeoutExpired as ex:
            out=(ex.stdout or b'').decode(errors='replace');err=(ex.stderr or b'').decode(errors='replace');code='timeout'
        seconds=time.monotonic()-start
        (work/(label+'.stdout')).write_text(out);(work/(label+'.stderr')).write_text(err)
        bad=kind!='healthy';passed=code==(1 if bad else 0) and seconds<60
        passed &= ('Correctness: FAIL' if bad else 'Correctness: PASS') in err
        if kind in ('no_keys','no_seen'):
            passed &= 'Sort verification unavailable' in err and 'invalid pairs' not in err
        else:
            m=re.search(r'Sort verification: (\d+) invalid pairs in (\d+) outputs',err)
            passed &= bool(m) and int(m[2])==n
            if m:
                errors=int(m[1]);passed &= err.count('Sort mismatch at ')==min(errors,10)
                if kind=='healthy':passed &= errors==0
                elif kind in ('value','duplicate'):passed &= errors==1
                elif kind=='association':passed &= errors==2
                elif kind=='key':passed &= errors==(1 if at==n-1 else 2)
                elif kind=='many':passed &= errors==n
                else:passed &= errors>0
        if 'SORT_TEST_KIND' in env:passed &= f'SORT_TEST_INJECT {kind} {at}' in err
        try:
            records=[json.loads(line) for line in out.splitlines()];kernels,summary=records[:-1],records[-1]
            passed &= len(kernels)==5 and [r['name'] for r in kernels]==['radixSortBlocks','findRadixOffsets','scan','vectorAddUniform4','reorderData']
            for r in kernels:
                passed &= r['type']=='kernel' and r['iter']==0 and r['params']==dict(size=n,sort_block_size=128,scan_block_size=256,verification='full_pairs_v1')
                passed &= all(math.isfinite(r[k]) and r[k]>=0 for k in ('time_ms','host_time_ms'))
            passed &= summary['type']=='summary' and math.isfinite(summary['total_time_ms'])
            passed &= abs(summary['total_time_ms']-sum(r['time_ms'] for r in kernels))<.000004
        except (ValueError,KeyError,TypeError,IndexError):passed=False
        results.append(dict(label=label,variant=variant,n=n,warm=warm,kind=kind,index=at,exit=code,seconds=seconds,passed=bool(passed)))
        report();print(f"{'PASS' if passed else 'FAIL'} {label}: {seconds:.3f}s",flush=True)
    r=report(complete=True);print(f"{r['passed']}/{r['total']} complete measured sort-pair cases passed")
    return 0 if r['passed']==len(cases) else 1


if __name__=='__main__':raise SystemExit(main())
