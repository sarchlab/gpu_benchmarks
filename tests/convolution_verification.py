#!/usr/bin/env python3
"""Check both complete measured separable-convolution stages and CPU fixtures."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

HOOK = r'''
    if(const char* stage=std::getenv("CONV_TEST_STAGE")) {
        const size_t total=(size_t)width*height;
        size_t at=std::strtoull(std::getenv("CONV_TEST_INDEX"),nullptr,10);
        size_t count=std::getenv("CONV_TEST_MANY")?12:1;
        if(count>total || at>total-count) std::abort();
        float* data=!std::strcmp(stage,"row")?d_temp:d_output;
        for(size_t j=0;j<count;++j) {
            float value;
            HIP_CHECK(hipMemcpy(&value,data+at+j,sizeof(float),hipMemcpyDeviceToHost));
            value=std::getenv("CONV_TEST_NAN")?NAN:value+1234.f;
            HIP_CHECK(hipMemcpy(data+at+j,&value,sizeof(float),hipMemcpyHostToDevice));
        }
        std::fprintf(stderr,"CONV_TEST_INJECT %s %zu %zu\n",stage,at,count);
    }
'''


def instrument(source):
    anchor = '    // Verify both measured stages after all timing, using the original'
    allocation = '        std::vector<float> row_ref(count), ref(count);'
    if source.count(anchor) != 1 or source.count(allocation) != 1:
        raise ValueError('expected measured-output and reference-allocation anchors')
    source = source.replace(anchor, HOOK + anchor)
    return source.replace(allocation, '''        if(std::getenv("CONV_TEST_ALLOC_FAIL")) {
            std::fprintf(stderr,"CONV_TEST_ALLOC_FAIL\\n"); throw std::bad_alloc();
        }
''' + allocation)


def host_reference(source, work, compiler):
    start = source.index('static void conv_separable_cpu(')
    end = source.index('// ---------------------------------------------------------------------------', start)
    program = '#include <vector>\n#include <cassert>\n#include <cstring>\n'
    program += source[start:end]
    program += r'''
int main() {
    // Asymmetric exact fixture fixes coefficient direction and both stage results.
    float in[]={1,2,3,4,5,6}, weights[]={1,2,3}, row[8], out[8];
    row[0]=out[0]=12345;row[7]=out[7]=-12345;
    const float erow[]={8,14,8,23,32,17}, eout[]={85,124,67,54,78,42};
    conv_separable_cpu(in,row+1,out+1,3,2,weights,1);
    assert(!std::memcmp(row+1,erow,sizeof(erow)));
    assert(!std::memcmp(out+1,eout,sizeof(eout)));
    assert(row[0]==12345 && out[0]==12345 && row[7]==-12345 && out[7]==-12345);
    const int shapes[][2]={{1,1},{3,2},{17,11},{129,65}};
    for(const auto& shape:shapes)for(int radius: {1,4,8,16,32}) {
        int w=shape[0],h=shape[1],n=w*h;
        std::vector<float> x(n),k(2*radius+1),r(n+2,12345),y(n+2,12345);
        // Small integers keep direct 2D and separable sums exactly representable.
        for(int i=0;i<n;++i)x[i]=(i*7)%13-6;
        for(int j=0;j<2*radius+1;++j)k[j]=j%5-2;
        const auto saved_x=x,saved_k=k;
        conv_separable_cpu(x.data(),r.data()+1,y.data()+1,w,h,k.data(),radius);
        assert(x==saved_x && k==saved_k);
        assert(r.front()==12345 && r.back()==12345 && y.front()==12345 && y.back()==12345);
        for(int i=0;i<h;++i)for(int j=0;j<w;++j) {
            float a=0,b=0;
            for(int dx=-radius;dx<=radius;++dx)if(j+dx>=0 && j+dx<w) {
                a+=x[i*w+j+dx]*k[dx+radius];
                for(int dy=-radius;dy<=radius;++dy)if(i+dy>=0 && i+dy<h)
                    b+=x[(i+dy)*w+j+dx]*k[dx+radius]*k[dy+radius];
            }
            assert(r[1+i*w+j]==a && y[1+i*w+j]==b);
        }
    }
    // Normalized constant image: zero padding attenuates each edge independently.
    std::vector<float> x(15,1),r(15),y(15);float k[]={.25f,.5f,.25f};
    conv_separable_cpu(x.data(),r.data(),y.data(),5,3,k,1);
    for(int i=0;i<3;++i)for(int j=0;j<5;++j) {
        float horizontal=(j==0 || j==4)?.75f:1.f;
        assert(r[i*5+j]==horizontal);
        assert(y[i*5+j]==horizontal*((i==0 || i==2)?.75f:1.f));
    }
}
'''
    path = work / 'reference.cpp'
    path.write_text(program)
    binary = work / 'reference'
    subprocess.run([compiler, '-O2', '-std=c++14', '-Wall', '-Wextra', '-Werror',
                    '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                    str(path), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=60)
    print('PASS CPU: asymmetric known answer, 20 direct-2D fixtures, zero padding, guards and immutable input', flush=True)


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
    relative = Path('tier2/cuda_convolution_separable/hip')
    src = args.source / relative
    source = (src / 'cuda_convolution_separable.hip').read_text()
    if args.host_only:
        host_reference(source, args.work_dir, args.compiler)
        return 0
    inputs = [relative / p for p in ('cuda_convolution_separable.hip', 'Makefile', 'params.json')]
    inputs += [Path('common') / p for p in ('bench_timing.h', 'bench_warmup.h', 'hip_cuda_compat.h')]
    hashes = {str(p): hashlib.sha256((args.source / p).read_bytes()).hexdigest() for p in inputs}
    binaries = {}
    for variant in ('production', 'faults'):
        root = args.work_dir / variant
        dst = root / relative
        dst.mkdir(parents=True)
        shutil.copytree(args.source / 'common', root / 'common')
        for name in ('Makefile', 'params.json'):
            shutil.copy2(src / name, dst / name)
        (dst / 'cuda_convolution_separable.hip').write_text(source if variant == 'production' else instrument(source))
        with (root / 'build.log').open('w') as log:
            subprocess.run(['make', 'PLATFORM=' + args.platform], cwd=dst, stdout=log, stderr=subprocess.STDOUT, check=True)
        binaries[variant] = dst / 'cuda_convolution_separable'
    if args.compile_only:
        print('Production and actual-device-fault variants compiled; hardware checks not run')
        return 0
    clean_env = {k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_', 'CONV_TEST_'))}
    # variant,width,height,radius,warm,stage,index,count,NaN,allocation failure
    cases = []
    for warm in ('false', 'true'):
        for w,h,r in ((1,1,32), (17,11,4), (129,65,32), (640,1024,8)):
            cases.append(('production',w,h,r,warm,None,None,0,False,False))
        for stage in ('row', 'column'):
            for at in (0,511,512,640*512,640*1024-1):
                cases.append(('faults',640,1024,8,warm,stage,at,1,False,False))
            cases.append(('faults',640,1024,8,warm,stage,640*1024-1,1,True,False))
            cases.append(('faults',640,1024,8,warm,stage,640*1024-12,12,False,False))
        cases.append(('faults',640,1024,8,warm,None,None,0,False,True))
    cases.append(('production',16384,8192,32,'false',None,None,0,False,False))
    if args.sweep:
        m = json.loads((src / 'params.json').read_text())
        p = {x['name']:x['values'] for x in m['non_scaling_params']}
        cases += [('production',w,h,r,'true',None,None,0,False,False)
                  for w,h,r in itertools.product(reversed(m['scaling_param']['values']),
                                                reversed(p['height']),reversed(p['radius']))]
    results = []

    def report(complete=False):
        data = dict(complete=complete, source_sha256=hashes,
                    binary_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in binaries.items()},
                    expected=len(cases), total=len(results), passed=sum(x['passed'] for x in results), cases=results)
        (args.work_dir / 'report.json').write_text(json.dumps(data, indent=2) + '\n')
        return data

    for variant,w,h,r,warm,stage,at,count,nan,allocation in cases:
        label = f'{len(results):03d}-{variant}-{w}-{h}-{r}-{warm}-{stage}-{at}-{count}-{nan}-{allocation}'
        env = dict(clean_env, BENCH_PARAM_width=str(w), BENCH_PARAM_height=str(h),
                   BENCH_PARAM_radius=str(r), BENCH_PARAM_vector_add_warmup=warm)
        if stage:
            env.update(CONV_TEST_STAGE=stage, CONV_TEST_INDEX=str(at))
        if count == 12:
            env['CONV_TEST_MANY'] = '1'
        if nan:
            env['CONV_TEST_NAN'] = '1'
        if allocation:
            env['CONV_TEST_ALLOC_FAIL'] = '1'
        start = time.monotonic()
        try:
            p = subprocess.run([str(binaries[variant])], env=env, capture_output=True, text=True, timeout=60)
            stdout,stderr,code = p.stdout,p.stderr,p.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or b'').decode(errors='replace')
            stderr = (exc.stderr or b'').decode(errors='replace')
            code = 'timeout'
        seconds = time.monotonic() - start
        (args.work_dir / (label + '.stdout')).write_text(stdout)
        (args.work_dir / (label + '.stderr')).write_text(stderr)
        bad = bool(stage or allocation)
        passed = code == (1 if bad else 0) and seconds < 60
        if allocation:
            passed &= 'CONV_TEST_ALLOC_FAIL' in stderr and 'FAIL: complete CPU reference unavailable:' in stderr
        else:
            passed &= f"{'FAIL' if bad else 'PASS'}: {count} mismatches in {2*w*h} values ({w*h} pixels per stage)" in stderr
            passed &= stderr.count('Mismatch ') == min(count,10)
        if stage:
            passed &= f'CONV_TEST_INJECT {stage} {at} {count}' in stderr
            passed &= f'Mismatch {stage} at ({at//w},{at%w}):' in stderr
        if bad:
            passed &= 'PASS:' not in stderr
        try:
            row,col,summary = [json.loads(line) for line in stdout.splitlines()]
            for k,name in ((row,'convolution_row_kernel'),(col,'convolution_col_kernel')):
                passed &= k['type']=='kernel' and k['name']==name and k['iter']==0
                passed &= k['params']==dict(width=w,height=h,radius=r,verification='full_cpu_v1')
                passed &= all(math.isfinite(k[f]) and k[f]>=0 for f in ('time_ms','host_time_ms'))
            passed &= summary['type']=='summary' and math.isfinite(summary['total_time_ms'])
            passed &= abs(summary['total_time_ms']-row['time_ms']-col['time_ms'])<.000003
        except (ValueError,KeyError,TypeError):
            passed = False
        results.append(dict(label=label,variant=variant,width=w,height=h,radius=r,warm=warm,stage=stage,index=at,
                            count=count,nan=nan,allocation=allocation,exit=code,seconds=seconds,passed=bool(passed)))
        report()
        print(f"{'PASS' if passed else 'FAIL'} {label}: {seconds:.3f}s", flush=True)
    final = report(complete=True)
    print(f"{final['passed']}/{final['total']} complete convolution cases passed")
    return 0 if final['passed']==len(cases) else 1


if __name__ == '__main__':
    raise SystemExit(main())
