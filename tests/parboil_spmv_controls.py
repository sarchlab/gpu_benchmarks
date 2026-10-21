#!/usr/bin/env python3
"""Check actual SpMV control resolution, geometry and JSON without a GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def between(source, start, end):
    for anchor in (start, end):
        count = source.count(anchor)
        if count != 1:
            raise ValueError(f'expected one occurrence of source anchor {anchor!r}, found {count}')
    return source.split(start, 1)[1].split(end, 1)[0]


def host_source(source):
    parse = 'static int parseIntParam' + between(source, 'static int parseIntParam', '// ---------------------------------------------------------------------------\n// Constant memory')
    geometry = 'static void compute_active_thread' + between(source, 'static void compute_active_thread', '// ---------------------------------------------------------------------------\n// COO -> JDS')
    resolve = '    int num_rows   = 131072;' + between(source, '    int num_rows   = 131072;', '    // ---- Build the matrix')
    record = '    unsigned int grid, block;' + between(source, '    unsigned int grid, block;', '    // One pass = one spmv_jds launch.')
    return '''#include <cstdio>
#include <cstdlib>
#include <cstring>
#define WARP_SIZE 32
''' + parse + geometry + '''
int main(int argc, char** argv) {
''' + resolve + '''
    struct {int major,minor,warpSize,multiProcessorCount;} prop;
    if (sscanf(getenv("SPMV_TEST_DEVICE"),"%d,%d,%d,%d",&prop.major,&prop.minor,&prop.warpSize,&prop.multiProcessorCount)!=4) return 2;
    const int dim=num_rows,nzcnt_len=(num_rows+31)/32,nz=11,len=22,depth=3;
''' + record + '''
    printf("{%s}\\n",params);
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', default='c++')
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1]/'tier2/parboil_spmv/hip/parboil_spmv.hip')
    args = parser.parse_args()
    # Expected resolved controls are literal, independent of production parsing.
    cases = [([], {}, 131072,80,0,42),
             (['--num_rows','2048','--avg_nnz_per_row','20'], {},2048,20,0,42),
             (['--num_rows','160000'],{},160000,80,0,42),
             *[(['--num_rows',str(n)],{},n,80,0,42) for n in (768,1024,1536,1537)],
             (['--block_size','128','--seed','7'],{},131072,80,128,7),
             (['--block_size','128'],{'block_size':'256'},131072,80,256,42),
             (['--block_size','128'],{'block_size':'0'},131072,80,0,42),
             (['--block_size','128'],{'block_size':'-17'},131072,80,-17,42),
             (['--block_size','0','--seed','0'],{},131072,80,0,42),
             (['--block_size','-1','--seed','-7'],{},131072,80,0,42),
             (['--seed','7','--seed','19'],{},131072,80,0,7),
             (['--seed'],{},131072,80,0,42),
             (['--num_rows','8192','--avg_nnz_per_row','40'],{'num_rows':'4096','avg_nnz_per_row':'20'},4096,20,0,42),
             ([],{'block_size':'2147483647'},131072,80,2147483647,42),
             ([],{'block_size':'-2147483648'},131072,80,-2147483648,42)]
    devices = [(1,1,32,1),(1,2,32,1),(9,0,32,114),(0,0,64,120)]
    clean = {k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','SPMV_TEST_'))}
    with tempfile.TemporaryDirectory(prefix='spmv-controls-') as temp:
        root = Path(temp); cpp=root/'controls.cpp'; binary=root/'controls'
        try:
            cpp.write_text(host_source(args.source.read_text()))
        except ValueError as exc:
            raise ValueError(f'{args.source}: {exc}') from exc
        subprocess.run([args.compiler,'-std=c++14','-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all',str(cpp),'-o',str(binary)],check=True,timeout=60)
        for device in devices:
            for argv, overrides, n, nnz, requested, seed in cases:
                env = dict(clean,SPMV_TEST_DEVICE=','.join(map(str,device)))
                env.update({'BENCH_PARAM_'+k:v for k,v in overrides.items()})
                p = subprocess.run([str(binary),*argv],env=env,capture_output=True,text=True,timeout=30)
                if p.returncode:raise RuntimeError(p.stderr)
                major,minor,warp,sm=device
                threshold = (768 if minor<2 else 1024) if major==1 else 1536
                rows=(n+31)//32*32
                block=requested if requested>0 else threshold//8 if rows>sm*threshold else 32
                expected=dict(num_rows=n,avg_nnz_per_row=nnz,nnz=11,jds_len=22,jds_depth=3,
                              block_size=block,requested_block_size=requested,grid_size=(rows+block-1)//block if requested<=0 else (n+block-1)//block,
                              seed=seed,launch_policy='block_override_v1' if requested>0 else 'auto_v1')
                actual=json.loads(p.stdout)
                if actual!=expected:raise AssertionError((device,argv,overrides,actual,expected))
    print(f'{len(cases)*len(devices)} resolved-control/geometry records pass (host simulation; no GPU qualification)')
    return 0


if __name__ == '__main__':raise SystemExit(main())
