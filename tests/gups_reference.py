#!/usr/bin/env python3
"""Check the actual GUPS stream, partitioned kernels, full verifier and admission."""
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import tempfile

HEADERS = r'''
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
struct uint2 { unsigned x, y; };
#define __constant__
#define __device__
#define __global__
#define __clzll __builtin_clzll
struct { unsigned x; } gridDim, blockDim, blockIdx, threadIdx;
static unsigned long long cas_calls, xor_calls;
static unsigned long long atomicCAS(unsigned long long* p, unsigned long long a, unsigned long long b) {
    ++cas_calls; auto old=*p; if(old==a)*p=b; return old;
}
static void atomicXor(unsigned* p, unsigned v) { ++xor_calls; *p^=v; }
'''

MAIN = r'''
static void require(bool ok) { if(!ok) { fprintf(stderr,"GUPS reference check failed\n"); abort(); } }
// Independent polynomial reduction in 128 bits, including the x^64 term.
static gups_word_t oracle_step(gups_word_t x) {
    unsigned __int128 y=static_cast<unsigned __int128>(x)*2;
    if(y>>64) y^=(static_cast<unsigned __int128>(1)<<64)|7;
    return static_cast<gups_word_t>(y);
}
int main() {
    size_t checks=0;
    gups_square_table(gups_squares);
    gups_word_t state=1;
    for(unsigned i=0;i<=1048576;++i) {
        if(i<512 || i%997==0) { require(gups_starts(i)==state); ++checks; }
        require(gups_step(state)==oracle_step(state));
        state=oracle_step(state);
    }
    // Independently exponentiate the 64-column transition matrix. This checks
    // offsets far above a small serial test, including all production offsets.
    for(gups_word_t offset : {0ULL,1ULL,63ULL,64ULL,65ULL,0xffffffffULL,0x100000000ULL,
                             0x123456789ULL,0x7fffffffffffffffULL}) {
        gups_word_t columns[64];
        for(int j=0;j<64;++j)columns[j]=oracle_step(1ULL<<j);
        gups_word_t value=1, remaining=offset;
        while(remaining) {
            if(remaining&1) { gups_word_t next=0; for(int j=0;j<64;++j)if(value&(1ULL<<j))next^=columns[j]; value=next; }
            gups_word_t squared[64]={};
            for(int j=0;j<64;++j)for(int k=0;k<64;++k)if(columns[j]&(1ULL<<k))squared[j]^=columns[k];
            memcpy(columns,squared,sizeof(columns)); remaining>>=1;
        }
        require(gups_starts(offset)==value); ++checks;
    }
    for(int n : {1,2,4,16,64,1024,8192}) {
        std::vector<GupsWord> exact(n), expected(n);
        for(int i=0;i<n;++i)exact[i].value=expected[i].value=i;
        state=1;
        for(int i=0;i<4*n;++i) { state=oracle_step(state); exact[state%static_cast<unsigned>(n)].value^=state; }
        gups_cpu_reference(expected.data(),n);
        require(gups_verify(exact.data(),expected.data(),n)==0); ++checks;
        // Execute the actual partitioned kernel bodies sequentially with host
        // atomic stubs. Real concurrent atomics are covered by the GPU test.
        for(int b : {1,3,32,256})for(int threads : {1,3,32,257,1024})for(bool cas : {false,true}) {
            GupsGeometry g; require(gups_geometry(n,b,threads,&g));
            gridDim.x=g.blocks; blockDim.x=b; cas_calls=xor_calls=0;
            std::vector<GupsWord> table(n); for(int i=0;i<n;++i)table[i].value=i;
            for(blockIdx.x=0;blockIdx.x<gridDim.x;++blockIdx.x)
                for(threadIdx.x=0;threadIdx.x<blockDim.x;++threadIdx.x)
                    if(cas)gups_cas_kernel(table.data(),n);else gups_xor_kernel(table.data(),n);
            require(cas_calls==(cas ? 4ULL*n : 0) && xor_calls==(cas ? 0 : 8ULL*n));
            require(gups_verify(table.data(),exact.data(),n)==0); ++checks;
        }
        std::vector<int> locations;
        if(n<=64)for(int i=0;i<n;++i)locations.push_back(i);
        else locations={0,n/2,n-1};
        for(int i : locations)for(int bit : {0,31,32,63}) {
            exact[i].value^=1ULL<<bit;
            require(gups_verify(exact.data(),expected.data(),n)==1); ++checks;
            exact[i].value^=1ULL<<bit;
        }
        for(int i=0;i<n;++i)exact[i].value^=1;
        require(gups_verify(exact.data(),expected.data(),n)==static_cast<size_t>(n)); ++checks;
    }
    printf("%zu sequence/jump/partition/atomic-count/full-table checks pass\n",checks);
}
'''


def functions(source):
    start = '// Host arithmetic and independent serial reference.'
    end = 'int main(int argc, char** argv) try {'
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError('GUPS test extraction boundaries changed')
    return source.split(start, 1)[1].split(end, 1)[0]


def geometry_program(source):
    prefix=source.split('int main(int argc, char** argv) try {',1)[1].split('    int device_id = 0;',1)[0]
    return HEADERS+functions(source)+'int main(int argc,char** argv) {'+prefix+r'''
    printf("%d,%d,%d,%d,%llu,%llu,%zu,%s\n",table_size,block_size,num_threads,
           geometry.blocks,geometry.actual_threads,geometry.total_updates,geometry.table_bytes,mode);
}
'''


def main():
    if not __debug__:
        raise RuntimeError('Run without Python optimization')
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--compiler',default='c++')
    ap.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]/'tier2/altis_gups/hip/altis_gups.hip')
    args=ap.parse_args();source=args.source.read_text()
    manifest=json.loads((args.source.parent/'params.json').read_text())
    triples=list(itertools.product(manifest['scaling_param']['values'],[256],[16384,65536,262144]))
    triples+=list(itertools.product([-2147483648,-1,0,1,2,3,1024,1073741824,2147483647],
                                   [0,1,3,256,2147483647],[-1,0,1,257,2147483647]))
    cases=[]
    for n,b,t in dict.fromkeys(triples):
        for mode in ['cas','xor']:
            valid=min(n,b,t)>0 and n&(n-1)==0
            want=None if not valid else ','.join(map(str,(n,b,t,(t+b-1)//b,((t+b-1)//b)*b,4*n,8*n,mode)))
            cases.append((dict(table_size=str(n),block_size=str(b),num_threads=str(t),atomic_mode=mode),want))
    cases += [({},'1048576,256,65536,256,65536,4194304,8388608,xor'),
              (dict(atomic_mode='CAS'),None),(dict(atomic_mode=''),None),
              (dict(num_updates='128'),None),(dict(num_updates=''),None)]
    clean={k:v for k,v in os.environ.items() if not k.startswith('BENCH_PARAM_')}
    with tempfile.TemporaryDirectory(prefix='gups-reference-') as folder:
        cpp=Path(folder)/'test.cpp';binary=Path(folder)/'test'
        for label,text in [('reference',HEADERS+functions(source)+MAIN),('geometry',geometry_program(source))]:
            cpp.write_text(text)
            subprocess.run([args.compiler,'-std=c++14','-O2','-fsanitize=address,undefined',
                            '-fno-sanitize-recover=all',str(cpp),'-o',str(binary)],check=True,timeout=60)
            if label=='reference':
                p=subprocess.run([str(binary)],text=True,capture_output=True,timeout=60)
                if p.returncode:raise AssertionError(p.stdout+p.stderr)
                print(p.stdout.strip());continue
            for values,want in cases:
                p=subprocess.run([str(binary)],env=dict(clean,**{'BENCH_PARAM_'+k:v for k,v in values.items()}),
                                 text=True,capture_output=True,timeout=30)
                if want is None:
                    assert p.returncode==1 and not p.stdout and ('GUPS requires' in p.stderr or 'Unsupported' in p.stderr),(values,p.stdout,p.stderr)
                else:
                    assert p.returncode==0 and not p.stderr and p.stdout.strip()==want,(values,want,p.stdout,p.stderr)
    print(f'{len(cases)} actual input/geometry/range cases pass under fatal sanitizers')


if __name__=='__main__':
    main()
