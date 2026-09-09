#!/usr/bin/env python3
"""Check the actual NW initialization, full CPU verifier and index/score admission."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

INT_MAX = (1 << 31) - 1
# Comma-separated integer entries from Rodinia3.1 pin366b283 cuda/nw/needle.cu.
TABLE_SHA256 = '40e3f2fbde3daa5d83aacb4dab8f14a278ff6a2dacd886a015d3e99095bd3320'


def host_functions(source):
    start = 'static const int blosum62[24][24]'
    end = '// ---------------------------------------------------------------------------\n// Kernel 1:'
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError('NW host function boundaries changed')
    functions = start + source.split(start, 1)[1].split(end, 1)[0]
    table = functions.split('};', 1)[0].split('=', 1)[1]
    values = re.findall(r'-?\d+', table)
    if len(values) != 576 or hashlib.sha256(','.join(values).encode()).hexdigest() != TABLE_SHA256:
        raise ValueError('BLOSUM62 differs from the pinned table')
    return functions


HEADERS = '''#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
'''

TEST_MAIN = r'''
static void require(bool ok) { if (!ok) { fprintf(stderr,"NW reference check failed\n"); abort(); } }
int main() {
    unsigned checks = 0;
    // Independent anti-diagonal, int64 dense oracle. The production verifier
    // traverses rows using two int vectors; no measured cell drives its state.
    for (int n : {1,2,3,7,16,33,64}) for (int penalty : {1,2,5,10}) {
        const int cols = n+1;
        std::vector<int> rows(cols), columns(cols), reference(cols*cols), initial(cols*cols);
        nw_initialize(n,penalty,rows.data(),columns.data(),reference.data(),initial.data());
        srand(7);
        for (int i=1;i<=n;++i) require(rows[i]==rand()%10+1);
        for (int j=1;j<=n;++j) require(columns[j]==rand()%10+1);
        require(rows[0]==0 && columns[0]==0);
        for (int i=0;i<=n;++i) for (int j=0;j<=n;++j) {
            require(initial[i*cols+j]==(i==0 ? -j*penalty : j==0 ? -i*penalty : 0));
            require(reference[i*cols+j]==(i && j ? blosum62[rows[i]][columns[j]] : 0));
        }
        ++checks;
        std::vector<long long> exact(cols*cols);
        for (int i=0;i<=n;++i) exact[i]=exact[i*cols]=-i*penalty;
        for (int d=2;d<=2*n;++d) for (int i=1;i<=n;++i) {
            int j=d-i;if(j<1 || j>n)continue;
            exact[i*cols+j]=std::max(exact[(i-1)*cols+j-1]+blosum62[rows[i]][columns[j]],
                std::max(exact[(i-1)*cols+j]-penalty,exact[i*cols+j-1]-penalty));
        }
        std::vector<int> actual(exact.begin(),exact.end());
        require(nw_verify(actual.data(),rows.data(),columns.data(),n,penalty)==0);++checks;
        std::vector<int> locations;
        if(n<=3)for(int k=0;k<cols*cols;++k)locations.push_back(k);
        else locations={0,n,n*cols,cols+1,(n/2)*cols+n/2,n*cols+n-1,cols*cols-1};
        for(int k:locations) {
            ++actual[k];require(nw_verify(actual.data(),rows.data(),columns.data(),n,penalty)==1);
            --actual[k];++checks;
        }
        for(int& value:actual)++value;
        require(nw_verify(actual.data(),rows.data(),columns.data(),n,penalty)==actual.size());++checks;
    }
    // Analytic one-character alignments span every substitution table entry.
    for(int a=0;a<24;++a)for(int b=0;b<24;++b)for(int p:{1,2,10}) {
        int rows[2]={0,a},columns[2]={0,b};
        int actual[4]={0,-p,-p,std::max(blosum62[a][b],-2*p)};
        require(nw_verify(actual,rows,columns,1,p)==0);++checks;
    }
    printf("%u initialization/analytic/full-reference/corruption checks pass\n",checks);
}
'''


def geometry_source(source):
    parser = source.split('static int parseIntParam(', 1)[1].split('static const int blosum62', 1)[0]
    prefix = source.split('int main(int argc, char** argv) try {', 1)[1].split('    int device_id = 0;', 1)[0]
    return HEADERS + host_functions(source) + 'static int parseIntParam(' + parser + 'int main(int argc,char** argv) {' + prefix + r'''
    printf("%d,%d,%d,%d,%d,%d,%zu,%zu\n",seq_len,block_size,penalty,
           geometry.blocks,geometry.length,geometry.columns,geometry.cells,geometry.shared_bytes);
}
'''


def expected(n, block, penalty):
    if min(n, block, penalty) <= 0:
        return None
    blocks = (n + block - 1) // block
    length = blocks * block
    cols = length + 1
    if cols*cols > INT_MAX or cols*max(penalty, 11) > INT_MAX:
        return None
    return ','.join(map(str, (n, block, penalty, blocks, length, cols, cols*cols,
                              4*((block+1)**2+block**2))))


def main():
    if not __debug__:
        raise RuntimeError("Run without Python optimization")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--compiler', default='c++')
    ap.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1]/'tier2/rodinia_nw/rodinia_nw.hip')
    args = ap.parse_args()
    source = args.source.read_text()
    manifest = json.loads((args.source.parent/'params.json').read_text())
    triples = list(itertools.product(manifest['scaling_param']['values'], [16], [1,2,5,10]))
    triples += [(n,b,p) for n in [-2147483648,-1,0,1,15,16,17,31,32,33,46336,46339,46340,INT_MAX]
                for b in [0,1,3,16,32,INT_MAX] for p in [0,1,INT_MAX]]
    for n in [1,16,33,2048,24576,46336]:
        boundary=INT_MAX//(((n+15)//16)*16+1)
        triples += [(n,16,p) for p in [boundary,boundary+1]]
    cases=[([],dict(sequence_length=str(n),block_size=str(b),penalty=str(p)),expected(n,b,p))
           for n,b,p in dict.fromkeys(triples)]
    cases += [([],{},expected(2048,16,2)),
              (['--sequence_length','17','--block_size','32','--penalty','5'],{},expected(17,32,5)),
              (['--sequence_length','0','--block_size','-1','--penalty','0'],{},expected(2048,16,2)),
              (['--sequence_length','17'],dict(sequence_length='33'),expected(33,16,2))]
    clean={k:v for k,v in os.environ.items() if not k.startswith('BENCH_PARAM_')}
    with tempfile.TemporaryDirectory(prefix='nw-reference-') as tmp:
        cpp=Path(tmp)/'test.cpp';binary=Path(tmp)/'test'
        for label,text in [('reference',HEADERS+host_functions(source)+TEST_MAIN),('geometry',geometry_source(source))]:
            cpp.write_text(text)
            subprocess.run([args.compiler,'-std=c++14','-O2','-fsanitize=address,undefined',
                            '-fno-sanitize-recover=all',str(cpp),'-o',str(binary)],check=True,timeout=60)
            if label=='reference':
                p=subprocess.run([str(binary)],text=True,capture_output=True,timeout=60)
                if p.returncode:raise AssertionError(p.stdout+p.stderr)
                print(p.stdout.strip())
                continue
            for argv,values,want in cases:
                p=subprocess.run([str(binary),*argv],env=dict(clean,**{'BENCH_PARAM_'+k:v for k,v in values.items()}),
                                 text=True,capture_output=True,timeout=30)
                if want is None:
                    assert p.returncode==1 and not p.stdout and p.stderr=='Invalid NW dimensions/penalty or signed index/score range\n',(argv,values,p.stdout,p.stderr)
                else:
                    assert p.returncode==0 and not p.stderr and p.stdout.strip()==want,(argv,values,want,p.stdout,p.stderr)
    print(f'{len(cases)} actual parameter/index/score cases pass under fatal sanitizers')


if __name__=='__main__':
    main()
