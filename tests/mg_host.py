#!/usr/bin/env python3
"""Pinned MG host oracle, actual parameter admission and worker cleanup."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    if not __debug__:
        raise RuntimeError('Run without Python optimization')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--compiler', default='c++')
    ap.add_argument('--all-inputs', action='store_true')
    ap.add_argument('--thread-sanitizer', action='store_true')
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    source = (repo/'tier2/npb_mg/hip/npb_mg.hip').read_text()
    header = (repo/'tier2/npb_mg/hip/nas_mg.h').read_text()
    flags = ['-std=c++14','-O2','-pthread','-fno-sanitize-recover=all',
             '-fsanitize=thread' if args.thread_sanitizer else '-fsanitize=address,undefined']
    clean = {k:v for k,v in os.environ.items() if not k.startswith('BENCH_PARAM_')}
    with tempfile.TemporaryDirectory(prefix='mg-host-') as folder:
        work = Path(folder);binary = work/'test'
        subprocess.run([args.compiler,*flags,str(repo/'tests/mg_reference.cpp'),'-o',str(binary)],check=True)
        subprocess.run([str(binary),*(['--all-inputs'] if args.all_inputs else [])],check=True,timeout=600 if args.all_inputs else 120)
        anchor = '                threads.emplace_back([this,id]() {'
        assert header.count(anchor) == 1
        (work/'nas_mg.h').write_text(header.replace(anchor,
            '                if(static_cast<int>(id)-1==mg_fail_after)throw std::runtime_error("injected worker creation failure");\n'+anchor))
        (work/'mg_reference.h').write_bytes((repo/'tier2/npb_mg/hip/mg_reference.h').read_bytes())
        cpp = work/'pool.cpp';cpp.write_text(r'''
static int mg_fail_after=-1;
#include "mg_reference.h"
int main() {
    const auto c=*nas_mg::problem('S');const nas_mg::Layout l(c);
    const auto baseline=nas_mg::reference(c,l,1);
    for(int round=0;round<4;++round) {
        for(int count:{0,1,6}) {
            mg_fail_after=count;bool threw=false;
            try {nas_mg::reference(c,l,8);}catch(const std::runtime_error&){threw=true;}
            if(!threw)return 1;
        }
        mg_fail_after=-1;
        if(nas_mg::compare(nas_mg::reference(c,l,8),baseline))return 1;
    }
    puts("PASS: 12 worker creation failures cleaned up and four complete references after cleanup");
}
''')
        subprocess.run([args.compiler,*flags,str(cpp),'-o',str(binary)],check=True)
        subprocess.run([str(binary)],check=True,timeout=120)
        if args.thread_sanitizer:
            return
        start='static int integer(';end='int main('
        assert source.count(start)==source.count(end)==1
        cpp.write_text('#include <cerrno>\n#include <climits>\n#include <cstring>\n#include <cstdlib>\n'
            '#include "'+str(repo/'tier2/npb_mg/hip/nas_mg.h')+'"\n'+start+source.split(start,1)[1].split(end,1)[0]+r'''
int main(int argc,char**argv) try {
    int b;const auto c=*parameters(argc,argv,&b);
    printf("%c,%d,%d,%d\n",c.name,c.n,c.iterations,b);
} catch(const std::exception& error){fprintf(stderr,"FAIL: %s\n",error.what());return 1;}
''')
        subprocess.run([args.compiler,*flags,str(cpp),'-o',str(binary)],check=True)
        cases=[]
        for name,n,it in [('S',32,4),('W',128,4),('A',256,4),('B',256,20),('C',512,20)]:
            for block in [1,8,32,256,1024]:cases.append((dict(npb_class=name,block_size=str(block)),[],f'{name},{n},{it},{block}'))
        cases += [({},[],'A,256,4,256'),(dict(npb_class='B'),['--class','S'],'S,32,4,256')]
        cases += [(dict(npb_class=x),[],None) for x in ['', 's','AA','D','32',' A','A ']]
        cases += [(dict(block_size=x),[],None) for x in ['', '0','-1','3','1025','1x','2147483648','-2147483649','9999999999999999999999']]
        cases += [({k:x},[],None) for k in ['grid_size','num_vcycles','num_smooth_steps'] for x in ['', '4']]
        cases += [({},x,None) for x in [['--size','256'],['--class'],['--class','X'],['S'],['--other','S']]]
        for params,cli,want in cases:
            run=subprocess.run([str(binary),*cli],env=dict(clean,**{'BENCH_PARAM_'+k:v for k,v in params.items()}),text=True,capture_output=True,timeout=10)
            assert (run.returncode==1 and not run.stdout and run.stderr.startswith('FAIL:')) if want is None else (run.returncode==0 and not run.stderr and run.stdout.strip()==want),(params,cli,run)
        print(f'PASS: {len(cases)} actual class/launch/obsolete-control admission checks')


if __name__=='__main__':
    main()
