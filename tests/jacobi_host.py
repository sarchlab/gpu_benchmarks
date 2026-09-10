#!/usr/bin/env python3
"""Check pinned Jacobi input/recurrence and actual parameter admission."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    if not __debug__:
        raise RuntimeError('Run without Python optimization')
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--compiler',default='c++')
    ap.add_argument('--all-inputs',action='store_true')
    args=ap.parse_args();repo=Path(__file__).resolve().parents[1]
    flags=['-O2','-std=c++14','-fsanitize=address,undefined','-fno-sanitize-recover=all']
    with tempfile.TemporaryDirectory(prefix='jacobi-host-') as directory:
        work=Path(directory);binary=work/'test'
        subprocess.run([args.compiler,*flags,str(repo/'tests/jacobi_reference.cpp'),'-o',str(binary)],check=True)
        subprocess.run([str(binary),*(['--all-inputs'] if args.all_inputs else [])],check=True,timeout=300)
        cpp=work/'parameters.cpp';cpp.write_text('#include "'+str(repo/'tier2/polybench_jacobi2d/jacobi_reference.h')+'"\n'+r'''
int main(int argc,char**argv)try{auto p=jacobi::parameters(argc,argv);printf("%d,%d\n",p.n,p.steps);}
catch(const std::exception& e){fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
''')
        subprocess.run([args.compiler,*flags,str(cpp),'-o',str(binary)],check=True)
        clean={k:v for k,v in os.environ.items() if not k.startswith('BENCH_PARAM_')}
        cases=[({},[],'1024,50'),({},['--n','65','--tsteps','3'],'65,3'),
               ({'n':'17','tsteps':'2'},['--n','65','--tsteps','3'],'17,2'),
               ({},['--n','17','--n','65','--tsteps','2','--tsteps','3'],'17,2')]
        cases += [({'n':str(n),'tsteps':str(t)},[],f'{n},{t}') for n in [3,64,65,8192,46340] for t in [1,3,200,2147483647]]
        cases += [({'n':x},[],None) for x in ['', '0','1','2','-1','46341','2147483647','2147483648','999999999999999999999','64x']]
        cases += [({'tsteps':x},[],None) for x in ['', '0','-1','2147483648','-2147483649','200x']]
        cases += [({},x,None) for x in [['--n'],['--tsteps'],['--other','3'],['--n','garbage'],['64'],['--tsteps','0']]]
        for values,cli,want in cases:
            p=subprocess.run([str(binary),*cli],env=dict(clean,**{'BENCH_PARAM_'+k:v for k,v in values.items()}),text=True,capture_output=True,timeout=10)
            assert (p.returncode==1 and not p.stdout and p.stderr.startswith('FAIL:')) if want is None else (p.returncode==0 and not p.stderr and p.stdout.strip()==want),(values,cli,p)
        print(f'PASS: {len(cases)} actual parameter/index/CLI precedence cases')


if __name__=='__main__':main()
