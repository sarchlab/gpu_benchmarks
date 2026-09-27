#!/usr/bin/env python3
"""Exercise actual atomic parameter admission and arithmetic without a GPU."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

def main():
    if not __debug__: raise RuntimeError('Run without Python optimization')
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--compiler',default='c++');args=ap.parse_args()
    repo=Path(__file__).resolve().parents[1]
    flags=['-std=c++14','-O2','-fsanitize=address,undefined','-fno-sanitize-recover=all']
    with tempfile.TemporaryDirectory(prefix='atomic-host-') as d:
        work=Path(d);binary=work/'test'
        subprocess.run([args.compiler,*flags,str(repo/'tests/atomic_reference.cpp'),'-o',str(binary)],check=True)
        subprocess.run([str(binary)],check=True,timeout=30)
        src=work/'parameters.cpp';src.write_text('#include "'+str(repo/'tier1/atomic_operations/atomic_reference.h')+'"\n'+r"""
int main(int argc,char**argv)try{auto p=atomic_bench::parameters(argc,argv);printf("%d,%d,%d,%s\n",p.ops,p.blocks,p.block,p.operation);}
catch(const std::exception& e){fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
""")
        subprocess.run([args.compiler,*flags,str(src),'-o',str(binary)],check=True)
        clean={k:v for k,v in os.environ.items() if not k.startswith('BENCH_PARAM_')}
        cases=[({},[],'1024,2048,256,atomicAdd'),({},['7'],'7,2048,256,atomicAdd'),
               ({'ops_per_thread':'9'},['invalid'],'9,2048,256,atomicAdd'),
               ({'num_blocks':'1','threads_per_block':'1','atomic_op':'atomicCAS'},['3'],'3,1,1,atomicCAS'),
               ({'num_blocks':'4194303','threads_per_block':'1024','ops_per_thread':'2147483647'},[],'2147483647,4194303,1024,atomicAdd')]
        for name in ['ops_per_thread','num_blocks','threads_per_block']:
            cases += [({name:v},[],None) for v in ['', '0','-1','2147483648','999999999999999999999','8x']]
        cases += [({'atomic_op':v},[],None) for v in ['', 'atomicSub','atomicCASx']]
        cases += [({'threads_per_block':'1025'},[],None),({'num_blocks':'4194304','threads_per_block':'1024'},[],None),({},['2','3'],None),({},['bad'],None)]
        for env,cli,want in cases:
            run=subprocess.run([str(binary),*cli],env=dict(clean,**{'BENCH_PARAM_'+k:v for k,v in env.items()}),text=True,capture_output=True,timeout=10)
            assert (run.returncode==1 and not run.stdout and run.stderr.startswith('FAIL:')) if want is None else (run.returncode==0 and not run.stderr and run.stdout.strip()==want),(env,cli,run)
        print(f'PASS: {len(cases)} actual parameter/range/precedence cases')
if __name__=='__main__':main()
