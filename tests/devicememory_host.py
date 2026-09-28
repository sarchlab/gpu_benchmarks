#!/usr/bin/env python3
"""Independent complete references, strict controls and signed-loop admission."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

def main():
    if not __debug__:raise RuntimeError('Run without Python optimization')
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--compiler',default='c++');a=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='devicememory-host-') as folder:
        exe=Path(folder)/'test';subprocess.run([a.compiler,'-O2','-std=c++14','-fsanitize=address,undefined','-fno-sanitize-recover=all',str(root/'tests/devicememory_reference.cpp'),'-o',str(exe)],check=True)
        subprocess.run([str(exe)],check=True,timeout=120)
        clean={k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','DM_TEST_'))}
        cases=[]
        for mib in [1,64,96,4096]:
            for block in [1,32,64,255,256,1024]:
                for mode in ['read','write','readwrite']:
                    n=mib*262144;blocks=min((n+block-1)//block,65535);stride=blocks*block
                    cases.append(({},['--array_size_mb',str(mib),'--block_size',str(block),'--transfer_mode',mode],f'{mib},{block},{mode},{n},{blocks},{stride},{min(n,stride)}'))
        cases += [({'array_size_mb':'2','block_size':'32','transfer_mode':'write'},['--array_size_mb','1'], '2,32,write,524288,16384,524288,524288'),({},['--array_size_mb','1','--array_size_mb','2'],'1,256,readwrite,262144,1024,262144,262144')]
        for key in ['array_size_mb','block_size']:
            for value in ['', '0','-1','2147483648','9999999999999999999','1x']:
                cases.append(({key:value},['--array_size_mb','1'],None))
        cases += [({'transfer_mode':value},['--array_size_mb','1'],None) for value in ['', 'other','READ','read"']]
        cases += [({},cli,None) for cli in [['--array_size_mb'],['--unknown','1'],['1'],['--array_size_mb','8192'],['--array_size_mb','8191'],['--block_size','2147483647'],['--array_size_mb','bad'],['--transfer_mode','other']]]
        cases.append(({'array_size_mb':'1'},['--array_size_mb','bad'],None))
        for values,cli,want in cases:
            run=subprocess.run([str(exe),*cli],env=dict(clean,**{'BENCH_PARAM_'+k:v for k,v in values.items()}),text=True,capture_output=True,timeout=10)
            assert (run.returncode==1 and not run.stdout and run.stderr.startswith('FAIL:')) if want is None else (run.returncode==0 and not run.stderr and run.stdout.strip()==want),(values,cli,run)
        print(f'PASS: {len(cases)} actual parser/precedence/signed-loop cases')
if __name__=='__main__':main()
