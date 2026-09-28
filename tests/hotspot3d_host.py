#!/usr/bin/env python3
"""Check pinned HotSpot3D physics, full references and actual parameter admission."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

def main():
    if not __debug__: raise RuntimeError('Run without Python optimization')
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--compiler',default='c++');ap.add_argument('--all-inputs',action='store_true');args=ap.parse_args()
    repo=Path(__file__).resolve().parents[1];flags=['-O2','-std=c++14','-fsanitize=address,undefined','-fno-sanitize-recover=all']
    with tempfile.TemporaryDirectory(prefix='hotspot3d-host-') as d:
        work=Path(d);binary=work/'test'
        subprocess.run([args.compiler,*flags,str(repo/'tests/hotspot3d_reference.cpp'),'-o',str(binary)],check=True)
        subprocess.run([str(binary),*(['--all-inputs'] if args.all_inputs else [])],check=True,timeout=300)
        source=work/'params.cpp';source.write_text('#include "'+str(repo/'tier2/rodinia_hotspot3d/hotspot3d_reference.h')+'"\n'+r"""
int main(int argc,char**argv)try{auto p=hotspot3d::parameters(argc,argv);hotspot3d::admissible(hotspot3d::coefficients(p.n));printf("%d,%d,%d,%.17g,%.9g\n",p.n,p.block,p.steps,p.requested_ambient,p.ambient);}
catch(const std::exception& e){fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
""")
        subprocess.run([args.compiler,*flags,str(source),'-o',str(binary)],check=True)
        clean={k:v for k,v in os.environ.items() if not k.startswith('BENCH_PARAM_')}
        cases=[({},[],'64,8,10,80,80'),({},['--grid_size','17','--num_iterations','3','--amb_temp','80.125'],'17,8,3,80.125,80.125'),({'grid_size':'3'},['--grid_size','17'],'3,8,10,80,80'),({},['--grid_size','3','--grid_size','17'],'3,8,10,80,80'),({'amb_temp':'-20.5'},[],'64,8,10,-20.5,-20.5')]
        cases += [({'grid_size':str(n),'block_size':str(b),'num_iterations':str(t)},[],f'{n},{b},{t},80,80') for n in [1,16,17,512,1024,1290] for b in [1,8,10] for t in [1,50,2147483647]]
        for name in ['grid_size','block_size','num_iterations']:
            cases += [({name:x},[],None) for x in ['', '0','-1','2147483648','99999999999999999999','8x']]
        cases += [({'grid_size':'1291'},[],None),({'block_size':'11'},[],None)]
        cases += [({'amb_temp':x},[],None) for x in ['', 'nan','inf','-inf','80x','1e99','1e-99']]
        cases += [({},x,None) for x in [['--grid_size'],['--amb_temp'],['--bad','3'],['3'],['--grid_size','bad']]]
        for values,cli,want in cases:
            run=subprocess.run([str(binary),*cli],env=dict(clean,**{'BENCH_PARAM_'+k:v for k,v in values.items()}),text=True,capture_output=True,timeout=10)
            assert (run.returncode==1 and not run.stdout and run.stderr.startswith('FAIL:')) if want is None else (run.returncode==0 and not run.stderr and run.stdout.strip()==want),(values,cli,run)
        print(f'PASS: {len(cases)} actual input/geometry/precedence cases')
if __name__=='__main__':main()
