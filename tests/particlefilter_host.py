#!/usr/bin/env python3
"""Input fidelity, RNG endpoint, worker fallback, and strict size admission."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

def main():
    if not __debug__:raise RuntimeError('Run without Python optimization')
    ap=argparse.ArgumentParser();ap.add_argument('--compiler',default='c++');args=ap.parse_args();repo=Path(__file__).resolve().parents[1]
    flags=['-O2','-std=c++14','-pthread','-fwrapv','-fsanitize=address,undefined,float-cast-overflow','-fno-sanitize-recover=all']
    with tempfile.TemporaryDirectory(prefix='particlefilter-host-') as d:
        work=Path(d);binary=work/'test'
        subprocess.run([args.compiler,*flags,str(repo/'tests/particlefilter_reference.cpp'),'-o',str(binary)],check=True);subprocess.run([str(binary)],check=True,timeout=120)
        folder=work/'tier2/altis_particlefilter/hip';folder.mkdir(parents=True);tests=work/'tests';tests.mkdir();shutil.copytree(repo/'tests/fixtures',tests/'fixtures')
        shutil.copy2(repo/'tests/particlefilter_reference.cpp',tests);shutil.copy2(repo/'tier2/altis_particlefilter/hip/particlefilter_input.h',folder)
        ref=(repo/'tier2/altis_particlefilter/hip/particlefilter_reference.h').read_text();anchor='for(unsigned i=1;i<workers;++i)threads.emplace_back(work);';assert ref.count(anchor)==1
        for started in [0,2]:
            for failure in ['std::bad_alloc()','std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again))']:
                modified=ref.replace('std::min(16u,std::max(1u,std::thread::hardware_concurrency()))','4u').replace(anchor,'for(unsigned i=1;i<workers;++i) {if(threads.size()=='+str(started)+')throw '+failure+';threads.emplace_back(work);}')
                (folder/'particlefilter_reference.h').write_text('#include <system_error>\n'+modified)
                subprocess.run([args.compiler,*flags,str(tests/'particlefilter_reference.cpp'),'-o',str(binary)],check=True);subprocess.run([str(binary)],check=True,timeout=120)
        source=work/'params.cpp';source.write_text('#include "'+str(folder/'particlefilter_input.h')+'"\n#include <cstdio>\nint main(int argc,char**argv)try{printf("%d\\n",pf::parameters(argc,argv));}catch(const std::exception&e){fprintf(stderr,"FAIL: %s\\n",e.what());return 1;}\n')
        subprocess.run([args.compiler,*flags,str(source),'-o',str(binary)],check=True)
        clean={k:v for k,v in os.environ.items() if not k.startswith('BENCH_PARAM_')};cases=[({},[],100000),({},['--size','1'],1),({'size':'513'},['--size','2'],513),({'size':str(2147483647//69)},[],2147483647//69)]
        cases += [({'size':s},[],None) for s in ['', '0','-1','+1',' 1','1 ','1e3','10x','9999999999999999999999999',str(2147483647//69+1)]]
        cases += [({},cli,None) for cli in [['--size'],['--seed','1'],['--frames','2'],['--size','bad'],['100']]]
        for env,cli,want in cases:
            run=subprocess.run([str(binary),*cli],env=dict(clean,**{'BENCH_PARAM_'+k:v for k,v in env.items()}),capture_output=True,text=True,timeout=10)
            assert (run.returncode==1 and not run.stdout and run.stderr.startswith('FAIL:')) if want is None else (run.returncode==0 and run.stdout.strip()==str(want)),(env,cli,run)
        print(f'PASS: complete references survive four worker failures; {len(cases)} admission and precedence cases')
if __name__=='__main__':main()
