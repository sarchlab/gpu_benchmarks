#!/usr/bin/env python3
"""Check pinned HotSpot physics, full references, geometry and worker cleanup."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

def main():
    if not __debug__:raise RuntimeError('Run without Python optimization')
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--compiler',default='c++');ap.add_argument('--all-inputs',action='store_true');ap.add_argument('--thread-sanitizer',action='store_true');args=ap.parse_args()
    repo=Path(__file__).resolve().parents[1];flags=['-O2','-std=c++14','-pthread','-fsanitize=thread' if args.thread_sanitizer else '-fsanitize=address,undefined','-fno-sanitize-recover=all']
    with tempfile.TemporaryDirectory(prefix='hotspot-host-') as name:
        work=Path(name);exe=work/'test'
        subprocess.run([args.compiler,*flags,str(repo/'tests/hotspot_reference.cpp'),'-o',str(exe)],check=True)
        subprocess.run([str(exe),*(['--all-inputs'] if args.all_inputs else [])],check=True,timeout=300)
        source=work/'params.cpp';source.write_text('#include "'+str(repo/'tier2/rodinia_hotspot/hip/hotspot_reference.h')+'"\n'+r"""
int main(int argc,char**argv)try{auto p=hotspot::parameters(argc,argv);auto g=hotspot::geometry(p);printf("%d,%d,%d,%d,%llu\n",p.n,p.steps,p.pyramid,g.blocks,(unsigned long long)g.launches);}
catch(const std::exception& e){fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
""")
        subprocess.run([args.compiler,*flags,str(source),'-o',str(exe)],check=True)
        clean={k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','HS2D_TEST_'))}
        cases=[({},[],'1024,360,2,86,180'),({},['--grid_size','17','--total_iterations','3','--pyramid_height','7'],'17,3,7,9,1'),({'grid_size':'3'},['--grid_size','17'],'3,360,2,1,180'),({},['--grid_size','3','--grid_size','17'],'3,360,2,1,180')]
        for n in [1,17,8192]:
            for p in range(1,8):
                for t in [1,3,360]:cases.append(({'grid_size':str(n),'pyramid_height':str(p),'total_iterations':str(t)},[],f'{n},{t},{p},{(n+15-2*p)//(16-2*p)},{(t+p-1)//p}'))
        cases += [({'grid_size':'46340','pyramid_height':'1','total_iterations':'2147483647'},[],'46340,2147483647,1,3310,2147483647')]
        for k in ['grid_size','pyramid_height','total_iterations']:
            for value in ['', '0','-1','2147483648','99999999999999999999','4x']:cases.append(({k:value},[],None))
        cases += [({'grid_size':'46341'},[],None),({'pyramid_height':'8'},[],None),({'grid_size':'20000','pyramid_height':'7','total_iterations':'1'},[],None),({'grid_size':'46340','pyramid_height':'2','total_iterations':'2'},[],None),({'grid_size':'32000','pyramid_height':'7','total_iterations':'2147483647'},[],None)]
        cases += [({},x,None) for x in [['--grid_size'],['--bad','1'],['3'],['--grid_size','bad']]]
        cases += [({'grid_size':'17'},['--grid_size','bad'],None)]
        for values,cli,want in cases:
            run=subprocess.run([str(exe),*cli],env=dict(clean,**{'BENCH_PARAM_'+k:v for k,v in values.items()}),text=True,capture_output=True,timeout=10)
            assert (run.returncode==1 and not run.stdout and run.stderr.startswith('FAIL:')) if want is None else (run.returncode==0 and not run.stderr and run.stdout.strip()==want),(values,cli,run)
        print(f'PASS: {len(cases)} actual parameter/precedence/overflow cases')
        # Inject partial thread-creation failures only into an isolated header.
        header=(repo/'tier2/rodinia_hotspot/hip/hotspot_reference.h').read_text();anchor='                threads.emplace_back([this,id]() {';assert header.count(anchor)==1
        header=header.replace(anchor,'                if(const char* fail=std::getenv("HS2D_TEST_WORKER_FAIL_AFTER"))if(id-1==static_cast<unsigned>(std::strtoul(fail,nullptr,10)))throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));\n'+anchor)
        (work/'hotspot_reference.h').write_text(header);source.write_text(r"""#include "hotspot_reference.h"
int main()try{hotspot::RowPool pool(8);return 2;}catch(const std::system_error&){return 0;}
""")
        subprocess.run([args.compiler,*flags,str(source),'-o',str(exe)],check=True)
        for count in ['0','3','6']:subprocess.run([str(exe)],env=dict(clean,HS2D_TEST_WORKER_FAIL_AFTER=count),check=True,timeout=10)
        print('PASS: 3 partial-worker creation failures stop and join cleanly')
if __name__=='__main__':main()
