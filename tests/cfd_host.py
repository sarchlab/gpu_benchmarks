#!/usr/bin/env python3
"""Independent mesh geometry, complete references, and actual CFD admission controls."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

def main():
    if not __debug__:raise RuntimeError('Run without Python optimization')
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--compiler',default='c++');args=ap.parse_args()
    repo=Path(__file__).resolve().parents[1];flags=['-O2','-std=c++14','-pthread','-fsanitize=address,undefined','-fno-sanitize-recover=all']
    with tempfile.TemporaryDirectory(prefix='cfd-host-') as d:
        work=Path(d);binary=work/'reference'
        subprocess.run([args.compiler,*flags,str(repo/'tests/cfd_reference.cpp'),'-o',str(binary)],check=True)
        subprocess.run([str(binary)],check=True,timeout=120)
        # Exercise both zero-worker and partial-worker creation failure using
        # isolated header copies; production has no failure-injection controls.
        header=(repo/'tier2/altis_cfd/cfd_reference.h').read_text()
        loop='for(unsigned i=1;i<count;++i)workers.emplace_back(work);'
        assert header.count(loop)==1
        for failure in ['std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again))','std::bad_alloc()']:
            for started in [0,2]:
                altered=header.replace(loop,'for(unsigned i=1;i<count;++i) { if(workers.size()=='+str(started)+')throw '+failure+'; workers.emplace_back(work); }')
                altered=altered.replace('std::min(16u,available)','4u')  # force partial-team coverage on two-core CI
                (work/'cfd_reference.h').write_text(altered)
                fallback=work/'fallback.cpp';fallback.write_text((repo/'tests/cfd_reference.cpp').read_text().replace('../tier2/altis_cfd/cfd_reference.h','cfd_reference.h'))
                subprocess.run([args.compiler,*flags,'-I'+str(repo/'tests'),str(fallback),'-o',str(binary)],check=True)
                subprocess.run([str(binary)],check=True,timeout=120)
        print('PASS: complete references survive system_error/bad_alloc at zero-worker and partial-worker creation')
        # Reject the former double-literal rounding independently in the boundary
        # constants and the approved perturbed initial-state energy expression.
        for line in ['u[4]=u[0]*', 'in.initial[i+4*n]=']:
            original=next(x for x in header.splitlines() if line in x)
            assert 'float(1.4f-1.0f)' in original
            altered=header.replace(original,original.replace('float(1.4f-1.0f)','float(1.4-1.0f)'))
            (work/'cfd_reference.h').write_text(altered)
            subprocess.run([args.compiler,*flags,'-I'+str(repo/'tests'),str(fallback),'-o',str(binary)],check=True)
            rejected=subprocess.run([str(binary)],capture_output=True,text=True,timeout=120)
            expected_function='check_farfield' if line.startswith('u[4]') else 'check_mesh'
            assert rejected.returncode!=0 and 'assert' in rejected.stderr.lower() and expected_function in rejected.stderr,rejected
        print('PASS: independent upstream setup rejects both former gamma-rounding mutations')
        source=work/'parameters.cpp';source.write_text('#include "'+str(repo/'tier2/altis_cfd/cfd_reference.h')+'"\n'+r'''
int main(int argc,char**argv)try{auto p=cfd::parameters(argc,argv);printf("%d,%d,%d\n",p.cells,p.block,p.padded);}
catch(const std::exception&e){fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
''')
        subprocess.run([args.compiler,*flags,str(source),'-o',str(binary)],check=True)
        clean={k:v for k,v in os.environ.items() if not k.startswith('BENCH_PARAM_')}
        cases=[({},[],'97000,256,97024'),({},['--size','7','--block_size','3'],'7,3,9'),({'size':'503'},['--size','31'],'503,256,512'),({},['--size','1','--size','7'],'7,256,256')]
        cases += [({'size':str(n),'block_size':str(b)},[],f'{n},{b},{(n+b-1)//b*b}') for n in [1,6,7,500,503,97000,10000000] for b in [1,7,32,256,1024]]
        for key in ['size','block_size']:
            cases += [({key:v},[],None) for v in ['', '0','-1','+1',' 1','1 ','1e3','3x','2147483648','99999999999999999999']]
        cases += [({'size':'178956970','block_size':'1'},[],'178956970,1,178956970'),({'size':'178956971','block_size':'1'},[],None),({'size':'178956970','block_size':'1024'},[],None),({'block_size':'1025'},[],None)]
        cases += [({},x,None) for x in [['--size'],['--block_size'],['--unknown','1'],['10'],['--size','bad']]]
        for env,cli,want in cases:
            r=subprocess.run([str(binary),*cli],env=dict(clean,**{'BENCH_PARAM_'+k:v for k,v in env.items()}),text=True,capture_output=True,timeout=10)
            assert (r.returncode==1 and not r.stdout and r.stderr.startswith('FAIL:')) if want is None else (r.returncode==0 and r.stdout.strip()==want),(env,cli,r)
        print(f'PASS: {len(cases)} actual control, padding/index range and precedence cases')
if __name__=='__main__':main()
