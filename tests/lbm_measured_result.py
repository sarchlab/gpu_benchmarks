#!/usr/bin/env python3
"""Build/run complete LBM measured-output and pinned-GPU regressions."""
import argparse,json,os,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--platform',choices=['cuda','rocm'],default='cuda')
    ap.add_argument('--compile-only',action='store_true');ap.add_argument('--work-dir',type=Path,required=True);a=ap.parse_args()
    work=a.work_dir.resolve();work.mkdir(parents=True,exist_ok=False)
    for path in ('common','tier2/parboil_lbm','tests/upstream/lbm'):
        shutil.copytree(ROOT/path,work/path,ignore=shutil.ignore_patterns('parboil_lbm','original_gpu_*'))
    shutil.copyfile(ROOT/'tests/lbm_original_gpu.hip',work/'tests/lbm_original_gpu.hip')
    d=work/'tier2/parboil_lbm';source=d/'parboil_lbm.hip';text=source.read_text()
    text=text.replace('const auto expected=lbm::reference','auto expected=lbm::reference')
    anchor='    const auto result=lbm::compare(l,actual,expected);';assert text.count(anchor)==1
    hook=r'''    if(const char* bad=std::getenv("LBM_TEST_BAD")) {
        const int q=std::atoi(std::getenv("LBM_TEST_Q"));
        const float poison=std::strcmp(bad,"nan")==0?NAN:std::strcmp(bad,"inf")==0?INFINITY:std::strcmp(bad,"neg_inf")==0?-INFINITY:1.0f;
        if(std::getenv("LBM_TEST_REF"))expected.at(p.n-1,p.n-1,p.n-1,q)=poison;
        else if(q==lbm::FLAGS)actual[l.index(p.n-1,p.n-1,p.n-1,q)]=0;
        else actual[l.index(p.n-1,p.n-1,p.n-1,q)]=poison;
    }
'''
    text=text.replace(anchor,hook+anchor)
    anchor='    for(int t=0;t<p.steps;++t) {';assert text.count(anchor)==1
    text=text.replace(anchor,anchor+'\n        if(std::getenv("LBM_TEST_SKIP_LAST") && t==p.steps-1)continue;')
    source.write_text(text)
    subprocess.run(['make','-C',str(d),'PLATFORM='+a.platform],check=True)
    compiler=['nvcc','-O3','-std=c++14','-x','cu','-Xcompiler','-pthread'] if a.platform=='cuda' else ['hipcc','-O3','-std=c++14','-pthread']
    for n in (16,17,64):
        subprocess.run(compiler+['-DLBM_TEST_N='+str(n),str(work/'tests/lbm_original_gpu.hip'),'-o',str(work/f'original-{n}')],check=True)
    if a.compile_only:print('PASS: compiled measured faults and pinned GPU comparisons');return
    clean={k:v for k,v in os.environ.items() if not(k.startswith('BENCH_PARAM_') or k.startswith('LBM_TEST_'))};results=[]
    def run(binary,argv,env,want,label):
        p=subprocess.run([str(binary),*argv],env=dict(clean,**env),text=True,capture_output=True,timeout=60)
        (work/(label+'.stdout')).write_text(p.stdout);(work/(label+'.stderr')).write_text(p.stderr)
        assert p.returncode==(0 if want else 1),(label,p.returncode,p.stdout,p.stderr)
        if not want:assert 'FAIL: cells=4096 populations=77824 errors=' in p.stderr,(label,p.stderr)
        results.append(dict(label=label,exit=p.returncode));(work/'report.json').write_text(json.dumps(results,indent=2)+'\n')
        return p
    for warm in ('false','true'):
        base=dict(BENCH_PARAM_grid_dim='16',BENCH_PARAM_num_timesteps='10',BENCH_PARAM_vector_add_warmup=warm)
        p=run(d/'parboil_lbm',[],base,True,'healthy-'+warm)
        records=[json.loads(s) for s in p.stdout.splitlines()];assert len(records)==2 and records[0]['type']=='kernel' and records[1]['type']=='summary'
        assert records[0]['params']['verification']=='full_fp32_upstream_v1' and 'cells=4096 populations=77824 errors=0' in p.stderr
        for q in range(19):
            run(d/'parboil_lbm',[],dict(base,LBM_TEST_BAD='bad',LBM_TEST_Q=str(q)),False,f'tail-{q}-{warm}')
        for side in ('actual','reference'):
            for bad in ('nan','inf','neg_inf'):
                extra=dict(LBM_TEST_BAD=bad,LBM_TEST_Q='18')
                if side=='reference':extra['LBM_TEST_REF']='1'
                run(d/'parboil_lbm',[],dict(base,**extra),False,f'{side}-{bad}-{warm}')
        run(d/'parboil_lbm',[],dict(base,LBM_TEST_BAD='bad',LBM_TEST_Q='19'),False,'flags-'+warm)
        run(d/'parboil_lbm',[],dict(base,LBM_TEST_SKIP_LAST='1'),False,'missing-step-'+warm)
    for n in (16,17,64):
        for t in (1,2,10,100):run(work/f'original-{n}',[str(t)],{},True,f'upstream-{n}-{t}')
    print(f'PASS: {len(results)} actual GPU checks; every measured tail population, nonfinite references, flags, missing work and frozen upstream comparisons')
if __name__=='__main__':main()
