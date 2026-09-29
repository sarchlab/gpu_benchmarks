#!/usr/bin/env python3
"""Exercise the production LBM admission and padded layouts without a GPU."""
import argparse,itertools,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--compiler',default='c++');ap.add_argument('--binary',type=Path)
    ap.add_argument('--work-dir',type=Path,required=True);a=ap.parse_args()
    a.work_dir.mkdir(parents=True,exist_ok=False)
    source=a.work_dir/'ranges.cpp';binary=a.work_dir/'ranges'
    source.write_text(r'''#include "tier2/parboil_lbm/lbm_common.h"
int main()try{const auto p=lbm::parameters();const lbm::Layout l(p.n);
printf("%d %d %d %zu\n",p.n,p.steps,l.entries,l.bytes());return 0;
}catch(const std::exception& e){fprintf(stderr,"FAIL: %s\n",e.what());return 1;}''')
    subprocess.run([a.compiler,'-std=c++14','-O1','-g','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all','-I',str(ROOT),str(source),'-o',str(binary)],check=True)
    manifest=json.loads((ROOT/'tier2/parboil_lbm/params.json').read_text())
    ns=manifest['scaling_param']['values'];ts=next(p['values'] for p in manifest['non_scaling_params'] if p['name']=='num_timesteps')
    cases=[dict(grid_dim=str(n),num_timesteps=str(t)) for n,t in itertools.product(ns,ts)]
    cases += [dict(grid_dim=str(n),num_timesteps='1') for n in (-2147483648,-1,0,1,2,3,17,460,470,471,475,480,483,484,50000,2147483647)]
    cases += [dict(num_timesteps=str(t)) for t in (-2147483648,-1,0,1,2147483647)]
    cases += [{},dict(grid_dim=''),dict(num_timesteps='bad'),dict(tau='.7'),dict(block_size='128'),dict(grid_dim=' 17suffix',num_timesteps=' 2suffix')]
    clean={k:v for k,v in os.environ.items() if not k.startswith('BENCH_PARAM_')};results=[]
    for overrides in cases:
        def atoi(v):
            import re
            m=re.match(r'\s*([+-]?\d+)',v);return int(m[1]) if m else 0
        n=atoi(overrides.get('grid_dim','64'));t=atoi(overrides.get('num_timesteps','100'))
        entries=20*(n+8)*n*(n+8)
        ok=3<=n<=483 and t>0 and entries<=2147483647 and (n,t)!=(256,500) and not({'tau','block_size'}&overrides.keys())
        env=dict(clean,**{'BENCH_PARAM_'+k:v for k,v in overrides.items()})
        for kind,target in [('host',binary)]+([('actual',a.binary.resolve())] if a.binary and not ok else []):
            if kind=='actual':env.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
            p=subprocess.run([str(target)],env=env,capture_output=True,text=True,timeout=30)
            assert p.returncode==(0 if ok else 1),(overrides,p.stdout,p.stderr)
            if ok:assert p.stdout==f'{n} {t} {entries} {entries*4}\n' and not p.stderr
            else:assert not p.stdout and p.stderr.startswith('FAIL: LBM ') and len(p.stderr.splitlines())==1
            results.append(dict(kind=kind,inputs=overrides,accepted=ok))
    (a.work_dir/'report.json').write_text(json.dumps(results,indent=2)+'\n')
    print(f'PASS {len(cases)} host admission cases; {len(results)-len(cases)} actual pre-GPU rejections')
if __name__=='__main__':main()
