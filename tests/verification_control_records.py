#!/usr/bin/env python3
"""Exercise actual verification flag resolution and JSON for unswept controls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

BENCHMARKS = ('npb_is','rodinia_gaussian','cuda_scan_large')


def section(source, start, end):
    for anchor in (start, end):
        count=source.count(anchor)
        if count!=1:
            raise ValueError(f'expected one production anchor {anchor!r}, found {count}')
    return start+source.split(start,1)[1].split(end,1)[0]


def host_source(source, name):
    if name=='cuda_scan_large':
        controls=section(source,'    const char* verify_env = getenv("BENCH_PARAM_verify");','    int N     = size;')
    else:
        end='    // rank(10)' if name=='npb_is' else '    size_t bytes_a'
        controls=section(source,'    bool do_verify = true;',end)
    start='    char params[160];' if name=='cuda_scan_large' else '    char params[128];'
    record=section(source,start,'    // Per-kernel accumulators')
    return '''#include <cstdio>
#include <cstdlib>
#include <cstring>
int main(int argc,char**argv) {
    if(argc!=2)return 2;
    int N=atoi(argv[1]),size=N,block_size=256;
    const char* env_val;
'''+controls+record+'''    printf("{%s}\\n",params);
}
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler',default='c++')
    parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    values=(None,'','false','0','skip','true','1','FALSE','False','SKIP',' skip','false ','false\n')
    clean={k:v for k,v in os.environ.items() if not k.startswith('BENCH_PARAM_')}
    count=0
    with tempfile.TemporaryDirectory(prefix='verification-controls-') as temp:
        root=Path(temp)
        for name in BENCHMARKS:
            path=args.source/'tier2'/name/(name+'.hip')
            source=path.read_text()
            cpp=root/(name+'.cpp');binary=root/name
            try:
                cpp.write_text(host_source(source,name))
            except ValueError as exc:
                raise ValueError(f'{path}: {exc}') from exc
            subprocess.run([args.compiler,'-std=c++14','-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all',str(cpp),'-o',str(binary)],check=True,timeout=60)
            for size in (12345,2147483647,-2147483648):
                for value in values:
                    env=clean.copy()
                    if value is not None:env['BENCH_PARAM_verify']=value
                    p=subprocess.run([str(binary),str(size)],env=env,capture_output=True,text=True,timeout=30)
                    if p.returncode:raise RuntimeError(p.stderr)
                    actual=json.loads(p.stdout)
                    expected=dict(size=size,verification_enabled=value not in ('false','0','skip'))
                    if name=='npb_is':expected['block_size']=256
                    if actual!=expected or type(actual.get('verification_enabled')) is not bool:
                        raise AssertionError((name,size,value,actual,expected))
                    count+=1
    print(f'{count} actual flag-resolution/JSON cases pass (host formatting test, no GPU qualification)')
    return 0


if __name__=='__main__':raise SystemExit(main())
