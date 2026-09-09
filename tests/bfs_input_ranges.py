#!/usr/bin/env python3
"""Check actual BFS input guards and size arithmetic without allocating graphs."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

INT_MAX=2147483647


def section(source,start,end):
    for anchor in (start,end):
        count=source.count(anchor)
        if count!=1:raise ValueError(f'expected one source anchor {anchor!r}, found {count}')
    return start+source.split(start,1)[1].split(end,1)[0]


def host_source(source):
    parse=section(source,'static int parseIntParam','// ---------------------------------------------------------------------------\n// BFS kernels')
    prefix=section(source,'    int num_nodes      = parseIntParam','    // Generate random graph')
    row_sizes=re.findall(r'\*h_row_offsets = \(int\*\)malloc\((.+)\);',source)
    if len(row_sizes)!=1:raise ValueError('expected one row-offset allocation size')
    return '''#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
'''+parse+'''int main(int argc,char**argv) {
'''+prefix+'    size_t row_bytes='+row_sizes[0]+r''';
    printf("{\"num_nodes\":%d,\"edges_per_node\":%d,\"block_size\":%d,\"grid_blocks\":%llu,\"row_bytes\":%zu}\n",
           num_nodes,edges_per_node,block_size,grid_blocks,row_bytes);
}
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler',default='c++')
    parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]/'tier2/rodinia_bfs/rodinia_bfs.hip')
    args=parser.parse_args();clean={k:v for k,v in os.environ.items() if not k.startswith('BENCH_PARAM_')}
    values=[(1,0,1),(17,0,33),(33,5,3),(1025,80,513),(8388608,80,256),
            (0,0,1),(-1,20,256),(1,-1,256),(1,1,0),(1,1,-1),
            (INT_MAX,1,1),(INT_MAX,1,2),(INT_MAX,1,256),(INT_MAX,1,3),
            (INT_MAX-1,1,3),(INT_MAX,0,3),(1073741824,2,256),
            (1073741823,2,256),(46341,46341,256),(2,INT_MAX,1),
            (1,INT_MAX,1),(INT_MAX,1,INT_MAX),(1,1,INT_MAX)]
    manifest=json.loads((args.source.parent/'params.json').read_text())
    values += [(n,e,256) for n in manifest['scaling_param']['values'] for e in (5,20,80)]
    cases=[([],{'num_nodes':str(n),'edges_per_node':str(e),'block_size':str(b)},(n,e,b)) for n,e,b in values]
    cases += [([],{},(65536,20,256)),(['--num_nodes','0','--edges_per_node','0','--block_size','0'],{},(65536,20,256)),
              (['--num_nodes','17','--edges_per_node','5','--block_size','3'],{'num_nodes':'33','edges_per_node':'0','block_size':'32'},(33,0,32)),
              (['--num_nodes','17'],{'num_nodes':'0'},(0,20,256)),
              (['--block_size','33'],{'block_size':'0'},(65536,20,0)),
              (['--edges_per_node','5'],{'edges_per_node':'-1'},(65536,-1,256))]
    with tempfile.TemporaryDirectory(prefix='bfs-ranges-') as temp:
        root=Path(temp);cpp=root/'ranges.cpp';binary=root/'ranges'
        try:cpp.write_text(host_source(args.source.read_text()))
        except ValueError as exc:raise ValueError(f'{args.source}: {exc}') from exc
        subprocess.run([args.compiler,'-std=c++14','-O1','-g','-fsanitize=address,undefined',str(cpp),'-o',str(binary)],check=True,timeout=60)
        for argv,overrides,(n,e,b) in cases:
            env=dict(clean,**{'BENCH_PARAM_'+k:v for k,v in overrides.items()})
            p=subprocess.run([str(binary),*argv],env=env,capture_output=True,text=True,timeout=30)
            message=None
            if n<=0 or e<0 or b<=0:message='Invalid BFS parameters'
            elif n*e>INT_MAX:message='BFS edge count exceeds INT_MAX'
            elif ((n+b-1)//b)*b-1>INT_MAX:message='BFS padded vertex indices exceed INT_MAX'
            if message:
                if p.returncode!=1 or message not in p.stderr or p.stdout:raise AssertionError((argv,overrides,p.returncode,p.stdout,p.stderr))
            else:
                expected=dict(num_nodes=n,edges_per_node=e,block_size=b,grid_blocks=(n+b-1)//b,row_bytes=(n+1)*4)
                if p.returncode or json.loads(p.stdout)!=expected:raise AssertionError((argv,overrides,p.returncode,p.stdout,p.stderr,expected))
    print(f'{len(cases)} BFS resolved-range/size cases pass (no graph allocation or GPU access)')
    return 0


if __name__=='__main__':raise SystemExit(main())
