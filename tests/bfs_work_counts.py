#!/usr/bin/env python3
"""Count actual BFS GPU operations and compare every level with CPU snapshots."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import time

DUMP = r'''
static void bfs_test_dump(int n,int edges,const int* row,const int* col) {
    const char* path=getenv("BFS_TEST_CSR");if(!path)abort();
    FILE* file=fopen(path,"wb");if(!file)abort();int header[]={n,edges};
    if(fwrite(header,sizeof(int),2,file)!=2 || fwrite(row,sizeof(int),n+1,file)!=(size_t)n+1
       || fwrite(col,sizeof(int),edges,file)!=(size_t)edges || fclose(file))abort();
}
'''


def replace_checked(source, old, new, expected=1):
    count = source.count(old)
    if count != expected:
        raise ValueError(f"expected {expected} occurrences of production anchor {old!r}, found {count}")
    return source.replace(old, new)


def instrument(source,graph_hook):
    header='#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)\n'
    source=replace_checked(source,header,header+graph_hook+DUMP)
    source=replace_checked(source,'    // Host arrays for BFS state','    bfs_test_graph(num_nodes,h_row_offsets,&h_col_indices,&num_edges);\n    bfs_test_dump(num_nodes,num_edges,h_row_offsets,h_col_indices);\n\n    // Host arrays for BFS state')
    signature='int        num_nodes) {'
    source=replace_checked(source,signature,'int        num_nodes, unsigned long long* work_counts) {',expected=2)
    source=replace_checked(source,'            int nb = col_indices[e];','            atomicAdd(work_counts,1ULL);\n            int nb = col_indices[e];')
    source=replace_checked(source,'                atomicExch(&cost[nb], new_cost);','                atomicAdd(work_counts+1,1ULL);\n                atomicExch(&cost[nb], new_cost);')
    source=replace_checked(source,'            atomicExch(updated, 1);','            atomicAdd(work_counts+2,1ULL);\n            atomicExch(updated, 1);')
    source=replace_checked(source,'    // Device allocations','''    unsigned long long* d_test_counts;
    HIP_CHECK(hipMalloc(&d_test_counts,3*sizeof(unsigned long long)));
    HIP_CHECK(hipMemset(d_test_counts,0,3*sizeof(unsigned long long)));
    // Device allocations''')
    source=replace_checked(source,'d_frontier, d_visited, num_nodes);','d_frontier, d_visited, num_nodes, d_test_counts);')
    source=replace_checked(source,'d_visited, d_updated, num_nodes);','d_visited, d_updated, num_nodes, d_test_counts);')
    source=replace_checked(source,'            ++bfs_levels;','''            ++bfs_levels;
            unsigned long long observed[3];
            HIP_CHECK(hipMemcpy(observed,d_test_counts,sizeof(observed),hipMemcpyDeviceToHost));
            fprintf(stderr,"BFS_TEST_WORK %llu %llu %llu %llu\\n",bfs_levels,observed[0],observed[1],observed[2]);''')
    source=replace_checked(source,'    // Cleanup','    HIP_CHECK(hipFree(d_test_counts));\n    // Cleanup')
    return source


def expected_work(path):
    data=path.read_bytes();n,edges=struct.unpack_from('=ii',data)
    assert len(data)==4*(2+n+1+edges)
    row=struct.unpack_from(f'={n+1}i',data,8)
    col=struct.unpack_from(f'={edges}i',data,8+4*(n+1))
    visited={0};frontier={0};levels=[];counts=[0,0,0]
    while frontier:
        targets=set()
        for u in sorted(frontier):
            for at in range(row[u],row[u+1]):
                v=col[at];counts[0]+=1
                if v not in visited:counts[1]+=1;targets.add(v)
        counts[2]+=len(targets);visited|=targets;frontier=targets
        levels.append([len(levels)+1,*counts])
    return levels


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--platform',choices=('cuda','rocm'),required=True)
    parser.add_argument('--compile-only',action='store_true')
    args=parser.parse_args();args.source=args.source.resolve();args.work_dir=args.work_dir.resolve()
    try:args.work_dir.mkdir(parents=True,exist_ok=False)
    except FileExistsError:parser.error('--work-dir must be new')
    spec=importlib.util.spec_from_file_location('bfs_graphs',args.source/'tests/bfs_measured_result.py')
    graphs=importlib.util.module_from_spec(spec);spec.loader.exec_module(graphs)
    src=args.source/'tier2/rodinia_bfs';dst=args.work_dir/'tier2/rodinia_bfs';dst.mkdir(parents=True)
    shutil.copytree(args.source/'common',args.work_dir/'common')
    for name in ('Makefile','params.json','cpu_reference.h'):shutil.copy2(src/name,dst/name)
    source=(src/'rodinia_bfs.hip').read_text();(dst/'rodinia_bfs.hip').write_text(instrument(source,graphs.HOOK))
    with (args.work_dir/'build.log').open('w') as log:
        subprocess.run(['make','PLATFORM='+args.platform],cwd=dst,stdout=log,stderr=subprocess.STDOUT,check=True)
    if args.compile_only:print('BFS operation-counter regression compiled; GPU checks not run');return 0
    clean={k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_','BFS_TEST_'))}
    cases=[(65,20,graph,block,rep) for graph in graphs.GRAPHS for block in (32,64,128,256,512) for rep in range(3)]
    cases += [(1024,degree,'random',block,rep) for degree in (5,20,80) for block in (32,64,128,256,512) for rep in range(3)]
    rows=[];identities={}
    for n,degree,graph,block,rep in cases:
        label=f'{n}-{degree}-{graph}-{block}-{rep}';csr=args.work_dir/(label+'.csr')
        env=dict(clean,BENCH_PARAM_num_nodes=str(n),BENCH_PARAM_edges_per_node=str(degree),BENCH_PARAM_block_size=str(block),BENCH_PARAM_vector_add_warmup='false',BFS_TEST_CSR=str(csr))
        if graph!='random':env['BFS_TEST_GRAPH']=graph
        start=time.monotonic()
        try:
            p=subprocess.run([str(dst/'rodinia_bfs')],env=env,capture_output=True,text=True,timeout=60)
            code,stdout,stderr=p.returncode,p.stdout,p.stderr
        except subprocess.TimeoutExpired as exc:
            def decoded(value):return value.decode(errors='replace') if isinstance(value,bytes) else value or ''
            code,stdout,stderr='timeout',decoded(exc.stdout),decoded(exc.stderr)
        for suffix,text in [('stdout',stdout),('stderr',stderr)]:(args.work_dir/(label+'.'+suffix)).write_text(text)
        passed=code==0 and 'BFS verification: PASS' in stderr
        expected=expected_work(csr) if csr.exists() else []
        observed=[list(map(int,m)) for m in re.findall(r'BFS_TEST_WORK (\d+) (\d+) (\d+) (\d+)',stderr)]
        passed &= bool(expected) and expected==observed
        try:
            records=[json.loads(line) for line in stdout.splitlines()];params=records[0]['params']
            passed &= [params[k] for k in ('bfs_levels','frontier_edges','discovery_atomics','promotion_atomics')]==expected[-1]
            passed &= records[1]['params']==params and params['discovery_policy']=='level_visited_atomic_v1'
        except (IndexError,KeyError,TypeError,ValueError):passed=False
        identity=(n,degree,graph)
        if identity in identities:passed &= identities[identity]==observed
        else:identities[identity]=observed
        rows.append(dict(case=label,passed=bool(passed),seconds=time.monotonic()-start,expected=expected,observed=observed))
        report=dict(complete=len(rows)==len(cases),total=len(rows),expected_cases=len(cases),passed=sum(r['passed'] for r in rows),results=rows)
        (args.work_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if not passed:print('FAIL',label,code,stderr[-1200:],flush=True)
    print(f"{report['passed']}/{report['total']} exact per-level GPU operation counts pass")
    return 0 if report['passed']==report['expected_cases'] else 1


if __name__=='__main__':raise SystemExit(main())
