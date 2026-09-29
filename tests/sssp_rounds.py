#!/usr/bin/env python3
"""Compare every GPU round and counted atomic attempt with a dense snapshot model.

Use the faults binary built by sssp_verification.py. Instrumentation downloads
rounds and counts actual atomic attempts only in this test copy, outside the
production benchmark. Raw trajectories and reports remain in the work directory.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import re
import subprocess

INF = 2147483647


def graph(n, degree, fixture):
    if fixture == 'weighted':
        return [0,3,5,6,8,9,10,10], [0,1,1,2,3,3,2,4,4,6], [7,9,2,3,15,4,1,2,1,1], 0
    row, col, weight = [0], [], []
    if fixture:
        for u in range(n):
            v = u-1 if fixture == 'reverse' else u+1
            if 0 <= v < n: col.append(v); weight.append(1)
            row.append(len(col))
        return row, col, weight, n-1 if fixture == 'reverse' else 0
    state = 42
    def rng():
        nonlocal state
        state ^= (state << 13) & 0xffffffff
        state ^= state >> 17
        state ^= (state << 5) & 0xffffffff
        return state
    degrees = [max(1, degree//2 + rng() % (degree+1)) for _ in range(n)]
    for d in degrees:
        for _ in range(d):
            col.append(rng() % n); weight.append(1 + rng() % 100)
        row.append(len(col))
    return row, col, weight, 0


def model(n, degree, fixture):
    row, col, weight, source = graph(n, degree, fixture)
    dist = [INF]*n; dist[source] = 0
    rounds, attempts = [], 0
    for step in range(n):
        previous = dist[:]
        for u in range(n):
            if previous[u] == INF: continue
            for e in range(row[u], row[u+1]):
                attempts += 1
                dist[col[e]] = min(dist[col[e]], previous[u] + weight[e])
        changed = int(previous != dist)
        rounds.append([step+1, changed] + dist[:])
        if not changed: break
    return len(col), rounds, attempts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--work-dir', type=Path, required=True)
    args = ap.parse_args(); work = args.work_dir.resolve(); work.mkdir(parents=True, exist_ok=False)
    clean = {k:v for k,v in os.environ.items() if not k.startswith(('BENCH_PARAM_', 'SSSP_TEST_'))}
    cases = [(n,d,256,w,None,False) for n,d,w in itertools.product((1,2,9,1025),(4,16,64),('false','true'))]
    cases += [(257,16,b,w,None,False) for b,w in itertools.product((32,128,512),('false','true'))]
    cases += [(7 if f=='weighted' else 257,4,b,w,f,False)
              for f,b,w in itertools.product(('weighted','forward','reverse'),(32,256),('false','true'))]
    cases += [(1025,16,256,w,None,False) for w in ('false','true') for _ in range(3)]
    cases += [(9,4,256,w,None,True) for w in ('false','true')]
    results = []
    for n,d,b,w,f,bad in cases:
        label=f'{len(results):03d}-{n}-{d}-{b}-{w}-{f}-{bad}'
        env=dict(clean,BENCH_PARAM_vertices=str(n),BENCH_PARAM_degree=str(d),BENCH_PARAM_block_size=str(b),
                 BENCH_PARAM_vector_add_warmup=w,SSSP_TEST_TRACE='1')
        if f:env['SSSP_TEST_GRAPH']=f
        if bad:env['SSSP_TEST_BAD_ROUNDS']='1'
        p=subprocess.run([str(args.binary.resolve())],env=env,capture_output=True,text=True,timeout=60)
        (work/(label+'.stdout')).write_text(p.stdout);(work/(label+'.stderr')).write_text(p.stderr)
        edges, expected, attempts=model(n,d,f)
        observed=[list(map(int,line.split()[1:])) for line in p.stderr.splitlines() if line.startswith('SSSP_ROUND ')]
        count=re.search(r'^SSSP_ATTEMPTS (\d+)$',p.stderr,re.M)
        records=[json.loads(line) for line in p.stdout.splitlines()]
        passed=p.returncode==int(bad) and observed==expected and bool(count) and int(count[1])==attempts
        passed &= len(records)==3
        for record in records[:2]:
            params=record['params']
            passed &= params['graph_edges']==edges and params['iterations']==len(expected)+int(bad) and params['edge_relaxations']==attempts
        passed &= ('FAIL: BF iterations' in p.stderr)==bad
        results.append(dict(label=label,passed=bool(passed),rounds=len(expected),edge_relaxations=attempts))
        (work/'report.json').write_text(json.dumps(dict(expected=len(cases),cases=results),indent=2)+'\n')
        print(f"{'PASS' if passed else 'FAIL'} {label}: {len(expected)} rounds, {attempts} atomic attempts",flush=True)
    assert all(x['passed'] for x in results)
    print(f'{len(results)} complete GPU trajectories, counters and round-failure checks passed')


if __name__=='__main__':main()
