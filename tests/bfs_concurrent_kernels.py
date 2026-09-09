#!/usr/bin/env python3
"""Exercise the actual BFS kernels concurrently using host atomic emulation."""
import argparse
from pathlib import Path
import subprocess
import tempfile

HEADER = r'''
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <thread>
#include <vector>
#include <atomic>
#include <cassert>
#include <cstdio>
#define __global__
struct Dim {int x;};
static thread_local Dim blockIdx,blockDim,threadIdx;
static std::atomic<int> cost_writes(0),flag_writes(0);
static int* cost_target;
static int atomicExch(int* address,int value) {
    if(address==cost_target)++cost_writes;else ++flag_writes;
    return __atomic_exchange_n(address,value,__ATOMIC_RELAXED);
}
'''

MAIN = r'''

int main() {
 const int n=65,target=n-1;
 for(int repeat=0;repeat<10;++repeat) {
  int row[n+1]={},cost[n],frontier[n],visited[n],updated=0;std::vector<int> col;
  for(int p=0;p<n;++p) {
   cost[p]=p==0?0:p<target?1:-1;visited[p]=p<target;frontier[p]=p>0&&p<target;
   if(frontier[p]) {col.push_back(target);col.push_back(target);col.push_back(p);}
   row[p+1]=col.size();
  }
  cost_target=&cost[target];cost_writes=0;flag_writes=0;
  std::vector<std::thread> workers;
  for(int p=0;p<n;++p)workers.emplace_back([&,p] {
   blockIdx.x=0;blockDim.x=n;threadIdx.x=p;
   bfs_kernel(row,col.data(),cost,frontier,visited,n);
  });
  for(auto& t:workers)t.join();workers.clear();assert(cost[target]==2);
  // Several newly discovered vertices exercise the shared completion flag.
  for(int p=1;p<target;++p) {visited[p]=0;cost[p]=2;}
  for(int p=0;p<n;++p)workers.emplace_back([&,p] {
   blockIdx.x=0;blockDim.x=n;threadIdx.x=p;
   bfs_update(frontier,cost,visited,&updated,n);
  });
  for(auto& t:workers)t.join();assert(updated==1);
  assert(cost_writes==2*(n-2) && flag_writes==n-1);
 }
 puts("Concurrent extracted kernels: PASS (10 rounds, competing discoveries and completion updates)");
}
'''

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler',default='c++')
    parser.add_argument('--sanitizer',choices=('address','thread'),default='address')
    parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]/'tier2/rodinia_bfs/rodinia_bfs.hip')
    args=parser.parse_args();source=args.source.read_text()
    start=source.index('__global__ void bfs_kernel');end=source.index('// Graph generation',start)
    kernels=source[start:end]
    with tempfile.TemporaryDirectory(prefix='bfs-concurrent-') as temp:
        root=Path(temp);cpp=root/'test.cpp';binary=root/'test'
        cpp.write_text(HEADER+kernels+MAIN)
        sanitizer='thread' if args.sanitizer=='thread' else 'address,undefined'
        subprocess.run([args.compiler,'-O1','-g','-std=c++14','-pthread','-fsanitize='+sanitizer,'-fno-sanitize-recover=all',str(cpp),'-o',str(binary)],check=True,timeout=60)
        subprocess.run([str(binary)],check=True,timeout=60)
    return 0


if __name__=='__main__':raise SystemExit(main())
