#pragma once
#include <algorithm>
#include <cmath>
#include <exception>
#include <thread>
#include <vector>
// Independent double fields; electric and magnetic phases join before reuse.
struct CpuFdtd { std::vector<double> ex,ey,hz; };
template<class Function> static void fdtd_rows(int rows,unsigned workers,Function compute) {
    // Reserve before starting workers. Each owns one disjoint row range.
    std::vector<std::thread> threads;threads.reserve(workers-1);unsigned worker=1;
    auto part=[&](unsigned id){compute((int)((size_t)rows*id/workers),(int)((size_t)rows*(id+1)/workers));};
    try {for(;worker<workers;++worker)threads.emplace_back(part,worker);}
    // Finish failed/uncreated partitions on the caller, then join prior workers.
    catch(const std::exception&) {for(;worker<workers;++worker)part(worker);}
    part(0);for(auto& t:threads)t.join();
}
static CpuFdtd cpu_fdtd(const float* ex,const float* ey,const float* hz,int rows,int cols,int steps,unsigned requested=0) {
    const size_t count=(size_t)rows*cols;
    CpuFdtd r{{ex,ex+count},{ey,ey+count},{hz,hz+count}};
    unsigned wanted=requested?requested:(count<512u*512u?1u:std::max(1u,std::thread::hardware_concurrency()));
    const unsigned workers=std::max(1u,std::min({16u,wanted,(unsigned)rows}));
    for(int t=0;t<steps;++t) {
        fdtd_rows(rows,workers,[&](int begin,int end) {
            for(int i=begin;i<end;++i) {
                for(int j=1;j<cols;++j) {size_t q=(size_t)i*cols+j;r.ex[q]-=0.5*(r.hz[q]-r.hz[q-1]);}
                if(i==0)for(int j=0;j<cols;++j)r.ey[j]=(float)t;
                else for(int j=0;j<cols;++j) {size_t q=(size_t)i*cols+j;r.ey[q]-=0.5*(r.hz[q]-r.hz[q-cols]);}
            }
        });
        fdtd_rows(rows,workers,[&](int begin,int end) {
            for(int i=begin;i<std::min(end,rows-1);++i)for(int j=0;j<cols-1;++j) {
                const size_t q=(size_t)i*cols+j;
                r.hz[q]-=(double)0.7f*(r.ex[q+1]-r.ex[q]+r.ey[q+cols]-r.ey[q]);
            }
        });
    }
    return r;
}
static bool fdtd_value_matches(float actual, double expected, bool exact) {
    const double error = std::fabs((double)actual - expected);
    return std::isfinite(actual) && std::isfinite(expected) && std::isfinite(error)
        && (exact ? (double)actual == expected
                  : error <= 1e-3 * std::max(1.0, std::fabs(expected)));
}
