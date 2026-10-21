#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <system_error>
#include <thread>
#include <vector>
// Full independent double matrix product. Each worker owns disjoint output row tiles;
// the increasing-k accumulation order is identical for every worker count.
template<class Left,class Right>
static std::vector<double> cpu_product(const Left* a,const Right* b,
                                       int rows,int cols,int inner,unsigned requested_workers=0) {
    std::vector<double> result((size_t)rows*cols);
    // Pack each right-hand column tile contiguously in reduction order.
    // This preserves every input value and product order while avoiding a
    // full-matrix row stride between consecutive reduction samples.
    std::vector<Right> packed((size_t)inner*cols);
    for(int j=0;j<cols;j+=32) {
        const int width=std::min(32,cols-j);
        for(int k=0;k<inner;++k)
            for(int c=0;c<width;++c)
                packed[(size_t)j*inner+(size_t)k*width+c]=b[(size_t)k*cols+j+c];
    }
    unsigned wanted=requested_workers;
    if(!wanted)wanted=(size_t)rows*cols<512u*512u?1u:std::max(1u,std::thread::hardware_concurrency());
    const unsigned workers=std::max(1u,std::min({16u,wanted,(unsigned)((rows+3)/4)}));
    auto compute=[&](unsigned worker) {
        for(int i=(int)worker*4;i<rows;i+=(int)workers*4)
            for(int j=0;j<cols;j+=32) {
                const int nr=std::min(4,rows-i),nc=std::min(32,cols-j);
                double acc[4][32]={};
                for(int k=0;k<inner;++k)
                    for(int r=0;r<nr;++r) {
                        const double av=a[(size_t)(i+r)*inner+k];
                        for(int c=0;c<nc;++c)acc[r][c]+=av*packed[(size_t)j*inner+(size_t)k*nc+c];
                    }
                for(int r=0;r<nr;++r)for(int c=0;c<nc;++c) {
                    const size_t q=(size_t)(i+r)*cols+j+c;
                    result[q]=acc[r][c];
                }
            }
    };
    std::vector<std::thread> threads;
    threads.reserve(workers-1);
    unsigned worker=1;
    try {
        for(;worker<workers;++worker)threads.emplace_back(compute,worker);
    } catch(const std::exception&) {
        // Both system errors and thread-state allocation failures can prevent
        // creation. Complete those partitions on the caller; already
        // running threads still own their original disjoint partitions.
        for(;worker<workers;++worker)compute(worker);
    }
    compute(0);
    for(auto& thread:threads)thread.join();
    return result;
}
struct Cpu3mm { std::vector<double> e,f,g; };
static Cpu3mm cpu_3mm(const float* a,const float* b,const float* c,const float* d,
                     int ni,int nj,int nk,int nl,int nm,unsigned workers=0) {
    Cpu3mm ref;
    ref.e=cpu_product(a,b,ni,nj,nk,workers);
    ref.f=cpu_product(c,d,nj,nl,nm,workers);
    ref.g=cpu_product(ref.e.data(),ref.f.data(),ni,nl,nj,workers);
    return ref;
}
static bool mm3_value_matches(float actual,double expected,bool final_stage) {
    const double error=std::fabs((double)actual-expected);
    const double tolerance=final_stage?2e-3:5e-4;
    return std::isfinite(actual)&&std::isfinite(expected)&&std::isfinite(error)
        &&error<=tolerance*std::max(1.0,std::fabs(expected));
}
