#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <system_error>
#include <thread>
#include <vector>
// Full independent double GEMM. Each worker owns disjoint output row tiles;
// the increasing-k accumulation order is identical for every worker count.
static std::vector<double> cpu_gemm(const float* a,const float* b,const float* initial_c,
                                    int rows,int cols,int inner,float alpha,float beta,
                                    unsigned requested_workers=0) {
    std::vector<double> result((size_t)rows*cols);
    unsigned wanted=requested_workers;
    if(!wanted)wanted=(size_t)rows*cols<512u*512u?1u:std::max(1u,std::thread::hardware_concurrency());
    const unsigned workers=std::max(1u,std::min({8u,wanted,(unsigned)((rows+3)/4)}));
    auto compute=[&](unsigned worker) {
        for(int i=(int)worker*4;i<rows;i+=(int)workers*4)
            for(int j=0;j<cols;j+=32) {
                const int nr=std::min(4,rows-i),nc=std::min(32,cols-j);
                double acc[4][32]={};
                for(int k=0;k<inner;++k)
                    for(int r=0;r<nr;++r) {
                        const double av=a[(size_t)(i+r)*inner+k];
                        for(int c=0;c<nc;++c)acc[r][c]+=av*b[(size_t)k*cols+j+c];
                    }
                for(int r=0;r<nr;++r)for(int c=0;c<nc;++c) {
                    const size_t q=(size_t)(i+r)*cols+j+c;
                    result[q]=(double)alpha*acc[r][c]+(double)beta*initial_c[q];
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
static bool gemm_value_matches(float actual,double expected) {
    const double error=std::fabs((double)actual-expected);
    return std::isfinite(actual)&&std::isfinite(expected)&&std::isfinite(error)
        &&error<=5e-4*std::max(1.0,std::fabs(expected));
}
