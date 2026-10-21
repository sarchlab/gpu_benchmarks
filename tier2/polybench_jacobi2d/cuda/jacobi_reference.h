#pragma once
// PolyBench/GPU Jacobi-2D problem and independent CPU reference.
// Adapted from the pin in README.md; see UPSTREAM_LICENSE for notices.
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace jacobi {
struct Problem {int n=1024,steps=50;};
inline int integer(const char* value) {
    if(!value || !*value)throw std::invalid_argument("empty Jacobi integer parameter");
    char* end=nullptr;errno=0;const long parsed=std::strtol(value,&end,10);
    if(errno || *end || parsed<INT_MIN || parsed>INT_MAX)
        throw std::invalid_argument("invalid Jacobi integer parameter");
    return static_cast<int>(parsed);
}
inline Problem parameters(int argc,char** argv) {
    Problem p;bool have_n=false,have_steps=false;
    for(int i=1;i<argc;++i) {
        const bool is_n=std::strcmp(argv[i],"--n")==0;
        const bool is_steps=std::strcmp(argv[i],"--tsteps")==0;
        if((!is_n && !is_steps) || i+1==argc)
            throw std::invalid_argument("expected --n N or --tsteps T");
        const int value=integer(argv[++i]);
        // Preserve the original first matching CLI flag and environment precedence.
        if(is_n && !have_n) {p.n=value;have_n=true;}
        if(is_steps && !have_steps) {p.steps=value;have_steps=true;}
    }
    if(const char* v=std::getenv("BENCH_PARAM_n"))p.n=integer(v);
    if(const char* v=std::getenv("BENCH_PARAM_tsteps"))p.steps=integer(v);
    if(p.n<3 || static_cast<int64_t>(p.n)*p.n>INT_MAX)
        throw std::invalid_argument("Jacobi n must be 3..46340 (32-bit cell indices)");
    if(p.steps<1)throw std::invalid_argument("Jacobi tsteps must be positive");
    return p;
}
struct Grids {
    std::vector<float> a,b;
    explicit Grids(int n):a(static_cast<size_t>(n)*n),b(a.size()){}
};
inline Grids input(int n) {
    Grids grids(n);
    for(int i=0;i<n;++i)for(int j=0;j<n;++j) {
        grids.a[static_cast<size_t>(i)*n+j]=(static_cast<float>(i)*(j+2)+10)/n;
        grids.b[static_cast<size_t>(i)*n+j]=(static_cast<float>(i-4)*(j-1)+11)/n;
    }
    return grids;
}
inline Grids reference(const Problem& p) {
    Grids grids=input(p.n);
    const int n=p.n;
    for(int t=0;t<p.steps;++t) {
        for(int i=1;i<n-1;++i)for(int j=1;j<n-1;++j) {
            const size_t k=static_cast<size_t>(i)*n+j;
            grids.b[k]=0.2f*(grids.a[k]+grids.a[k-1]+grids.a[k+1]+grids.a[k+n]+grids.a[k-n]);
        }
        for(int i=1;i<n-1;++i)
            std::copy_n(grids.b.data()+static_cast<size_t>(i)*n+1,n-2,
                        grids.a.data()+static_cast<size_t>(i)*n+1);
    }
    return grids;
}
inline size_t compare(const Grids& actual,const Grids& expected) {
    if(actual.a.size()!=expected.a.size() || actual.b.size()!=expected.b.size())
        throw std::invalid_argument("Jacobi verification grid sizes differ");
    size_t errors=0;
    const std::vector<float>* got[]={&actual.a,&actual.b};
    const std::vector<float>* ref[]={&expected.a,&expected.b};
    for(int array=0;array<2;++array)for(size_t i=0;i<got[array]->size();++i) {
        const float a=(*got[array])[i],b=(*ref[array])[i];
        // Same per-cell FP32 operation order, without reductions or reassociation.
        if(!std::isfinite(a) || !std::isfinite(b) || a!=b) {
            if(errors<10)std::fprintf(stderr,"Mismatch %c[%zu]: GPU=%.9g CPU=%.9g\n",'A'+array,i,a,b);
            ++errors;
        }
    }
    return errors;
}
} // namespace jacobi
