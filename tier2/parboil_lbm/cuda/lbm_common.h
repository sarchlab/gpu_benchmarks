// (C) Copyright 2010 The Board of Trustees of the University of Illinois.
// Adapted from Parboil a4c63da28f45699e9a274cbc49af3e067162aa2b.
// See UPSTREAM_LICENSE and README.md for provenance and adaptations.
#pragma once
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace lbm {
enum { C,N,S,E,W,T,B,NE,NW,SE,SW,NT,NB,ST,SB,ET,EB,WT,WB,FLAGS,Q=19 };
enum { OBSTACLE=1, ACCEL=2 };
constexpr float omega=1.95f;
constexpr double population_tolerance=1e-5, velocity_tolerance=1e-5;
struct Parameters { int n=64,steps=100; };
inline Parameters parameters() {
    Parameters p;
    if(const char* v=std::getenv("BENCH_PARAM_grid_dim"))p.n=std::atoi(v);
    if(const char* v=std::getenv("BENCH_PARAM_num_timesteps"))p.steps=std::atoi(v);
    if(std::getenv("BENCH_PARAM_block_size") || std::getenv("BENCH_PARAM_tau"))
        throw std::invalid_argument("LBM block_size and tau are fixed by upstream (block=grid_dim, omega=1.95); remove obsolete overrides");
    if(p.n<3 || p.steps<=0)throw std::invalid_argument("LBM requires grid_dim>=3 and positive num_timesteps");
    // Bound first, before multiplication. Both layouts use signed offsets.
    if(p.n>483 || 20LL*(p.n+8)*p.n*(p.n+8)>INT_MAX)
        throw std::invalid_argument("LBM padded distribution storage exceeds INT_MAX");
    if(p.n==256 && p.steps==500)
        throw std::invalid_argument("LBM 256^3/500 is excluded from collection: exceeds upstream velocity error limit");
    return p;
}
struct Layout {
    int n,px,plane,cells,margin,entries;
    explicit Layout(int size):n(size),px(n+8),plane(px*n),cells(plane*(n+4)),
        margin(2*plane),entries(20*cells+40*margin) {}
    size_t bytes() const {return size_t(entries)*sizeof(float);}
    int index(int x,int y,int z,int q) const {return margin+q*cells+x+y*px+z*plane;}
};
inline unsigned flags(int x,int y,int z,int n) {
    if(x==0 || x==n-1 || y==0 || y==n-1 || z==0 || z==n-1)return OBSTACLE;
    return (z==1 || z==n-2) && x>1 && x<n-2 && y>1 && y<n-2 ? ACCEL:0;
}
inline float weight(int q){return q==C ? 1.0f/3.0f : q<=B ? 1.0f/18.0f : 1.0f/36.0f;}
inline std::vector<float> input(const Layout& l) {
    std::vector<float> a(l.entries,0.0f);
    for(int z=0;z<l.n;++z)for(int y=0;y<l.n;++y)for(int x=0;x<l.n;++x)
        for(int q=0;q<Q;++q)a[l.index(x,y,z,q)]=weight(q);
    // Upstream initializes wall flags in two exterior z planes as well.
    for(int z=-2;z<l.n+2;++z)for(int y=0;y<l.n;++y)for(int x=0;x<l.n;++x) {
        const unsigned f=flags(x,y,z,l.n);std::memcpy(&a[l.index(x,y,z,FLAGS)],&f,sizeof(f));
    }
    return a;
}
struct CPUGrid {
    std::vector<float> values;
    Layout layout;
    explicit CPUGrid(const Layout& l):values(l.entries,0.0f),layout(l) {
        for(int z=0;z<l.n;++z)for(int y=0;y<l.n;++y)for(int x=0;x<l.n;++x)
            for(int q=0;q<Q;++q)at(x,y,z,q)=weight(q);
    }
    float& at(int x,int y,int z,int q){return values[20*(layout.margin+x+y*layout.px+z*layout.plane)+q];}
    const float& at(int x,int y,int z,int q) const {return values[20*(layout.margin+x+y*layout.px+z*layout.plane)+q];}
};
// Match the upstream FP32 velocity output recipe, including summation order.
inline void velocity(const float* f,float* v) {
    float rho=+f[C]+f[N]+f[S]+f[E]+f[W]+f[T]+f[B]+f[NE]+f[NW]+f[SE]+f[SW]+f[NT]+f[NB]+f[ST]+f[SB]+f[ET]+f[EB]+f[WT]+f[WB];
    v[0]=(+f[E]-f[W]+f[NE]-f[NW]+f[SE]-f[SW]+f[ET]+f[EB]-f[WT]-f[WB])/rho;
    v[1]=(+f[N]-f[S]+f[NE]+f[NW]-f[SE]-f[SW]+f[NT]+f[NB]-f[ST]-f[SB])/rho;
    v[2]=(+f[T]-f[B]+f[NT]-f[NB]+f[ST]-f[SB]+f[ET]-f[EB]+f[WT]-f[WB])/rho;
}
struct Comparison {size_t errors=0,populations=0,cells=0;double max_population=0,max_velocity=0;};
inline Comparison compare(const Layout& l,const std::vector<float>& actual,const CPUGrid& expected) {
    Comparison r;
    if(actual.size()!=size_t(l.entries) || expected.layout.n!=l.n){r.errors=1;return r;}
    for(int z=0;z<l.n;++z)for(int y=0;y<l.n;++y)for(int x=0;x<l.n;++x) {
        float a[Q],b[Q],av[3],bv[3];
        for(int q=0;q<Q;++q) {
            a[q]=actual[l.index(x,y,z,q)];b[q]=expected.at(x,y,z,q);
            const double error=std::fabs(double(a[q])-b[q]);
            if(!std::isfinite(a[q]) || !std::isfinite(b[q]) || error>population_tolerance)++r.errors;
            r.max_population=std::max(r.max_population,error);++r.populations;
        }
        unsigned f;std::memcpy(&f,&actual[l.index(x,y,z,FLAGS)],sizeof(f));
        if(f!=flags(x,y,z,l.n))++r.errors;
        velocity(a,av);velocity(b,bv);
        double sum=0;
        for(int k=0;k<3;++k){const float delta=av[k]-bv[k];sum+=double(delta)*delta;}
        const double error=std::sqrt(sum);
        if(!std::isfinite(error) || error>velocity_tolerance)++r.errors;
        r.max_velocity=std::max(r.max_velocity,error);++r.cells;
    }
    return r;
}
} // namespace lbm
