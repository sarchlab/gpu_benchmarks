#pragma once
// Rodinia 3.1 HotSpot3D physics; pin and notices in README.md / UPSTREAM_LICENSE.
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace hotspot3d {
struct Problem { int n=64, block=8, steps=10; float ambient=80.0f; double requested_ambient=80.0; };
inline int integer(const char* text) {
    if(!text || !*text)throw std::invalid_argument("empty integer parameter");
    errno=0;char* end=nullptr;const long value=std::strtol(text,&end,10);
    if(errno || *end || value<1 || value>INT_MAX)
        throw std::invalid_argument("expected positive 32-bit integer parameter");
    return static_cast<int>(value);
}
inline double real(const char* text) {
    if(!text || !*text)throw std::invalid_argument("empty ambient temperature");
    errno=0;char* end=nullptr;const double value=std::strtod(text,&end);
    if(errno || *end || !std::isfinite(value) || !std::isfinite(static_cast<float>(value))
       || (value!=0 && static_cast<float>(value)==0))throw std::invalid_argument("expected finite ambient temperature");
    return value;
}
inline Problem parameters(int argc,char** argv) {
    Problem p;const char* names[]={"grid_size","block_size","num_iterations","amb_temp"};
    const char* values[4]={nullptr,nullptr,nullptr,nullptr};
    for(int i=1;i<argc;++i) {
        int k=0;for(;k<4;++k)if(std::strncmp(argv[i],"--",2)==0 && std::strcmp(argv[i]+2,names[k])==0)break;
        if(k==4 || i+1==argc)throw std::invalid_argument("expected --grid_size, --block_size, --num_iterations or --amb_temp and a value");
        const char* value=argv[++i];
        // Validate every supplied value, but preserve first matching flag semantics.
        if(k==3)real(value);else integer(value);
        if(!values[k])values[k]=value;
    }
    for(int k=0;k<4;++k) {
        char name[64];std::snprintf(name,sizeof(name),"BENCH_PARAM_%s",names[k]);
        if(const char* value=std::getenv(name))values[k]=value;
    }
    if(values[0])p.n=integer(values[0]);
    if(values[1])p.block=integer(values[1]);
    if(values[2])p.steps=integer(values[2]);
    if(values[3]) {p.requested_ambient=real(values[3]);p.ambient=static_cast<float>(p.requested_ambient);}
    // Bound n before cubing, and block before multiplying, to avoid overflow.
    if(p.n>1290)throw std::invalid_argument("grid_size must be 1..1290 (32-bit cell indices)");
    if(p.block>10)throw std::invalid_argument("block_size cubed must not exceed 1024 threads");
    return p;
}
struct Coefficients { float cap,rx,ry,rz,dt,upstream_dt,sdc,ce,cw,cn,cs,ct,cb,cc; };
inline double row_norm(const Coefficients& c);
inline Coefficients coefficients(int n) {
    // Preserve pinned host expressions, including double literal intermediates.
    const float t_chip=0.0005, chip_height=0.016, chip_width=0.016;
    const float dx=chip_height/n, dy=chip_width/n, dz=t_chip/n;
    Coefficients c;
    c.cap=0.5*1.75e6*t_chip*dx*dy;
    c.rx=dy/(2.0*100*t_chip*dx);
    c.ry=dx/(2.0*100*t_chip*dy);
    c.rz=dz/(100*dx*dy);
    const float max_slope=3.0e6/(0.5*t_chip*1.75e6);
    c.dt=c.upstream_dt=0.001/max_slope;
    auto weights=[&]() {
        c.sdc=c.dt/c.cap;
        c.ce=c.cw=c.sdc/c.rx;c.cn=c.cs=c.sdc/c.ry;c.ct=c.cb=c.sdc/c.rz;
        c.cc=1.0-(2.0*c.ce+2.0*c.cn+3.0*c.ct);
    };
    weights();
    if(row_norm(c)>1.0) {
        // Owner policy: change only grids that fail the non-amplification bound.
        // Positive center with 10% margin also makes every clamped row safe.
        c.dt=static_cast<float>(0.9*c.cap/(2.0/c.rx+2.0/c.ry+3.0/c.rz));
        weights();
    }
    return c;
}
// Outward rounding for the nonnegative error-bound arithmetic.
inline double add_up(double a,double b) { return std::nextafter(a+b,INFINITY); }
inline double mul_up(double a,double b) { return std::nextafter(a*b,INFINITY); }
inline double row_norm(const Coefficients& c) {
    double g=0;for(float v:{c.cc,c.cw,c.ce,c.cs,c.cn,c.cb,c.ct})g=add_up(g,std::fabs(v));
    return g;
}
inline void admissible(const Coefficients& c) {
    // A sufficient infinity-norm stability condition, including clamped rows.
    if(!std::isfinite(row_norm(c)) || row_norm(c)>1.0 || !std::isfinite(c.sdc))
        throw std::invalid_argument("HotSpot3D grid has amplifying or non-finite stencil coefficients");
}
struct Input { std::vector<float> temperature,power; };
inline Input input(const Problem& p) {
    const size_t cells=static_cast<size_t>(p.n)*p.n*p.n;
    Input in;in.temperature.resize(cells);in.power.resize(cells);
    std::srand(42);
    for(size_t i=0;i<cells;++i) {
        in.temperature[i]=p.ambient+static_cast<float>(std::rand()%200)/10.0f;
        in.power[i]=static_cast<float>(std::rand()%100)/500.0f;
        if(!std::isfinite(in.temperature[i]))throw std::invalid_argument("non-finite generated input");
    }
    return in;
}
inline double error_bound(const Problem& p,const Coefficients& c,const Input& in) {
    double magnitude=0,power=0;
    for(float v:in.temperature)magnitude=std::max(magnitude,std::fabs(static_cast<double>(v)));
    for(float v:in.power)power=std::max(power,std::fabs(static_cast<double>(v)));
    const double g=row_norm(c);
    const double source=add_up(mul_up(std::fabs(c.sdc),power),mul_up(std::fabs(c.ct),std::fabs(p.ambient)));
    // Each recurrence has 9 products and 8 additions; FMA only reduces rounding.
    // 18 sites conservatively cover both the FP32 GPU and FP64 reference.
    const double uf=std::ldexp(1.0,-24),ud=std::ldexp(1.0,-53);
    const double gf=std::nextafter(18*uf/(1-18*uf),INFINITY);
    const double gd=std::nextafter(18*ud/(1-18*ud),INFINITY);
    // Also cover underflow/flush-to-zero at any of the FP32 sites.
    const double tiny=18*static_cast<double>(std::numeric_limits<float>::min());
    double error=0;
    for(int t=0;t<p.steps;++t) {
        const double next=add_up(mul_up(g,magnitude),source);
        const double propagated=mul_up(g,error);
        error=add_up(add_up(propagated,mul_up(gf,add_up(next,propagated))),add_up(mul_up(gd,next),tiny));
        magnitude=add_up(next,add_up(mul_up(gd,next),tiny));
        if(!std::isfinite(error) || add_up(magnitude,error)>std::numeric_limits<float>::max())
            throw std::invalid_argument("HotSpot3D temperature range cannot be verified without overflow");
    }
    return error;
}
inline std::vector<double> reference(const Problem& p,const Coefficients& c,const Input& in) {
    const int n=p.n;const size_t plane=static_cast<size_t>(n)*n;
    std::vector<double> current(in.temperature.begin(),in.temperature.end()),next(current.size());
    for(int t=0;t<p.steps;++t) {
        for(int z=0;z<n;++z)for(int y=0;y<n;++y)for(int x=0;x<n;++x) {
            const size_t i=static_cast<size_t>(z)*plane+static_cast<size_t>(y)*n+x;
            const size_t west=x?i-1:i,east=x+1<n?i+1:i;
            const size_t north=y?i-n:i,south=y+1<n?i+n:i;
            const size_t bottom=z?i-plane:i,top=z+1<n?i+plane:i;
            // FP64, explicit scalar clamping; no GPU helper or GPU auxiliary run.
            next[i]=c.cc*current[i]+c.cw*current[west]+c.ce*current[east]
                    +c.cs*current[south]+c.cn*current[north]+c.cb*current[bottom]
                    +c.ct*current[top]+static_cast<double>(c.sdc)*in.power[i]
                    +static_cast<double>(c.ct)*p.ambient;
        }
        current.swap(next);
    }
    return current;
}
inline size_t compare(const std::vector<float>& actual,const std::vector<double>& expected,double bound) {
    if(actual.size()!=expected.size() || !std::isfinite(bound) || bound<0)
        throw std::invalid_argument("invalid HotSpot3D verification operands");
    size_t errors=0;
    for(size_t i=0;i<actual.size();++i) {
        const double difference=std::fabs(static_cast<double>(actual[i])-expected[i]);
        if(!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || !std::isfinite(difference) || difference>bound) {
            if(errors<10)std::fprintf(stderr,"Mismatch temperature[%zu]: GPU=%.9g CPU=%.17g bound=%.9g\n",i,actual[i],expected[i],bound);
            ++errors;
        }
    }
    return errors;
}
} // namespace hotspot3d
