// Host input and independent FP64 finite-volume reference for the Altis CFD port.
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

namespace cfd {
constexpr int steps=10, stages=3, fields=5, faces=4;
struct Problem { int cells,block,padded; };
inline int integer(const char* s) {
    if(!s || !*s)throw std::invalid_argument("CFD expects a positive integer");
    for(const char* p=s;*p;++p)if(*p<'0'||*p>'9')throw std::invalid_argument("CFD expects decimal digits only");
    errno=0;char* end=nullptr;long n=std::strtol(s,&end,10);
    if(errno || *end || n<1 || n>INT_MAX)throw std::invalid_argument("CFD integer is out of range");
    return static_cast<int>(n);
}
inline Problem parameters(int argc,char** argv) {
    int n=97000,b=256;
    for(int i=1;i<argc;++i) {
        const bool size=std::strcmp(argv[i],"--size")==0,block=std::strcmp(argv[i],"--block_size")==0;
        if((!size&&!block)||i+1==argc)throw std::invalid_argument("CFD accepts --size N and --block_size B");
        int v=integer(argv[++i]);if(size)n=v;else b=v;
    }
    if(const char* s=std::getenv("BENCH_PARAM_size"))n=integer(s);
    if(const char* s=std::getenv("BENCH_PARAM_block_size"))b=integer(s);
    if(b>1024)throw std::invalid_argument("CFD block_size must be in 1..1024");
    // Pinned kernels index 12 padded normal planes with signed int arithmetic.
    const long long padded=(static_cast<long long>(n)+b-1)/b*b;
    if(padded>INT_MAX/12)throw std::invalid_argument("CFD padded normal index exceeds INT_MAX");
    return {n,b,static_cast<int>(padded)};
}
using Point=std::array<int,3>;
using Tet=std::array<Point,4>;
using Face=std::array<Point,3>;
inline Point subtract(Point a,Point b) { return {{a[0]-b[0],a[1]-b[1],a[2]-b[2]}}; }
inline Point cross(Point a,Point b) { return {{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}}; }
inline int dot(Point a,Point b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
inline Face face(Tet t,int opposite,Point shift={{0,0,0}}) {
    Face f;int j=0;
    for(int v=0;v<4;++v)if(v!=opposite) {
        for(int k=0;k<3;++k)f[j][k]=t[v][k]+shift[k];++j;
    }
    std::sort(f.begin(),f.end());return f;
}
struct Neighbor { Point offset;int type; };
struct Topology {
    std::array<Tet,6> tetra;
    std::array<std::array<Neighbor,4>,6> adjacent;
    std::array<std::array<std::array<float,3>,4>,6> normal;
    Topology() {
        Point permutation={{0,1,2}};int t=0;
        do {
            Tet q={{{{0,0,0}},{{0,0,0}},{{0,0,0}},{{1,1,1}}}};
            q[1][permutation[0]]=1;q[2]=q[1];q[2][permutation[1]]=1;
            tetra[t++]=q;
        } while(std::next_permutation(permutation.begin(),permutation.end()));
        for(t=0;t<6;++t)for(int f=0;f<4;++f) {
            Face key=face(tetra[t],f);
            Point a=cross(subtract(key[1],key[0]),subtract(key[2],key[0]));
            const int sign=dot(a,subtract(tetra[t][f],key[0]))>0?1:-1;
            for(int k=0;k<3;++k)normal[t][f][k]=0.5f*sign*a[k]; // inward area vector
            int matches=0;
            for(int z=-1;z<=1;++z)for(int y=-1;y<=1;++y)for(int x=-1;x<=1;++x)
                for(int other=0;other<6;++other) {
                    if(x==0&&y==0&&z==0&&other==t)continue;
                    for(int g=0;g<4;++g)if(face(tetra[other],g,{{x,y,z}})==key) {
                        adjacent[t][f]={{{x,y,z}},other};++matches;
                    }
                }
            if(matches!=1)throw std::logic_error("CFD tetrahedral face must have one neighbor");
        }
    }
};
inline const Topology& topology() { static const Topology value;return value; }
inline std::array<float,5> farfield() {
    // Preserve the pinned host expressions, including FP32 gamma and the FP32 gamma-minus-one denominator.
    std::array<float,5> u;u[0]=float(1.4);
    float speed_of_sound=std::sqrt(1.4f*float(1.0f)/u[0]);
    float speed=float(1.2f)*speed_of_sound;
    u[1]=u[0]*speed;u[2]=0.0f;u[3]=0.0f;
    u[4]=u[0]*(float(0.5f)*(speed*speed))+(float(1.0f)/float(1.4f-1.0f));return u;
}
struct Input {
    int side;
    std::vector<int> neighbors;
    std::vector<float> normals,areas,initial;
};
inline Input input(const Problem& p) {
    const size_t n=p.padded;int side=1;
    while(6LL*side*side*side<p.cells)++side;
    Input in{side,std::vector<int>(4*n),std::vector<float>(12*n),std::vector<float>(n,1.0f/6.0f),std::vector<float>(5*n)};
    const auto& top=topology();
    for(int i=0;i<p.cells;++i) {
        const int cube=i/6,t=i%6,x=cube%side,y=(cube/side)%side,z=cube/(side*side);
        for(int f=0;f<4;++f) {
            const Neighbor& a=top.adjacent[t][f];int xx=x+a.offset[0],yy=y+a.offset[1],zz=z+a.offset[2];
            long long other=6LL*((static_cast<long long>(zz)*side+yy)*side+xx)+a.type;
            int neighbor=-2;
            if(xx>=0&&xx<side&&yy>=0&&yy<side&&zz>=0&&zz<side&&other>=0&&other<p.cells)neighbor=static_cast<int>(other);
            // The y=0 plane is a slip wall, parallel to the incoming Mach1.2 flow.
            Face q=face(top.tetra[t],f);
            if(neighbor<0&&y==0&&q[0][1]==0&&q[1][1]==0&&q[2][1]==0)neighbor=-1;
            in.neighbors[i+f*n]=neighbor;
            for(int k=0;k<3;++k)in.normals[i+(f+k*4)*n]=top.normal[t][f][k];
        }
        // Nonuniform, dyadic perturbation; no libc RNG or transcendental input recipe.
        const float perturb=static_cast<float>((11*x+7*y+3*z+t)%33-16)/256.0f;
        const float density=1.4f+perturb,pressure=1.0f+0.5f*perturb,speed=1.2f;
        in.initial[i]=density;in.initial[i+n]=density*speed;
        in.initial[i+2*n]=in.initial[i+3*n]=0.0f;
        in.initial[i+4*n]=density*(0.5f*(speed*speed))+pressure/float(1.4f-1.0f);
    }
    // Upstream padding duplicates the last real cell; all normal planes are initialized.
    for(int i=p.cells;i<p.padded;++i) {
        for(int f=0;f<4;++f)in.neighbors[i+f*n]=in.neighbors[p.cells-1+f*n];
        for(int f=0;f<12;++f)in.normals[i+f*n]=in.normals[p.cells-1+f*n];
        for(int f=0;f<5;++f)in.initial[i+f*n]=in.initial[p.cells-1+f*n];
    }
    return in;
}

template<class F> inline void parallel_cells(size_t n,const F& fn) {
    const unsigned available=std::max(1u,std::thread::hardware_concurrency());
    const unsigned count=n<4096?1:std::min(16u,available);
    std::atomic<size_t> next{0};
    auto work=[&]() { for(;;) { const size_t start=next.fetch_add(4096,std::memory_order_relaxed);if(start>=n)break;for(size_t i=start;i<std::min(start+4096,n);++i)fn(i); } };
    std::vector<std::thread> workers;workers.reserve(count-1);
    try { for(unsigned i=1;i<count;++i)workers.emplace_back(work); } catch(const std::exception&) { /* thread creation can fail with system_error or bad_alloc; live workers and caller still finish every cell */ }
    work();for(auto& worker:workers)worker.join();
}
using State=std::array<double,5>;
struct Physical { double pressure,speed,sound;std::array<std::array<double,3>,5> flux; };
inline Physical physical(const State& u) {
    Physical p{};std::array<double,3> v{{u[1]/u[0],u[2]/u[0],u[3]/u[0]}};
    const double squared=v[0]*v[0]+v[1]*v[1]+v[2]*v[2];
    p.pressure=double(1.4f-1.0f)*(u[4]-0.5*u[0]*squared);
    p.speed=std::sqrt(squared);p.sound=std::sqrt(double(1.4f)*p.pressure/u[0]);
    for(int d=0;d<3;++d) {
        p.flux[0][d]=u[1+d];p.flux[4][d]=(u[4]+p.pressure)*v[d];
        for(int m=0;m<3;++m)p.flux[1+m][d]=v[d]*u[1+m]+(m==d?p.pressure:0.0);
    }
    return p;
}
inline State load(const std::vector<double>& u,size_t n,size_t i) {
    State s;for(int k=0;k<5;++k)s[k]=u[i+k*n];return s;
}
struct Reference { std::vector<double> state,old,flux,factor; };
inline Reference reference(const Problem& p,const Input& in) {
    const size_t n=p.padded;Reference r{std::vector<double>(in.initial.begin(),in.initial.end()),std::vector<double>(5*n),std::vector<double>(5*n),std::vector<double>(n)};
    auto f=farfield();State ff;for(int k=0;k<5;++k)ff[k]=f[k];
    Physical far=physical(ff);
    // The upstream far-field flux is initialized using the specified pressure=1,
    // rather than recomputing pressure from the rounded conserved variables.
    for(int d=0;d<3;++d) {
        const double velocity=d==0?double(1.2f):0.0;
        for(int m=0;m<3;++m)far.flux[1+m][d]=double(float(float(velocity)*f[1+m]+(m==d?1.0f:0.0f)));
        far.flux[4][d]=double(float(float(velocity)*float(f[4]+1.0f)));
    }
    for(int iteration=0;iteration<steps;++iteration) {
        r.old=r.state;
        parallel_cells(n,[&](size_t i) { const Physical a=physical(load(r.state,n,i));r.factor[i]=0.5/(std::sqrt(double(in.areas[i]))*(a.speed+a.sound)); });
        for(int stage=0;stage<stages;++stage) {
            parallel_cells(n,[&](size_t i) {
                const State u=load(r.state,n,i);const Physical a=physical(u);State sum{};
                for(int face=0;face<4;++face) {
                    const int neighbor=in.neighbors[i+face*n];std::array<double,3> normal;
                    double length2=0;for(int d=0;d<3;++d){normal[d]=in.normals[i+(face+4*d)*n];length2+=normal[d]*normal[d];}
                    if(neighbor==-1) { for(int d=0;d<3;++d)sum[1+d]+=normal[d]*a.pressure;continue; }
                    State v=neighbor>=0?load(r.state,n,neighbor):ff;Physical b=neighbor>=0?physical(v):far;
                    if(neighbor>=0) {
                        const double viscosity=-std::sqrt(length2)*double(0.2f)*0.5*(a.speed+b.speed+a.sound+b.sound);
                        for(int k=0;k<5;++k)sum[k]+=viscosity*(u[k]-v[k]);
                    }
                    for(int d=0;d<3;++d)for(int k=0;k<5;++k)sum[k]+=0.5*normal[d]*(a.flux[k][d]+b.flux[k][d]);
                }
                for(int k=0;k<5;++k)r.flux[i+k*n]=sum[k];
            });
            parallel_cells(n,[&](size_t i) {const double scale=r.factor[i]/(4-stage);for(int k=0;k<5;++k)r.state[i+k*n]=r.old[i+k*n]+scale*r.flux[i+k*n];});
        }
    }
    return r;
}
inline bool matches(float actual,double expected) {
    return std::isfinite(actual)&&std::isfinite(expected)&&std::fabs(double(actual)-expected)<=5e-6+1e-5*std::fabs(expected);
}
} // namespace cfd
