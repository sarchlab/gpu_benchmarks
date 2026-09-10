#pragma once
// NAS MG problem, exact 46-bit input stream, reference storage and verification.
// See README.md for the upstream pin, class contract and numerical bounds.
#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace nas_mg {
struct Class { char name; int n,iterations; double l2; };
inline const Class* problem(char name) {
    static const Class classes[]={{'S',32,4,0.5307707005734e-04},
        {'W',128,4,0.6467329375339e-05},{'A',256,4,0.2433365309069e-05},
        {'B',256,20,0.1800564401355e-05},{'C',512,20,0.5706732285740e-06}};
    for(const auto& c:classes) if(c.name==name) return &c;
    return nullptr;
}
inline std::array<double,4> smoother(const Class& c) {
    return c.name=='S'||c.name=='W'||c.name=='A' ? std::array<double,4>{{-3.0/8,1.0/32,-1.0/64,0}}
                                              : std::array<double,4>{{-3.0/17,1.0/33,-1.0/61,0}};
}
struct Layout {
    std::vector<int> width;
    std::vector<size_t> offset;
    size_t cells=0;
    explicit Layout(const Class& c) {
        for(int n=2;n<=c.n;n*=2) {width.push_back(n+2);offset.push_back(cells);cells+=cube(n+2);}
    }
    static size_t cube(int n) {return static_cast<size_t>(n)*n*n;}
    int levels() const {return static_cast<int>(width.size());}
};
// Runtime-sized 3-D view. All production domains are admitted cubic NAS grids.
struct View {
    double* data; int n1,n2;
    struct Plane {double* data;int n1;double* operator[](int y) const {return data+static_cast<size_t>(y)*n1;}};
    Plane operator[](int z) const {return {data+static_cast<size_t>(z)*n1*n2,n1};}
};
inline uint64_t random_step(uint64_t state) {
    // Unsigned wrapping is intentional: only the low 46 product bits are used.
    return (state*UINT64_C(1220703125))&((UINT64_C(1)<<46)-1);
}
inline std::vector<double> input(const Class& c) {
    const int n=c.n,p=n+2;
    std::vector<double> values(Layout::cube(p),0.0);
    struct Candidate {uint64_t value;size_t index;};
    std::array<Candidate,10> large{},small{};
    for(auto& x:small)x={UINT64_C(1)<<46,0};
    uint64_t state=314159265;
    for(int z=1;z<=n;++z) for(int y=1;y<=n;++y) for(int x=1;x<=n;++x) {
        state=random_step(state);
        const size_t i=(static_cast<size_t>(z)*p+y)*p+x;
        // zran3's strict comparison and bubble ordering, streaming the identical
        // x-fastest sequence instead of retaining the temporary random grid.
        if(state>large[0].value) {
            large[0]={state,i};
            for(int j=0;j<9 && large[j].value>large[j+1].value;++j)std::swap(large[j],large[j+1]);
        }
        if(state<small[0].value) {
            small[0]={state,i};
            for(int j=0;j<9 && small[j].value<small[j+1].value;++j)std::swap(small[j],small[j+1]);
        }
    }
    for(int j=9;j>=0;--j)values[small[j].index]=-1.0;
    for(int j=9;j>=0;--j)values[large[j].index]=1.0;
    View v{values.data(),p,p};
    for(int z=1;z<p-1;++z)for(int y=1;y<p-1;++y){v[z][y][0]=v[z][y][p-2];v[z][y][p-1]=v[z][y][1];}
    for(int z=1;z<p-1;++z)for(int x=0;x<p;++x){v[z][0][x]=v[z][p-2][x];v[z][p-1][x]=v[z][1][x];}
    for(int y=0;y<p;++y)for(int x=0;x<p;++x){v[0][y][x]=v[p-2][y][x];v[p-1][y][x]=v[1][y][x];}
    return values;
}
inline unsigned reference_workers(int n) {
    return n<128 ? 1u : std::max(1u,std::min(8u,std::thread::hardware_concurrency()));
}
// A reference owns this pool. Tasks are non-throwing stencil loops over disjoint
// output planes, with no allocation or shared accumulation. run is a phase
// barrier; all workers stop and join on construction failure or destruction.
class PlanePool {
    const unsigned workers;
    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable start,done;
    unsigned generation=0,pending=0;
    bool stopping=false;
    int first=0,last=0;
    std::function<void(int,int)> task;
    void planes(unsigned id) {
        const int count=last-first;
        task(first+count*id/workers,first+count*(id+1)/workers);
    }
    void stop() {
        {std::lock_guard<std::mutex> lock(mutex);stopping=true;}
        start.notify_all();for(auto& t:threads)t.join();
    }
public:
    explicit PlanePool(unsigned count):workers(std::max(1u,std::min(8u,count))) {
        try {
            for(unsigned id=1;id<workers;++id) {
                threads.emplace_back([this,id]() {
                    unsigned seen=0;std::unique_lock<std::mutex> lock(mutex);
                    while(true) {
                        start.wait(lock,[&]() {return stopping || seen!=generation;});
                        if(stopping)return;
                        seen=generation;
                        lock.unlock();planes(id);lock.lock();
                        if(--pending==0)done.notify_one();
                    }
                });
            }
        } catch(...) {stop();throw;}
    }
    ~PlanePool(){stop();}
    PlanePool(const PlanePool&)=delete;
    PlanePool& operator=(const PlanePool&)=delete;
    void run(int begin,int end,std::function<void(int,int)> operation) {
        if(workers==1 || end-begin<16) {operation(begin,end);return;}
        {
            std::lock_guard<std::mutex> lock(mutex);
            task=std::move(operation);first=begin;last=end;pending=workers-1;++generation;
        }
        start.notify_all();planes(0);
        std::unique_lock<std::mutex> lock(mutex);done.wait(lock,[&]() {return pending==0;});
    }
};
struct Result {
    std::vector<double> u,r,v;
    std::array<double,4> norms{}; // initial L2/Linf, final L2/Linf
    explicit Result(const Layout& l):u(l.cells),r(l.cells),v(Layout::cube(l.width.back())){}
};
inline bool close(double a,double b) {
    return std::isfinite(a)&&std::isfinite(b)&&std::fabs(a-b)<=1e-11*(1.0+std::fabs(b));
}
inline size_t compare(const Result& actual,const Result& expected) {
    size_t errors=0;
    const std::vector<double>* a[]={&actual.u,&actual.r,&actual.v};
    const std::vector<double>* b[]={&expected.u,&expected.r,&expected.v};
    const char* names[]={"u","r","v","norms"};
    auto check=[&](double x,double y,int k,size_t i) {
        // The binary input spikes/ghosts must agree exactly.
        const bool ok=k==2 ? std::isfinite(x)&&std::isfinite(y)&&x==y : close(x,y);
        if(!ok) {
            if(errors<10)std::fprintf(stderr,"Mismatch %s[%zu]: GPU=%.17g CPU=%.17g\n",names[k],i,x,y);
            ++errors;
        }
    };
    for(int k=0;k<3;++k) {
        if(a[k]->size()!=b[k]->size())throw std::invalid_argument("MG result dimensions differ");
        for(size_t i=0;i<a[k]->size();++i)check((*a[k])[i],(*b[k])[i],k,i);
    }
    for(size_t i=0;i<4;++i)check(actual.norms[i],expected.norms[i],3,i);
    return errors;
}
inline bool published_l2(double value,const Class& c) {
    return std::isfinite(value)&&std::fabs(value-c.l2)/c.l2<=1e-8;
}
} // namespace nas_mg
