#pragma once
// Pinned Rodinia 3.1 physics and independent CPU verification; see README.md.
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace hotspot {
struct Problem { int n=1024, steps=360, pyramid=2; };
inline int integer(const char* text) {
    if(!text || !*text)throw std::invalid_argument("empty integer parameter");
    errno=0;char* end=nullptr;const long value=std::strtol(text,&end,10);
    if(errno || *end || value<1 || value>INT_MAX)
        throw std::invalid_argument("expected positive 32-bit integer parameter");
    return static_cast<int>(value);
}
inline uint64_t product(uint64_t a,uint64_t b) {
    if(b && a>UINT64_MAX/b)throw std::invalid_argument("HotSpot work count exceeds uint64 range");
    return a*b;
}
inline uint64_t sum(uint64_t a,uint64_t b) {
    if(a>UINT64_MAX-b)throw std::invalid_argument("HotSpot work count exceeds uint64 range");
    return a+b;
}
struct Geometry {
    int blocks;
    uint64_t cells,launches,threads_per_launch,launched_threads,logical_updates,executed_updates;
};
inline uint64_t launch_updates(const Problem& p,int blocks,int iteration) {
    // Each local time step updates the Cartesian product of the valid x/y
    // thread intervals. Summing each axis first counts overlapping halos too.
    const int64_t stride=16-2*iteration;
    const int64_t last_coordinate=stride*(blocks-1)-p.pyramid+15;
    const int64_t last_index=last_coordinate*(static_cast<int64_t>(p.n)+1);
    if(last_index>INT_MAX)throw std::invalid_argument("HotSpot padded cell indices exceed int32 range");
    uint64_t updates=0;
    for(int k=0;k<iteration;++k) {
        uint64_t axis=0;
        for(int block=0;block<blocks;++block) {
            const int64_t origin=stride*block-p.pyramid;
            const int64_t low=std::max<int64_t>(0,origin+k+1);
            const int64_t high=std::min<int64_t>(p.n-1,origin+14-k);
            if(high>=low)axis+=static_cast<uint64_t>(high-low+1);
        }
        updates=sum(updates,product(axis,axis));
    }
    return updates;
}
inline Geometry geometry(const Problem& p) {
    if(p.n<1 || p.n>46340 || p.steps<1 || p.pyramid<1 || p.pyramid>7)
        throw std::invalid_argument("HotSpot requires grid_size 1..46340, positive steps and pyramid_height 1..7");
    Geometry g{};const int tile=16-2*p.pyramid;
    g.blocks=(p.n+tile-1)/tile;g.cells=static_cast<uint64_t>(p.n)*p.n;
    const uint64_t full=p.steps/p.pyramid;const int tail=p.steps%p.pyramid;
    g.launches=full+(tail!=0);g.threads_per_launch=product(product(g.blocks,g.blocks),256);
    g.launched_threads=product(g.launches,g.threads_per_launch);
    g.logical_updates=product(g.cells,p.steps);
    if(full)g.executed_updates=product(full,launch_updates(p,g.blocks,p.pyramid));
    if(tail)g.executed_updates=sum(g.executed_updates,launch_updates(p,g.blocks,tail));
    return g;
}
inline Problem parameters(int argc,char** argv) {
    Problem p;const char* names[]={"grid_size","total_iterations","pyramid_height"};
    const char* values[3]={nullptr,nullptr,nullptr};
    for(int i=1;i<argc;++i) {
        int k=0;for(;k<3;++k)if(std::strncmp(argv[i],"--",2)==0 && std::strcmp(argv[i]+2,names[k])==0)break;
        if(k==3 || i+1==argc)throw std::invalid_argument("expected --grid_size, --total_iterations or --pyramid_height and a value");
        const char* value=argv[++i];integer(value);if(!values[k])values[k]=value;
    }
    for(int k=0;k<3;++k) {
        char name[64];std::snprintf(name,sizeof(name),"BENCH_PARAM_%s",names[k]);
        if(const char* value=std::getenv(name))values[k]=value;
    }
    if(values[0])p.n=integer(values[0]);if(values[1])p.steps=integer(values[1]);if(values[2])p.pyramid=integer(values[2]);
    geometry(p);return p;
}
inline double add_up(double a,double b){return std::nextafter(a+b,INFINITY);}
inline double mul_up(double a,double b){return std::nextafter(a*b,INFINITY);}
struct Coefficients { float cap,rx,ry,rz,dt,upstream_dt,sdc,ix,iy,iz; };
inline double row_norm(const Coefficients& c) {
    // Products of two FP32 coefficients are exact in FP64. Cover the few
    // rounded additions/subtractions when bounding the ideal linear operator.
    const double wx=static_cast<double>(c.sdc)*c.ix,wy=static_cast<double>(c.sdc)*c.iy,wa=static_cast<double>(c.sdc)*c.iz;
    const double q=2*wx+2*wy+wa,center=1-q;
    return add_up(add_up(std::fabs(center),add_up(2*wx,2*wy)),mul_up(8*std::numeric_limits<double>::epsilon(),add_up(1,q)));
}
inline Coefficients coefficients(int n) {
    // Preserve pinned compute_tran_temp expressions and double literals.
    const float t_chip=0.0005,chip_height=0.016,chip_width=0.016;
    const float grid_height=chip_height/n,grid_width=chip_width/n;
    Coefficients c;
    c.cap=0.5*1.75e6*t_chip*grid_width*grid_height;
    c.rx=grid_width/(2.0*100*t_chip*grid_height);
    c.ry=grid_height/(2.0*100*t_chip*grid_width);
    c.rz=t_chip/(100*grid_height*grid_width);
    const float max_slope=3.0e6/(0.5*t_chip*1.75e6);
    c.dt=c.upstream_dt=0.001/max_slope;
    auto weights=[&](){c.sdc=c.dt/c.cap;c.ix=1/c.rx;c.iy=1/c.ry;c.iz=1/c.rz;};
    weights();
    if(row_norm(c)>1) {c.dt=static_cast<float>(0.9*c.cap/(2.0/c.rx+2.0/c.ry+1.0/c.rz));weights();}
    if(!std::isfinite(row_norm(c)) || row_norm(c)>1 || !std::isfinite(c.sdc))
        throw std::invalid_argument("HotSpot coefficients are non-finite or amplifying");
    return c;
}
struct Input {std::vector<float> temperature,power;};
inline Input input(const Problem& p) {
    const size_t cells=static_cast<size_t>(p.n)*p.n;
    Input in;in.temperature.resize(cells);in.power.resize(cells);std::srand(42);
    for(size_t i=0;i<cells;++i){in.temperature[i]=80.0f+static_cast<float>(std::rand()%200)/10.0f;in.power[i]=static_cast<float>(std::rand()%100)/500.0f;}
    return in;
}
inline double error_bound(const Problem& p,const Coefficients& c,const Input& in) {
    double magnitude=0,power=0;
    for(float v:in.temperature)magnitude=std::max(magnitude,std::fabs(static_cast<double>(v)));
    for(float v:in.power)power=std::max(power,std::fabs(static_cast<double>(v)));
    const double g=row_norm(c);
    const double h=add_up(1,mul_up(c.sdc,add_up(add_up(4.0*c.ix,4.0*c.iy),c.iz)));
    const double source=mul_up(c.sdc,add_up(power,mul_up(c.iz,80)));
    // Upstream uses FP32 only for the two neighbor pair sums, ambient
    // subtraction/product and final store. The remaining expression is FP64
    // because its center multipliers are the double literal 2.0. Bound these
    // sites separately instead of treating the entire expression as FP32.
    const double uf=std::ldexp(1.0,-24),ud=std::ldexp(1.0,-53);
    const double gamma2=std::nextafter(2*uf/(1-2*uf),INFINITY);
    const double gd=std::nextafter(24*ud/(1-24*ud),INFINITY);
    const double tiny=mul_up(64.0*std::numeric_limits<float>::min(),add_up(1,mul_up(c.sdc,add_up(1,add_up(c.ix,add_up(c.iy,c.iz))))));
    double error=0;
    for(int t=0;t<p.steps;++t) {
        const double gpu_magnitude=add_up(magnitude,error);
        const double next=add_up(mul_up(g,magnitude),source);
        const double cpu_round=mul_up(gd,add_up(mul_up(h,magnitude),source));
        const double pairs=mul_up(2*uf,mul_up(add_up(c.ix,c.iy),gpu_magnitude));
        const double ambient=mul_up(gamma2,mul_up(c.iz,add_up(80,gpu_magnitude)));
        const double inner_round=mul_up(c.sdc,add_up(pairs,ambient));
        const double double_round=mul_up(gd,add_up(add_up(mul_up(h,gpu_magnitude),source),inner_round));
        const double before_store=add_up(add_up(mul_up(g,gpu_magnitude),source),add_up(inner_round,double_round));
        const double gpu_round=add_up(add_up(inner_round,double_round),mul_up(uf,before_store));
        error=add_up(mul_up(g,error),add_up(gpu_round,add_up(cpu_round,tiny)));
        magnitude=add_up(next,add_up(cpu_round,tiny));
        const double range=add_up(magnitude,error);
        if(!std::isfinite(error) || before_store>std::numeric_limits<float>::max()
           || mul_up(2,range)>std::numeric_limits<float>::max()
           || mul_up(add_up(80,range),c.iz)>std::numeric_limits<float>::max())
            throw std::invalid_argument("HotSpot temperature range cannot be verified without overflow");
    }
    return error;
}
inline unsigned reference_workers(int n){return n<512?1u:std::max(1u,std::min(8u,std::thread::hardware_concurrency()));}
// A reference owns this pool. Tasks are non-throwing stencil loops over disjoint
// output rows, with no allocation or shared accumulation. run is a phase
// barrier; all workers stop and join on construction failure or destruction.
class RowPool {
    const unsigned workers;
    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable start,done;
    unsigned generation=0,pending=0;
    bool stopping=false;
    int first=0,last=0;
    std::function<void(int,int)> task;
    void rows(unsigned id) {
        const int count=last-first;
        task(first+count*id/workers,first+count*(id+1)/workers);
    }
    void stop() {
        {std::lock_guard<std::mutex> lock(mutex);stopping=true;}
        start.notify_all();for(auto& t:threads)t.join();
    }
public:
    explicit RowPool(unsigned count):workers(std::max(1u,std::min(8u,count))) {
        try {
            for(unsigned id=1;id<workers;++id) {
                threads.emplace_back([this,id]() {
                    unsigned seen=0;std::unique_lock<std::mutex> lock(mutex);
                    while(true) {
                        start.wait(lock,[&]() {return stopping || seen!=generation;});
                        if(stopping)return;
                        seen=generation;
                        lock.unlock();rows(id);lock.lock();
                        if(--pending==0)done.notify_one();
                    }
                });
            }
        } catch(...) {stop();throw;}
    }
    ~RowPool(){stop();}
    RowPool(const RowPool&)=delete;
    RowPool& operator=(const RowPool&)=delete;
    void run(int begin,int end,std::function<void(int,int)> operation) {
        if(workers==1 || end-begin<16) {operation(begin,end);return;}
        {
            std::lock_guard<std::mutex> lock(mutex);
            task=std::move(operation);first=begin;last=end;pending=workers-1;++generation;
        }
        start.notify_all();rows(0);
        std::unique_lock<std::mutex> lock(mutex);done.wait(lock,[&]() {return pending==0;});
    }
};
inline std::vector<double> reference(const Problem& p,const Coefficients& c,const Input& in,unsigned workers) {
    const int n=p.n;std::vector<double> current(in.temperature.begin(),in.temperature.end()),next(current.size());
    if(current.size()!=static_cast<size_t>(n)*n || in.power.size()!=current.size())throw std::invalid_argument("HotSpot reference input shape mismatch");
    RowPool pool(workers);
    for(int t=0;t<p.steps;++t) {
        pool.run(0,n,[&](int begin,int end) {
            for(int y=begin;y<end;++y)for(int x=0;x<n;++x) {
                const size_t i=static_cast<size_t>(y)*n+x;
                const size_t north=y?i-n:i,south=y+1<n?i+n:i,west=x?i-1:i,east=x+1<n?i+1:i;
                const double center=current[i];
                next[i]=center+static_cast<double>(c.sdc)*(in.power[i]
                    +(current[south]+current[north]-2.0*center)*c.iy
                    +(current[east]+current[west]-2.0*center)*c.ix+(80.0-center)*c.iz);
            }
        });
        current.swap(next);
    }
    return current;
}
inline size_t compare(const std::vector<float>& actual,const std::vector<double>& expected,double bound) {
    if(actual.size()!=expected.size() || !std::isfinite(bound) || bound<0)throw std::invalid_argument("invalid HotSpot verification operands");
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
} // namespace hotspot
