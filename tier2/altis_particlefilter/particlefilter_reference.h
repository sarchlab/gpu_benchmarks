// Independent complete CPU trajectory; no GPU outputs enter this reference.
#pragma once
#include "particlefilter_input.h"
#include <algorithm>
#include <atomic>
#include <thread>

namespace pf {
template<class F> inline void parallel(size_t count,const F& fn) {
    const unsigned workers=count<8?1:std::min(16u,std::max(1u,std::thread::hardware_concurrency()));
    std::atomic<size_t> next{0};
    auto work=[&]() {for(;;){size_t i=next.fetch_add(1,std::memory_order_relaxed);if(i>=count)break;fn(i);}};
    std::vector<std::thread> threads;threads.reserve(workers-1);
    try {for(unsigned i=1;i<workers;++i)threads.emplace_back(work);}
    catch(const std::exception&) { /* caller and started workers finish all work */ }
    work();for(auto& t:threads)t.join();
}
struct Reference {
    std::vector<double> x,y,xj,yj,likelihood,weights,cdf,u,partial;
    std::vector<int32_t> seeds;
    double u0;
    explicit Reference(int n,const Input& in):x(n),y(n),xj(n,width/2),yj(n,height/2),
        likelihood(n),weights(n),cdf(n),u(n),partial((n+block-1)/block),seeds(in.seeds),u0(0) {}
};
inline Reference reference(int n,const Input& in) {
    Reference r(n,in);
    for(int frame=1;frame<frames;++frame) {
        parallel(r.partial.size(),[&](size_t b) {
            double sums[block]={};
            for(int lane=0;lane<block;++lane) {
                const size_t i=b*block+lane;if(i>=size_t(n))break;
                // Model the FP64 fused multiply-add used by the CUDA build.
                r.x[i]=std::fma(5.0,normal(r.seeds[i]),r.xj[i]+1.0);
                r.y[i]=std::fma(2.0,normal(r.seeds[i]),r.yj[i]-2.0);
                double score=0;
                for(auto offset:in.disk) {
                    const int pixel=in.video[gather_index(r.x[i],r.y[i],offset,frame)];
                    const double a=pixel-100,bg=pixel-228;
                    score+=(a*a-bg*bg)/50.0;
                }
                r.likelihood[i]=score/neighbors;
                r.weights[i]=(1.0/n)*std::exp(r.likelihood[i]);sums[lane]=r.weights[i];
            }
            // Match the specified block tree, then the ordered block total.
            for(int stride=block/2;stride;stride/=2)
                for(int lane=0;lane<stride;++lane)sums[lane]+=sums[lane+stride];
            r.partial[b]=sums[0];
        });
        double total=0;for(double sum:r.partial)total+=sum;r.partial[0]=total;
        if(!std::isfinite(total)||total<=0)throw std::runtime_error("Particle Filter CPU weights are not finite and positive");
        double cumulative=0;
        for(int i=0;i<n;++i) {r.weights[i]/=total;cumulative+=r.weights[i];r.cdf[i]=cumulative;}
        r.u0=(1.0/n)*uniform(r.seeds[0]);
        parallel(r.partial.size(),[&](size_t b) {
            for(size_t i=b*block;i<std::min((b+1)*block,size_t(n));++i) {
                r.u[i]=r.u0+double(i)/n;
                // Only the untimed oracle uses binary search; GPU retains the
                // pinned linear search, including its last-particle fallback.
                auto found=std::lower_bound(r.cdf.begin(),r.cdf.end(),r.u[i]);
                const size_t index=found==r.cdf.end()?n-1:found-r.cdf.begin();
                r.xj[i]=r.x[index];r.yj[i]=r.y[index];
            }
        });
    }
    return r;
}
inline bool matches(double actual,double expected) {
    return std::isfinite(actual)&&std::isfinite(expected)&&
        std::fabs(actual-expected)<=1e-11*std::max(1.0,std::fabs(expected));
}
} // namespace pf
