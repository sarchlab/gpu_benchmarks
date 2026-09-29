// (C) Copyright 2010 The Board of Trustees of the University of Illinois.
// Adapted from Parboil a4c63da28f45699e9a274cbc49af3e067162aa2b.
// See UPSTREAM_LICENSE and README.md for provenance and adaptations.
#pragma once
#include "lbm_common.h"
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
namespace lbm {
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
    explicit PlanePool(unsigned count):workers(std::max(1u,std::min(16u,count))) {
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
inline unsigned reference_workers(int n) {
    return n<64 ? 1u : std::max(1u,std::min(16u,std::thread::hardware_concurrency()));
}
inline void cpu_step(const CPUGrid& srcGrid,CPUGrid& dstGrid,int begin,int end) {
    constexpr double DFL1=1.0/3.0,DFL2=1.0/18.0,DFL3=1.0/36.0;
    const int n=srcGrid.layout.n;
    for(int z=begin;z<end;++z)for(int y=0;y<n;++y)for(int x=0;x<n;++x) {
        const unsigned cell_flags=flags(x,y,z,n);
 float ux,uy,uz,u2,rho;

		if( (cell_flags & OBSTACLE)) {
			dstGrid.at(x+0,y+0,z+0,C) = srcGrid.at(x,y,z,C);
			dstGrid.at(x+0,y-1,z+0,S) = srcGrid.at(x,y,z,N);
			dstGrid.at(x+0,y+1,z+0,N) = srcGrid.at(x,y,z,S);
			dstGrid.at(x-1,y+0,z+0,W) = srcGrid.at(x,y,z,E);
			dstGrid.at(x+1,y+0,z+0,E) = srcGrid.at(x,y,z,W);
			dstGrid.at(x+0,y+0,z-1,B) = srcGrid.at(x,y,z,T);
			dstGrid.at(x+0,y+0,z+1,T) = srcGrid.at(x,y,z,B);
			dstGrid.at(x-1,y-1,z+0,SW) = srcGrid.at(x,y,z,NE);
			dstGrid.at(x+1,y-1,z+0,SE) = srcGrid.at(x,y,z,NW);
			dstGrid.at(x-1,y+1,z+0,NW) = srcGrid.at(x,y,z,SE);
			dstGrid.at(x+1,y+1,z+0,NE) = srcGrid.at(x,y,z,SW);
			dstGrid.at(x+0,y-1,z-1,SB) = srcGrid.at(x,y,z,NT);
			dstGrid.at(x+0,y-1,z+1,ST) = srcGrid.at(x,y,z,NB);
			dstGrid.at(x+0,y+1,z-1,NB) = srcGrid.at(x,y,z,ST);
			dstGrid.at(x+0,y+1,z+1,NT) = srcGrid.at(x,y,z,SB);
			dstGrid.at(x-1,y+0,z-1,WB) = srcGrid.at(x,y,z,ET);
			dstGrid.at(x-1,y+0,z+1,WT) = srcGrid.at(x,y,z,EB);
			dstGrid.at(x+1,y+0,z-1,EB) = srcGrid.at(x,y,z,WT);
			dstGrid.at(x+1,y+0,z+1,ET) = srcGrid.at(x,y,z,WB);
			continue;
		}

		rho = + srcGrid.at(x,y,z,C) + srcGrid.at(x,y,z,N)
		      + srcGrid.at(x,y,z,S) + srcGrid.at(x,y,z,E)
		      + srcGrid.at(x,y,z,W) + srcGrid.at(x,y,z,T)
		      + srcGrid.at(x,y,z,B) + srcGrid.at(x,y,z,NE)
		      + srcGrid.at(x,y,z,NW) + srcGrid.at(x,y,z,SE)
		      + srcGrid.at(x,y,z,SW) + srcGrid.at(x,y,z,NT)
		      + srcGrid.at(x,y,z,NB) + srcGrid.at(x,y,z,ST)
		      + srcGrid.at(x,y,z,SB) + srcGrid.at(x,y,z,ET)
		      + srcGrid.at(x,y,z,EB) + srcGrid.at(x,y,z,WT)
		      + srcGrid.at(x,y,z,WB);

		ux = + srcGrid.at(x,y,z,E) - srcGrid.at(x,y,z,W)
		     + srcGrid.at(x,y,z,NE) - srcGrid.at(x,y,z,NW)
		     + srcGrid.at(x,y,z,SE) - srcGrid.at(x,y,z,SW)
		     + srcGrid.at(x,y,z,ET) + srcGrid.at(x,y,z,EB)
		     - srcGrid.at(x,y,z,WT) - srcGrid.at(x,y,z,WB);
		uy = + srcGrid.at(x,y,z,N) - srcGrid.at(x,y,z,S)
		     + srcGrid.at(x,y,z,NE) + srcGrid.at(x,y,z,NW)
		     - srcGrid.at(x,y,z,SE) - srcGrid.at(x,y,z,SW)
		     + srcGrid.at(x,y,z,NT) + srcGrid.at(x,y,z,NB)
		     - srcGrid.at(x,y,z,ST) - srcGrid.at(x,y,z,SB);
		uz = + srcGrid.at(x,y,z,T) - srcGrid.at(x,y,z,B)
		     + srcGrid.at(x,y,z,NT) - srcGrid.at(x,y,z,NB)
		     + srcGrid.at(x,y,z,ST) - srcGrid.at(x,y,z,SB)
		     + srcGrid.at(x,y,z,ET) - srcGrid.at(x,y,z,EB)
		     + srcGrid.at(x,y,z,WT) - srcGrid.at(x,y,z,WB);

		ux /= rho;
		uy /= rho;
		uz /= rho;

		if( (cell_flags & ACCEL)) {
			ux = 0.005f;
			uy = 0.002f;
			uz = 0.000f;
		}

		u2 = 1.5f * (ux*ux + uy*uy + uz*uz);
		dstGrid.at(x+0,y+0,z+0,C) = (1.0f-omega)*srcGrid.at(x,y,z,C) + DFL1*omega*rho*(1.0f                                 - u2);

		dstGrid.at(x+0,y+1,z+0,N) = (1.0f-omega)*srcGrid.at(x,y,z,N) + DFL2*omega*rho*(1.0f +       uy*(4.5f*uy       + 3.0f) - u2);
		dstGrid.at(x+0,y-1,z+0,S) = (1.0f-omega)*srcGrid.at(x,y,z,S) + DFL2*omega*rho*(1.0f +       uy*(4.5f*uy       - 3.0f) - u2);
		dstGrid.at(x+1,y+0,z+0,E) = (1.0f-omega)*srcGrid.at(x,y,z,E) + DFL2*omega*rho*(1.0f +       ux*(4.5f*ux       + 3.0f) - u2);
		dstGrid.at(x-1,y+0,z+0,W) = (1.0f-omega)*srcGrid.at(x,y,z,W) + DFL2*omega*rho*(1.0f +       ux*(4.5f*ux       - 3.0f) - u2);
		dstGrid.at(x+0,y+0,z+1,T) = (1.0f-omega)*srcGrid.at(x,y,z,T) + DFL2*omega*rho*(1.0f +       uz*(4.5f*uz       + 3.0f) - u2);
		dstGrid.at(x+0,y+0,z-1,B) = (1.0f-omega)*srcGrid.at(x,y,z,B) + DFL2*omega*rho*(1.0f +       uz*(4.5f*uz       - 3.0f) - u2);

		dstGrid.at(x+1,y+1,z+0,NE) = (1.0f-omega)*srcGrid.at(x,y,z,NE) + DFL3*omega*rho*(1.0f + (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
		dstGrid.at(x-1,y+1,z+0,NW) = (1.0f-omega)*srcGrid.at(x,y,z,NW) + DFL3*omega*rho*(1.0f + (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
		dstGrid.at(x+1,y-1,z+0,SE) = (1.0f-omega)*srcGrid.at(x,y,z,SE) + DFL3*omega*rho*(1.0f + (+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
		dstGrid.at(x-1,y-1,z+0,SW) = (1.0f-omega)*srcGrid.at(x,y,z,SW) + DFL3*omega*rho*(1.0f + (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
		dstGrid.at(x+0,y+1,z+1,NT) = (1.0f-omega)*srcGrid.at(x,y,z,NT) + DFL3*omega*rho*(1.0f + (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
		dstGrid.at(x+0,y+1,z-1,NB) = (1.0f-omega)*srcGrid.at(x,y,z,NB) + DFL3*omega*rho*(1.0f + (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
		dstGrid.at(x+0,y-1,z+1,ST) = (1.0f-omega)*srcGrid.at(x,y,z,ST) + DFL3*omega*rho*(1.0f + (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
		dstGrid.at(x+0,y-1,z-1,SB) = (1.0f-omega)*srcGrid.at(x,y,z,SB) + DFL3*omega*rho*(1.0f + (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
		dstGrid.at(x+1,y+0,z+1,ET) = (1.0f-omega)*srcGrid.at(x,y,z,ET) + DFL3*omega*rho*(1.0f + (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
		dstGrid.at(x+1,y+0,z-1,EB) = (1.0f-omega)*srcGrid.at(x,y,z,EB) + DFL3*omega*rho*(1.0f + (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
		dstGrid.at(x-1,y+0,z+1,WT) = (1.0f-omega)*srcGrid.at(x,y,z,WT) + DFL3*omega*rho*(1.0f + (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
		dstGrid.at(x-1,y+0,z-1,WB) = (1.0f-omega)*srcGrid.at(x,y,z,WB) + DFL3*omega*rho*(1.0f + (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);

    }
}
inline CPUGrid reference(const Layout& l,int steps,unsigned workers) {
    CPUGrid a(l),b(l);PlanePool pool(workers);
    for(int t=0;t<steps;++t){pool.run(0,l.n,[&](int first,int last){cpu_step(a,b,first,last);});a.values.swap(b.values);}
    return a;
}
} // namespace lbm
