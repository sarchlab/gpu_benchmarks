#include "../tier2/rodinia_hotspot/hip/hotspot_reference.h"
#include "upstream/hotspot_physics.h"
#include <cassert>

// Independent expanded long-double operator with coordinate clamping.
static std::vector<long double> oracle(const hotspot::Problem& p,const hotspot::Coefficients& c,const hotspot::Input& in) {
    const int n=p.n;std::vector<long double> a(in.temperature.begin(),in.temperature.end()),b(a.size());
    const long double wx=static_cast<long double>(c.sdc)*c.ix,wy=static_cast<long double>(c.sdc)*c.iy,wa=static_cast<long double>(c.sdc)*c.iz;
    const long double cc=1-2*wx-2*wy-wa;
    for(int t=0;t<p.steps;++t) {
        for(int y=0;y<n;++y)for(int x=0;x<n;++x) {
            const size_t i=static_cast<size_t>(y)*n+x;
            b[i]=static_cast<long double>(c.sdc)*in.power[i]+wa*80+cc*a[i]
                +wx*(a[static_cast<size_t>(y)*n+std::max(0,x-1)]+a[static_cast<size_t>(y)*n+std::min(n-1,x+1)])
                +wy*(a[static_cast<size_t>(std::max(0,y-1))*n+x]+a[static_cast<size_t>(std::min(n-1,y+1))*n+x]);
        }
        a.swap(b);
    }
    return a;
}
static std::vector<float> mixed_scalar(const hotspot::Problem& p,const hotspot::Coefficients& c,const hotspot::Input& in) {
    const int n=p.n;auto a=in.temperature;std::vector<float> b(a.size());
    for(int t=0;t<p.steps;++t) {
        for(int y=0;y<n;++y)for(int x=0;x<n;++x) {
            const size_t i=static_cast<size_t>(y)*n+x;
            const float ns=a[static_cast<size_t>(std::max(0,y-1))*n+x]+a[static_cast<size_t>(std::min(n-1,y+1))*n+x];
            const float ew=a[static_cast<size_t>(y)*n+std::max(0,x-1)]+a[static_cast<size_t>(y)*n+std::min(n-1,x+1)];
            const float ambient=(80.0f-a[i])*c.iz;
            b[i]=static_cast<float>(a[i]+static_cast<double>(c.sdc)*(in.power[i]+(ns-2.0*a[i])*c.iy+(ew-2.0*a[i])*c.ix+ambient));
        }
        a.swap(b);
    }
    return a;
}
static uint64_t brute_updates(const hotspot::Problem& p) {
    const int blocks=(p.n+15-2*p.pyramid)/(16-2*p.pyramid);uint64_t updates=0;
    for(int done=0;done<p.steps;) {
        const int it=std::min(p.pyramid,p.steps-done),stride=16-2*it;
        for(int by=0;by<blocks;++by)for(int bx=0;bx<blocks;++bx)
            for(int y=0;y<16;++y)for(int x=0;x<16;++x)for(int k=0;k<it;++k) {
                const int xx=stride*bx-p.pyramid+x,yy=stride*by-p.pyramid+y;
                if(xx>=0 && xx<p.n && yy>=0 && yy<p.n && x>=k+1 && x<=14-k && y>=k+1 && y<=14-k)++updates;
            }
        done+=it;
    }
    return updates;
}
static void check_input(int n) {
    hotspot::Problem p;p.n=n;auto in=hotspot::input(p);const int next=std::rand();std::srand(42);
    for(size_t i=0;i<in.temperature.size();++i) {
        const float temp=80.0f+static_cast<float>(std::rand()%200)/10.0f,power=static_cast<float>(std::rand()%100)/500.0f;
        assert(in.temperature[i]==temp && in.power[i]==power);
    }
    assert(std::rand()==next);
}
int main(int argc,char** argv) {
    using namespace hotspot;int limited=0;
    for(int n=1;n<=46340;++n) {
        const auto c=coefficients(n);const auto pin=pinned_hotspot::physics(n,n);
        assert(c.cap==pin.cap && c.rx==pin.rx && c.ry==pin.ry && c.rz==pin.rz && c.upstream_dt==pin.dt);
        Coefficients original=c;original.dt=pin.dt;original.sdc=pin.dt/pin.cap;
        if(row_norm(original)<=1)assert(c.dt==pin.dt);
        else {++limited;assert(c.dt<pin.dt);}
        assert(row_norm(c)<=1 && c.dt>0);
    }
    printf("PASS: 46340 pinned physical-parameter sets, %d limited timesteps\n",limited);
    for(int n:{1,2,15,16,17,31,32,33})for(int pyramid=1;pyramid<=7;++pyramid)for(int steps:{1,3,8,15}) {
        Problem p;p.n=n;p.pyramid=pyramid;p.steps=steps;const auto g=geometry(p);
        assert(g.executed_updates==brute_updates(p));assert(g.executed_updates>=g.logical_updates);
    }
    puts("PASS: 224 independently enumerated padded/partial-pyramid work counts");
    int trajectories=0;
    for(int physical_n:{1,64,1024,1536,8192})for(int n:{1,3,17,32})for(int steps:{1,2,3,10,360}) {
        Problem p;p.n=n;p.steps=steps;auto in=input(p);const auto c=coefficients(physical_n);const auto exact=oracle(p,c,in);
        const auto serial=reference(p,c,in,1);const auto rounded=mixed_scalar(p,c,in);assert(compare(rounded,serial,error_bound(p,c,in))==0);
        for(size_t i=0;i<serial.size();++i)assert(std::fabs(static_cast<long double>(serial[i])-exact[i])<1e-9L);
        assert(reference(p,c,in,2)==serial && reference(p,c,in,8)==serial);++trajectories;
    }
    printf("PASS: %d full long-double/mixed/serial/2/8-worker trajectories\n",trajectories);
    Problem p;p.n=17;p.steps=10;auto in=input(p);const auto c=coefficients(8192);
    std::fill(in.temperature.begin(),in.temperature.end(),80);std::fill(in.power.begin(),in.power.end(),0);
    const auto steady=reference(p,c,in,8);assert(std::all_of(steady.begin(),steady.end(),[](double v){return v==80;}));
    for(size_t i:{size_t(0),size_t(16),size_t(17),size_t(144),size_t(288)}) {
        in.temperature[i]+=10;const auto a=reference(p,c,in,8);const auto exact=oracle(p,c,in);
        for(size_t j=0;j<a.size();++j)assert(std::fabs(static_cast<long double>(a[j])-exact[j])<1e-9L);
        in.temperature[i]-=10;
    }
    std::vector<float> actual(289,1);std::vector<double> expected(289,1);const double bound=1e-4;
    assert(compare(actual,expected,bound)==0);
    for(size_t i:{size_t(0),size_t(16),size_t(17),size_t(144),size_t(288)}) {
        expected[i]+=bound/2;assert(compare(actual,expected,bound)==0);expected[i]=1;
        for(float v:{2.f,NAN,INFINITY,-INFINITY}){actual[i]=v;assert(compare(actual,expected,bound)==1);actual[i]=1;}
        for(double v:{1+2*bound,double(NAN),double(INFINITY),-double(INFINITY)}){expected[i]=v;assert(compare(actual,expected,bound)==1);expected[i]=1;}
    }
    std::fill(actual.begin(),actual.end(),2);assert(compare(actual,expected,bound)==actual.size());
    for(double bad:{-1.,double(NAN),double(INFINITY)}){bool caught=false;try{compare(actual,expected,bad);}catch(const std::invalid_argument&){caught=true;}assert(caught);}
    bool caught=false;try{compare({},expected,bound);}catch(const std::invalid_argument&){caught=true;}assert(caught);
    for(int n:{1,17,64})check_input(n);
    if(argc==2 && std::strcmp(argv[1],"--all-inputs")==0)for(int n:{64,128,192,256,384,512,640,768,1024,1280,1536,2048,2560,3072,3584,4096,5120,6144,7168,8192})check_input(n);
    puts("PASS: steady/impulse, legacy input and complete finite-comparator cases");
}
