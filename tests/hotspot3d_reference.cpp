#include "../tier2/rodinia_hotspot3d/hip/hotspot3d_reference.h"
#include "upstream/hotspot3d_cpu.h"
#include <cassert>
#include <cstring>

// A separate long-double six-neighbor sum, with explicit axis-coordinate clamping.
static std::vector<long double> oracle(const hotspot3d::Problem& p,const hotspot3d::Coefficients& c,const hotspot3d::Input& in) {
    const int n=p.n;std::vector<long double> a(in.temperature.begin(),in.temperature.end()),b(a.size());
    const int dx[]={-1,1,0,0,0,0},dy[]={0,0,1,-1,0,0},dz[]={0,0,0,0,-1,1};
    const float weight[]={c.cw,c.ce,c.cs,c.cn,c.cb,c.ct};
    for(int t=0;t<p.steps;++t) {
        for(int z=0;z<n;++z)for(int y=0;y<n;++y)for(int x=0;x<n;++x) {
            const size_t i=(static_cast<size_t>(z)*n+y)*n+x;
            long double sum=c.cc*a[i];
            for(int d=0;d<6;++d) {
                const int xx=std::max(0,std::min(n-1,x+dx[d]));
                const int yy=std::max(0,std::min(n-1,y+dy[d]));
                const int zz=std::max(0,std::min(n-1,z+dz[d]));
                sum+=weight[d]*a[(static_cast<size_t>(zz)*n+yy)*n+xx];
            }
            b[i]=sum+static_cast<long double>(c.sdc)*in.power[i]+static_cast<long double>(c.ct)*p.ambient;
        }
        a.swap(b);
    }
    return a;
}
static void check_input(int n) {
    hotspot3d::Problem p;p.n=n;
    const auto in=hotspot3d::input(p);const int next=std::rand();std::srand(42);
    for(size_t i=0;i<in.temperature.size();++i) {
        const float temp=p.ambient+(float)(std::rand()%200)/10.0f;
        const float power=(float)(std::rand()%100)/500.0f;
        assert(in.temperature[i]==temp && in.power[i]==power);
    }
    assert(std::rand()==next);
}
int main(int argc,char** argv) {
    using namespace hotspot3d;
    int limited=0;
    for(int n=1;n<=1290;++n) {
        const auto c=coefficients(n);const auto pin=pinned_hotspot3d::physics(n,n,n);admissible(c);
        assert(c.cap==pin.cap && c.rx==pin.rx && c.ry==pin.ry && c.rz==pin.rz && c.upstream_dt==pin.dt);
        const float sdc=pin.dt/pin.cap,ce=sdc/pin.rx,cn=sdc/pin.ry,ct=sdc/pin.rz;
        const float cc=1.0-(2.0*ce+2.0*cn+3.0*ct);
        const double norm=std::fabs(cc)+2.0*ce+2.0*cn+2.0*ct;
        if(norm<=1)assert(c.dt==pin.dt && c.sdc==sdc && c.ce==ce && c.cn==cn && c.ct==ct && c.cc==cc);
        else {++limited;assert(c.dt<pin.dt && c.cc>0.09f);}
        if(n<=512)assert(c.dt==pin.dt);
    }
    assert(limited>0);printf("PASS: all 1290 admitted dimensions, pinned physics and %d bounded-timestep grids\n",limited);
    for(int n:{1,2,3,8,17})for(int steps:{1,2,3,10,50}) {
        Problem p;p.n=n;p.steps=steps;auto in=input(p);const auto c=coefficients(n);
        const auto expected=reference(p,c,in);const auto exact=oracle(p,c,in);const double bound=error_bound(p,c,in);
        std::vector<float> pin_a=in.temperature,pin_b(pin_a.size());
        const float* pin=pinned_hotspot3d::computeTempCPU(in.power.data(),pin_a.data(),pin_b.data(),n,n,n,c.cap,c.rx,c.ry,c.rz,c.dt,steps);
        std::vector<float> actual(pin,pin+pin_a.size());assert(compare(actual,expected,bound)==0);
        for(size_t i=0;i<exact.size();++i)assert(std::fabs(static_cast<long double>(expected[i])-exact[i])<1e-11L);
    }
    // Exercise limiting without allocating a 1235^3 or 1290^3 hardware problem.
    // Spatial shape and physical coefficients are independent helper inputs.
    for(int physical_n:{1235,1290})for(int n:{1,3,8})for(int steps:{1,2,3,50}) {
        Problem p;p.n=n;p.steps=steps;auto in=input(p);const auto c=coefficients(physical_n);
        assert(c.dt<c.upstream_dt);admissible(c);
        const auto expected=reference(p,c,in);const auto exact=oracle(p,c,in);
        const double bound=error_bound(p,c,in);
        std::vector<float> pin_a=in.temperature,pin_b(pin_a.size());
        const float* pin=pinned_hotspot3d::computeTempCPU(in.power.data(),pin_a.data(),pin_b.data(),n,n,n,c.cap,c.rx,c.ry,c.rz,c.dt,steps);
        assert(compare(std::vector<float>(pin,pin+pin_a.size()),expected,bound)==0);
        for(size_t i=0;i<exact.size();++i)assert(std::fabs(static_cast<long double>(expected[i])-exact[i])<1e-11L);
    }
    puts("PASS: 24 small-grid trajectories with actual limited-timestep coefficients");
    // Nondefault ambient values and an asymmetric input exercise all six faces.
    for(float ambient:{-20.5f,0.0f,80.125f}) {
        Problem p;p.n=3;p.steps=3;p.ambient=ambient;auto in=input(p);const auto c=coefficients(p.n);
        for(size_t i=0;i<in.temperature.size();++i){in.temperature[i]=static_cast<float>(i*i+3*i);in.power[i]=static_cast<float>(i)/100;}
        const auto expected=reference(p,c,in);const auto exact=oracle(p,c,in);
        for(size_t i=0;i<exact.size();++i)assert(std::fabs(static_cast<long double>(expected[i])-exact[i])<1e-11L);
    }
    std::vector<float> actual(27,1.0f);std::vector<double> expected(27,1.0);
    const double bound=1e-4;assert(compare(actual,expected,bound)==0);
    for(size_t i:{size_t(0),size_t(1),size_t(13),size_t(26)}) {
        expected[i]=1+bound/2;assert(compare(actual,expected,bound)==0);
        expected[i]=1+bound*2;assert(compare(actual,expected,bound)==1);expected[i]=1;
        for(float value:{NAN,INFINITY,-INFINITY,2.0f}) {actual[i]=value;assert(compare(actual,expected,bound)==1);actual[i]=1;}
        for(double value:{double(NAN),double(INFINITY),-double(INFINITY)}) {expected[i]=value;assert(compare(actual,expected,bound)==1);expected[i]=1;}
    }
    std::fill(actual.begin(),actual.end(),2.0f);assert(compare(actual,expected,bound)==27);
    for(double bad:{-1.0,double(NAN),double(INFINITY)}){bool caught=false;try{compare(actual,expected,bad);}catch(const std::invalid_argument&){caught=true;}assert(caught);}
    bool caught=false;try{compare({},expected,bound);}catch(const std::invalid_argument&){caught=true;}assert(caught);
    for(int n:{1,2,17,64})check_input(n);
    if(argc==2 && std::strcmp(argv[1],"--all-inputs")==0)
        for(int n:{16,24,32,40,48,56,64,72,80,96,112,128,144,160,192,224,256,288,320,384,448,512})check_input(n);
    puts("PASS: pinned CPU trajectories, independent full-grid oracle, legacy inputs and complete comparator failure cases");
}
