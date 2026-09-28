#include "../tier2/altis_particlefilter/particlefilter_reference.h"
#include "fixtures/particlefilter_upstream_host.h"
#include <cassert>
#include <cstdio>
#include <limits>
#include <system_error>

static void input_equivalence() {
    for(uint32_t bits:{0u,1u,2u,12345u,0x7fffffffu,0x80000000u,4235699843u,0xffffffffu}) {
        int32_t actual=pf::signed_bits(bits);int frozen=actual;
        for(int i=0;i<10000;++i) {double a=pf::uniform(actual),b=pf_upstream_host::randu(&frozen,0);assert(a==b&&actual==frozen);}
    }
    int32_t zero=pf::signed_bits(4235699843u);assert(std::isfinite(pf::normal(zero)));assert(zero==12345);
    auto actual=pf::input(1031);std::vector<unsigned char> frozen(pf::video_size,0);
    std::vector<int> seeds(1031);for(int i=0;i<1031;++i)seeds[i]=pf::signed_bits(12345u*unsigned(i));
    pf_upstream_host::videoSequence(frozen.data(),128,128,11,seeds.data());
    assert(actual.video==frozen&&actual.seeds==seeds);
    int disk[81]={},offsets[2*pf::neighbors]={};pf_upstream_host::strelDisk(disk,5);
    pf_upstream_host::getneighbors(disk,pf::neighbors,offsets,5);
    assert(std::memcmp(actual.disk.data(),offsets,sizeof(offsets))==0);
    // Pinned roundDouble truncates negative and positive values alike.
    for(double x:{-100.9,-1.6,-0.6,0.0,0.6,1.6,100.9})assert(pf_upstream_host::roundDouble(x)==int(x));
}
static void trajectory(int n) {
    auto in=pf::input(n);auto r=pf::reference(n,in);
    double total=0;
    for(int i=0;i<n;++i) {
        assert(std::isfinite(r.x[i])&&std::isfinite(r.y[i]));
        assert(r.weights[i]>=0);total+=r.weights[i];assert(r.cdf[i]==total);
        // Independent linear search confirms every oracle resampling choice.
        int selected=n-1;for(int j=0;j<n;++j)if(r.cdf[j]>=r.u[i]){selected=j;break;}
        assert(r.xj[i]==r.x[selected]&&r.yj[i]==r.y[selected]);
        int seed=in.seeds[i];for(int k=0;k<40+(i==0?10:0);++k)pf_upstream_host::randu(&seed,0);
        assert(seed==r.seeds[i]);
    }
    assert(std::fabs(total-1.0)<1e-12);
}
int main() {
    input_equivalence();for(int n:{1,2,31,500,511,512,513,1031,4097})trajectory(n);
    assert(!pf::matches(NAN,0)&&!pf::matches(0,INFINITY)&&!pf::matches(1.001,1));
    printf("PASS: 80000 frozen LCG draws, repaired frozen video/seed/disk equality, zero endpoint, nine complete CPU trajectories and every resampling index\n");
}
