#include "../tier2/parboil_lbm/lbm_reference.h"
#include <cassert>
#include <limits>
int main() {
    const lbm::Layout l(16);auto in=lbm::input(l);
    const auto one=lbm::reference(l,1,1);
    // At a quiescent interior node equilibrium is conserved; the driven
    // plane sends a changed rest population and opposite-directed momentum.
    assert(std::fabs(one.at(8,8,8,lbm::C)-1.0f/3.0f)<5e-7);
    // Summing the FP32 weights makes density slightly different from one.
    // Allow five single-precision rounding units in this analytic check.
    const double u2=1.5*(.005*.005+.002*.002);
    const double rest=(1.0-1.95)/3.0+1.95/3.0*(1.0-u2);
    assert(std::fabs(one.at(8,8,1,lbm::C)-rest)<5e-7);
    auto ref=lbm::reference(l,10,1),parallel=lbm::reference(l,10,4);
    assert(ref.values==parallel.values);
    for(int z=0;z<l.n;++z)for(int y=0;y<l.n;++y)for(int x=0;x<l.n;++x)
        for(int q=0;q<lbm::Q;++q)in[l.index(x,y,z,q)]=ref.at(x,y,z,q);
    auto r=lbm::compare(l,in,ref);assert(!r.errors && r.cells==4096 && r.populations==19*4096);
    for(int q=0;q<lbm::Q;++q)for(float poison:{1.0f,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()}) {
        const auto i=l.index(15,15,15,q);const float old=in[i];in[i]=poison;
        assert(lbm::compare(l,in,ref).errors);in[i]=old;
        const float prior=ref.at(15,15,15,q);ref.at(15,15,15,q)=poison;
        assert(lbm::compare(l,in,ref).errors);ref.at(15,15,15,q)=prior;
    }
    in[l.index(15,15,15,lbm::FLAGS)]=0;assert(lbm::compare(l,in,ref).errors);
    puts("PASS: analytic one-step cavity; serial/parallel complete trajectory; all 19 boundary populations/output/reference/flags rejected when corrupted");
}
