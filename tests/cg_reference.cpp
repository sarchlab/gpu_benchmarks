#include "../tier2/npb_cg/hip/nas_cg.h"
#include "fixtures/npb_cg_original.h"
#include <cassert>
#include <cstring>
#include <limits>
#include <string>

static void require(bool ok) { if(!ok) { std::fprintf(stderr,"CG reference check failed\n"); std::abort(); } }
static void identical(const nas_cg::Result& a,const nas_cg::Result& b) {
    const std::vector<double>* x[]={&a.x,&a.z,&a.p,&a.q,&a.r,&a.zeta,&a.residual};
    const std::vector<double>* y[]={&b.x,&b.z,&b.p,&b.q,&b.r,&b.zeta,&b.residual};
    for(int k=0;k<7;++k) require(x[k]->size()==y[k]->size() &&
        std::memcmp(x[k]->data(),y[k]->data(),x[k]->size()*sizeof(double))==0);
}
template<int Nonzer> static void check(const nas_cg::Class& c,bool matrix_only) {
    auto actual=nas_cg::make_matrix(c);
    pinned_cg::Original<Nonzer> original(c.n,c.shift);
    auto expected=original.matrix();
    require(actual.offsets==expected.offsets && actual.columns==expected.columns);
    require(actual.values.size()==expected.values.size() &&
        std::memcmp(actual.values.data(),expected.values.data(),actual.values.size()*sizeof(double))==0);
    require(actual.offsets.front()==0 && actual.offsets.back()==static_cast<int>(actual.values.size()));
    for(int i=0;i<c.n;++i)for(int k=actual.offsets[i];k<actual.offsets[i+1];++k) {
        require(actual.columns[k]>=0 && actual.columns[k]<c.n && std::isfinite(actual.values[k]));
        if(k>actual.offsets[i])require(actual.columns[k-1]<actual.columns[k]);
    }
    std::printf("Class %c: %zu complete matrix values and all CSR indices match pinned makea exactly\n",c.name,actual.values.size());
    if(matrix_only)return;
    const auto baseline=original.solve(expected,c);
    const auto serial=nas_cg::reference(actual,c,1);
    identical(serial,baseline);
    for(unsigned workers : {2u,8u})identical(nas_cg::reference(actual,c,workers),serial);
    require(nas_cg::published_zeta(serial.zeta.back(),c));
    auto corrupt=serial;
    std::vector<double>* arrays[]={&corrupt.x,&corrupt.z,&corrupt.p,&corrupt.q,&corrupt.r,&corrupt.zeta,&corrupt.residual};
    size_t checks=0;
    for(auto* array:arrays)for(size_t i : {size_t(0),array->size()/2,array->size()-1})
        for(double value : {(*array)[i]+.01,std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()}) {
            const double saved=(*array)[i];(*array)[i]=value;
            require(nas_cg::compare(corrupt,serial)==1);(*array)[i]=saved;++checks;
        }
    for(size_t i=0;i<12;++i)corrupt.x[i]+=.01;
    require(nas_cg::compare(corrupt,serial)==12);++checks;
    auto bad_reference=serial;bad_reference.x.back()=std::numeric_limits<double>::quiet_NaN();
    require(nas_cg::compare(serial,bad_reference)==1);++checks;
    require(!nas_cg::published_zeta(std::numeric_limits<double>::infinity(),c));
    require(!nas_cg::published_zeta(c.zeta*(1.0+1e-8),c));
    std::printf("Class %c: pinned/serial/2-worker/8-worker complete trajectories identical; %zu finite/nonfinite/full-count checks pass, zeta=%.15g\n",c.name,checks,serial.zeta.back());
}
int main(int argc,char**argv) {
    setvbuf(stdout,nullptr,_IONBF,0);
    const bool all=argc>1 && std::string(argv[1])=="--all-matrices";
    // Independent floating randlc from the pin versus production integer LCG.
    pinned_cg::Original<7> random_source(1400,10.0);
    for(std::uint64_t seed : {1ULL,314159265ULL,(1ULL<<46)-1}) {
        double floating=static_cast<double>(seed);
        for(int i=0;i<10000;++i)require(nas_cg::random(seed)==random_source.randlc(&floating,1220703125.0));
    }
    check<7>(*nas_cg::problem(1400),false);
    if(all) { check<8>(*nas_cg::problem(7000),true);check<11>(*nas_cg::problem(14000),true);
              check<13>(*nas_cg::problem(75000),true);check<15>(*nas_cg::problem(150000),true); }
    else {
        // Tiny non-class fixtures exercise all NONZER specializations without
        // accepting these problems through the production command-line path.
        check<8>({32,8,3,12.0,0.0,'U'},true);check<11>({64,11,3,20.0,0.0,'U'},true);
        check<13>({64,13,3,60.0,0.0,'U'},true);check<15>({64,15,3,110.0,0.0,'U'},true);
    }
    std::puts("30000 RNG states and complete pinned matrix/reference checks pass");
}
