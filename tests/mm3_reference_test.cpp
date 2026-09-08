#include <cstdio>
#include <limits>
#include "../tier2/polybench_3mm/cpu_reference.h"
template<class Left,class Right>
static std::vector<long double> scalar_product(const std::vector<Left>& a,const std::vector<Right>& b,int rows,int cols,int inner) {
    std::vector<long double> out((size_t)rows*cols);
    for(int i=0;i<rows;++i)for(int j=0;j<cols;++j)
        for(int k=0;k<inner;++k)out[(size_t)i*cols+j]+=(long double)a[(size_t)i*inner+k]*b[(size_t)k*cols+j];
    return out;
}
int main() {
    int cases=0;
    const int shapes[][3]={{1,3,2},{3,7,5},{17,33,16},{33,16,17}};
    for(int ni:{1,3,5,17,33,65})for(int nj:{1,2,16,17,33,65})for(const auto& shape:shapes)for(int dataset=0;dataset<2;++dataset) {
        const int nk=shape[0],nl=shape[1],nm=shape[2];
        std::vector<float>a((size_t)ni*nk),b((size_t)nk*nj),c((size_t)nj*nm),d((size_t)nm*nl);
        for(size_t q=0;q<a.size();++q)a[q]=dataset?0:float(int(q*7%31)-15)/8;
        for(size_t q=0;q<b.size();++q)b[q]=float(int(q*11%29)-14)/8;
        for(size_t q=0;q<c.size();++q)c[q]=float(int(q*13%23)-11)/8;
        for(size_t q=0;q<d.size();++q)d[q]=float(int(q*17%19)-9)/8;
        const auto aa=a,bb=b,cc=c,dd=d;
        const auto e=scalar_product(a,b,ni,nj,nk),f=scalar_product(c,d,nj,nl,nm),g=scalar_product(e,f,ni,nl,nj);
        const std::vector<long double>* expected[]={&e,&f,&g};Cpu3mm first;
        for(unsigned workers:{1u,4u,16u}) {
            const auto actual=cpu_3mm(a.data(),b.data(),c.data(),d.data(),ni,nj,nk,nl,nm,workers);
            if(a!=aa || b!=bb || c!=cc || d!=dd)return 1;
            if(workers==1)first=actual;else if(actual.e!=first.e || actual.f!=first.f || actual.g!=first.g)return 1;
            const std::vector<double>* values[]={&actual.e,&actual.f,&actual.g};
            for(int stage=0;stage<3;++stage) {
                if(values[stage]->size()!=expected[stage]->size())return 1;
                for(size_t q=0;q<values[stage]->size();++q) {
                    const double value=(*values[stage])[q];const long double ref=(*expected[stage])[q];
                    if(!std::isfinite(value)||std::fabs((long double)value-ref)>1e-12L*std::max(1.0L,std::fabs(ref)))return 1;
                }
            }
        }
        ++cases;
    }
    for(bool final:{false,true}) {
        const float within=final?1e-3f:2e-4f,outside=final?4e-3f:1e-3f;
        if(!mm3_value_matches(0,0,final)||!mm3_value_matches(within,0,final)||mm3_value_matches(outside,0,final)||
           !mm3_value_matches(100+within*100,100,final)||mm3_value_matches(100+outside*100,100,final)||
           !mm3_value_matches(-100-within*100,-100,final)||mm3_value_matches(-100-outside*100,-100,final))return 1;
        for(float bad:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()})
            if(mm3_value_matches(bad,0,final)||mm3_value_matches(0,bad,final)||mm3_value_matches(bad,bad,final))return 1;
    }
    std::printf("PASS: %d rectangular three-product long-double cases, equal 1/4/16-worker outputs, inputs preserved, stage-specific finite/tolerance checks\n",cases);
}
