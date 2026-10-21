#include <cstdio>
#include <limits>
#include "../tier2/polybench_gemm/hip/cpu_reference.h"
int main() {
    int cases=0;
    for(int rows:{1,3,4,5,17,33,65})for(int cols:{1,7,31,32,33})for(int inner:{1,2,17,33})
        for(int dataset=0;dataset<2;++dataset)for(int coefficients=0;coefficients<3;++coefficients) {
            std::vector<float>a((size_t)rows*inner),b((size_t)inner*cols),c((size_t)rows*cols);
            for(size_t q=0;q<a.size();++q)a[q]=dataset?0:float(int(q*7%31)-15)/8;
            for(size_t q=0;q<b.size();++q)b[q]=dataset?0:float(int(q*11%29)-14)/8;
            for(size_t q=0;q<c.size();++q)c[q]=float(int(q*13%23)-11)/8;
            const float alpha=coefficients==0?1.5f:coefficients==1?-0.5f:0.0f;
            const float beta=coefficients==0?1.2f:coefficients==1?0.0f:-2.0f;
            const auto aa=a,bb=b,cc=c;std::vector<double> first;
            for(unsigned workers:{1u,2u,8u}) {
                const auto actual=cpu_gemm(a.data(),b.data(),c.data(),rows,cols,inner,alpha,beta,workers);
                if(actual.size()!=c.size() || a!=aa || b!=bb || c!=cc)return 1;
                if(workers==1)first=actual;else if(actual!=first)return 1;
                for(int i=0;i<rows;++i)for(int j=0;j<cols;++j) {
                    long double sum=0;
                    for(int k=0;k<inner;++k)sum+=(long double)a[(size_t)i*inner+k]*b[(size_t)k*cols+j];
                    const size_t q=(size_t)i*cols+j;
                    const long double expected=(long double)alpha*sum+(long double)beta*c[q];
                    if(!std::isfinite(actual[q])||std::fabs((long double)actual[q]-expected)>1e-12L*std::max(1.0L,std::fabs(expected)))return 1;
                }
            }
            ++cases;
        }
    if(!gemm_value_matches(0,0)||!gemm_value_matches(2e-4f,0)||gemm_value_matches(1e-3f,0)||
       !gemm_value_matches(100.02f,100)||gemm_value_matches(100.1f,100)||
       !gemm_value_matches(-100.02f,-100)||gemm_value_matches(-100.1f,-100))return 1;
    for(float bad:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()})
        if(gemm_value_matches(bad,0)||gemm_value_matches(0,bad)||gemm_value_matches(bad,bad))return 1;
    std::printf("PASS: %d rectangular long-double references, identical 1/2/8-worker outputs, original input preservation and finite/tolerance checks\n",cases);
}
