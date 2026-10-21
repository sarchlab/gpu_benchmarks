#include <cstdio>
#include <limits>
#include "../tier2/polybench_fdtd2d/hip/cpu_reference.h"
int main() {
    int cases=0;
    for(int rows:{1,2,3,17,33})for(int cols:{1,2,5,16,33})for(int steps:{1,2,10,50}) for(int dataset=0;dataset<3;++dataset) {
        size_t n=(size_t)rows*cols;std::vector<float>a(n),b(n),c(n);
        for(size_t i=0;i<n;++i){a[i]=(int(i*7%31)-15)/8.0f;b[i]=(int(i*11%29)-14)/8.0f;c[i]=(int(i*13%23)-11)/8.0f;}
        if(dataset) {std::fill(a.begin(),a.end(),0);std::fill(b.begin(),b.end(),0);std::fill(c.begin(),c.end(),0);if(dataset==2)c[n/2]=1;}
        const auto aa=a,bb=b,cc=c;std::vector<long double>x(a.begin(),a.end()),y(b.begin(),b.end()),z(c.begin(),c.end());
        for(int t=0;t<steps;++t) {
            for(int i=0;i<rows;++i)for(int j=1;j<cols;++j){size_t q=(size_t)i*cols+j;x[q]-=0.5L*(z[q]-z[q-1]);}
            for(int j=0;j<cols;++j)y[j]=t;
            for(int i=1;i<rows;++i)for(int j=0;j<cols;++j){size_t q=(size_t)i*cols+j;y[q]-=0.5L*(z[q]-z[q-cols]);}
            for(int i=0;i<rows-1;++i)for(int j=0;j<cols-1;++j){size_t q=(size_t)i*cols+j;z[q]-=(long double)0.7f*(x[q+1]-x[q]+y[q+cols]-y[q]);}
        }
        CpuFdtd first;
        for(unsigned workers:{1u,4u,16u}) {
            auto r=cpu_fdtd(a.data(),b.data(),c.data(),rows,cols,steps,workers);
            if(a!=aa||b!=bb||c!=cc)return 1;
            if(workers==1)first=r;else if(r.ex!=first.ex||r.ey!=first.ey||r.hz!=first.hz)return 1;
            const std::vector<double>* actual[]={&r.ex,&r.ey,&r.hz};const std::vector<long double>* expected[]={&x,&y,&z};
            for(int s=0;s<3;++s)for(size_t i=0;i<n;++i)if(!std::isfinite((*actual[s])[i])||std::fabs((*actual[s])[i]-(*expected[s])[i])>1e-12L*std::max(1.0L,std::fabs((*expected[s])[i])))return 1;
        }
        ++cases;
    }
    if (!fdtd_value_matches(0,0,false) || !fdtd_value_matches(5e-4f,0,false)
        || fdtd_value_matches(2e-3f,0,false) || !fdtd_value_matches(100.05f,100,false)
        || fdtd_value_matches(100.2f,100,false) || !fdtd_value_matches(-100.05f,-100,false)
        || fdtd_value_matches(-100.2f,-100,false)) return 1;
    for (float value : {0.0f,1.0f,-1.0f,100.0f}) {
        if (!fdtd_value_matches(value,value,true)
            || fdtd_value_matches(std::nextafter(value,std::numeric_limits<float>::infinity()),value,true)) return 1;
    }
    for (bool exact : {false,true}) for (float bad : {std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()})
        if (fdtd_value_matches(bad,0,exact) || fdtd_value_matches(0,bad,exact)
            || fdtd_value_matches(bad,bad,exact)) return 1;
    printf("PASS: %d full rectangular FDTD long-double cases, equal 1/4/16-worker results, inputs preserved, finite/tolerance/exact-boundary checks\n",cases);
}
