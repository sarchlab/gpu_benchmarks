#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
struct CpuCorrelation {
    std::vector<double> mean, stddev, normalized, correlation;
};
// Complete independent double reference from the original input. Accumulate
// column statistics in increasing sample order, with contiguous feature access.
// The standard-deviation guard uses the GPU's exact float literal; the final
// correlation diagonal is forced to one, including for constant columns.
static CpuCorrelation cpu_correlation(const float* input,int m,int n) {
    CpuCorrelation ref;
    ref.mean.assign(n,0);ref.stddev.assign(n,0);
    ref.normalized.resize((size_t)m*n);ref.correlation.resize((size_t)n*n);
    for(int i=0;i<m;++i)
        for(int j=0;j<n;++j)ref.mean[j]+=input[(size_t)i*n+j];
    for(int j=0;j<n;++j)ref.mean[j]/=m;
    for(int i=0;i<m;++i)
        for(int j=0;j<n;++j) {
            const double d=(double)input[(size_t)i*n+j]-ref.mean[j];
            ref.stddev[j]+=d*d;
        }
    for(int j=0;j<n;++j) {
        const double s=std::sqrt(ref.stddev[j]/m);
        ref.stddev[j]=s<(double)1e-12f?1.0:s;
    }
    const double sqrt_m=std::sqrt((double)m);
    for(int i=0;i<m;++i)
        for(int j=0;j<n;++j)
            ref.normalized[(size_t)i*n+j]=((double)input[(size_t)i*n+j]-ref.mean[j])/(sqrt_m*ref.stddev[j]);
    // Full transposed product, blocked only for CPU locality; no sampling.
    for(int i=0;i<n;i+=4)
        for(int j=0;j<n;j+=32) {
            const int rows=std::min(4,n-i),cols=std::min(32,n-j);
            double acc[4][32]={};
            for(int k=0;k<m;++k)
                for(int r=0;r<rows;++r) {
                    const double a=ref.normalized[(size_t)k*n+i+r];
                    for(int c=0;c<cols;++c)
                        acc[r][c]+=a*ref.normalized[(size_t)k*n+j+c];
                }
            for(int r=0;r<rows;++r)
                for(int c=0;c<cols;++c)
                    ref.correlation[(size_t)(i+r)*n+j+c]=(i+r==j+c)?1.0:acc[r][c];
        }
    return ref;
}
// Empirically qualified for the declared inputs. Other-device and pathological
// near-zero-variance input qualification remains separate.
static bool correlation_value_matches(float actual,double expected) {
    const double error=std::fabs((double)actual-expected);
    return std::isfinite(actual)&&std::isfinite(expected)&&std::isfinite(error)
        &&error<=1e-5*std::max(1.0,std::fabs(expected));
}
