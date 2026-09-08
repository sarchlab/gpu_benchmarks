#include <cstdio>
#include <limits>
#include "../tier2/polybench_correlation/cpu_reference.h"
static bool close_reference(double actual,long double expected) {
    return std::isfinite(actual)&&std::isfinite(expected)&&
        std::fabs((long double)actual-expected)<=1e-12L*std::max(1.0L,std::fabs(expected));
}
int main() {
    int cases=0;
    for(int m:{1,2,3,15,16,17,31,32,33,65})
        for(int n:{1,2,3,15,16,17,31,32,33,65})
            for(int dataset=0;dataset<4;++dataset) {
                std::vector<float> data((size_t)m*n);
                for(int i=0;i<m;++i)for(int j=0;j<n;++j) {
                    const size_t q=(size_t)i*n+j;
                    data[q]=dataset==0?float(int(q*7%31)-15)/8:
                        dataset==1?0.0f:dataset==2?2.0f:
                        j%3==0?1.0f:j%3==1?float(i)/8:-float(i)/8;
                }
                const auto original=data;
                std::vector<long double> means(n),stddev(n),normalized((size_t)m*n);
                // Scalar feature-wise oracle, independent of row-wise traversal
                // and blocked product used by the implementation.
                for(int j=0;j<n;++j) {
                    long double sum=0;
                    for(int i=0;i<m;++i)sum+=data[(size_t)i*n+j];
                    means[j]=sum/m;long double variance=0;
                    for(int i=0;i<m;++i) {
                        const long double delta=data[(size_t)i*n+j]-means[j];
                        variance+=delta*delta;
                    }
                    const long double s=std::sqrt(variance/m);
                    stddev[j]=s<(long double)1e-12f?1.0L:s;
                    for(int i=0;i<m;++i)normalized[(size_t)i*n+j]=
                        (data[(size_t)i*n+j]-means[j])/(std::sqrt((long double)m)*stddev[j]);
                }
                const auto actual=cpu_correlation(data.data(),m,n);
                if(data!=original || actual.mean.size()!=(size_t)n || actual.stddev.size()!=(size_t)n ||
                   actual.normalized.size()!=(size_t)m*n || actual.correlation.size()!=(size_t)n*n)return 1;
                for(int j=0;j<n;++j) {
                    if(!close_reference(actual.mean[j],means[j]) || !close_reference(actual.stddev[j],stddev[j]))return 1;
                    for(int i=0;i<m;++i)if(!close_reference(actual.normalized[(size_t)i*n+j],normalized[(size_t)i*n+j]))return 1;
                }
                for(int i=0;i<n;++i)for(int j=0;j<n;++j) {
                    long double expected=0;
                    for(int k=0;k<m;++k)expected+=normalized[(size_t)k*n+i]*normalized[(size_t)k*n+j];
                    if(i==j)expected=1;
                    if(!close_reference(actual.correlation[(size_t)i*n+j],expected))return 1;
                    if(i==j && actual.correlation[(size_t)i*n+j]!=1.0)return 1;
                }
                ++cases;
            }
    // Below/equal/above the exact float threshold. A double 1e-12 literal
    // would incorrectly clamp the middle column, because 1e-12f is smaller.
    const float threshold=1e-12f;
    const float lo=std::nextafter(threshold,0.0f);
    const float hi=std::nextafter(threshold,std::numeric_limits<float>::infinity());
    const float tiny[]={-lo,-threshold,-hi,lo,threshold,hi};
    const auto edge=cpu_correlation(tiny,2,3);
    if(edge.stddev[0]!=1 || edge.stddev[1]!=(double)threshold || edge.stddev[2]!=(double)hi)return 1;
    for(double mean:edge.mean)if(mean!=0)return 1;
    if(!close_reference(edge.correlation[1],lo) || !close_reference(edge.correlation[5],1))return 1;
    if(!correlation_value_matches(0,0)||!correlation_value_matches(4e-6f,0)||correlation_value_matches(2e-5f,0)||
       !correlation_value_matches(100.0004f,100)||correlation_value_matches(100.002f,100)||
       !correlation_value_matches(-100.0004f,-100)||correlation_value_matches(-100.002f,-100))return 1;
    for(float bad:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()})
        if(correlation_value_matches(bad,0)||correlation_value_matches(0,bad)||correlation_value_matches(bad,bad))return 1;
    std::printf("PASS: %d full four-stage long-double references; rectangular, zero/constant/mixed columns, exact diagonal, input preservation and finite/tolerance checks\n",cases);
    return 0;
}
