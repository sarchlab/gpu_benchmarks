#include "../tier2/polybench_atax/cpu_reference.h"
#include "../tier2/polybench_bicg/cpu_reference.h"
#include "../tier2/polybench_mvt/cpu_reference.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static bool output_equal(const std::vector<float>& actual,const std::vector<float>& expected,
                         int count,const char* name) {
    if (actual.front()!=12345.0f || actual.back()!=12345.0f) {
        fprintf(stderr,"FAIL: %s wrote outside its full output\n",name);return false;
    }
    for (int i=0;i<count;++i) {
        if (!std::isfinite(actual[i+1]) || std::memcmp(&actual[i+1],&expected[i],sizeof(float))) {
            fprintf(stderr,"FAIL: %s[%d] %.9g expected %.9g\n",name,i,actual[i+1],expected[i]);return false;
        }
    }
    return true;
}

static bool check_case(int m,int n,int pattern) {
    std::vector<float> A((size_t)m*n),p(n),r(m);
    for (int i=0;i<m;++i) for (int j=0;j<n;++j)
        A[(size_t)i*n+j]=pattern==0?float((i*17+j*31)%47-23)/10.0f:
            pattern==1?0.0f:pattern==2?(i==j?1.0f:0.0f):float((i*73+j*19)%101-50)/8.0f;
    for (int j=0;j<n;++j)p[j]=float((j*7)%17-8)/4.0f;
    for (int i=0;i<m;++i)r[i]=float((i*13)%23-11)/4.0f;
    // Complete references: accumulate every output while traversing rows.
    // Each output sees the original increasing reduction-index order.
    std::vector<float> s(n,0),q(m,0),y(n,0);
    for (int i=0;i<m;++i) for (int j=0;j<n;++j) {
        q[i]+=A[(size_t)i*n+j]*p[j];
        s[j]+=A[(size_t)i*n+j]*r[i];
    }
    for (int i=0;i<m;++i) for (int j=0;j<n;++j)y[j]+=A[(size_t)i*n+j]*q[i];
    int cm=m,cn=n;
    std::vector<float> actual_s(cn+2,12345),actual_q(cm+2,12345),actual_y(cn+2,12345);
    cpu_bicg(A.data(),p.data(),r.data(),actual_s.data()+1,actual_q.data()+1,m,n);
    cpu_atax(A.data(),p.data(),actual_y.data()+1,m,n);
    if (!output_equal(actual_s,s,cn,"BiCG s") || !output_equal(actual_q,q,cm,"BiCG q") ||
        !output_equal(actual_y,y,cn,"ATAX y")) return false;
    if (m==n) {
        std::vector<float> x1(cn+2,12345),x2(cn+2,12345);
        cpu_mvt(A.data(),p.data(),r.data(),x1.data()+1,x2.data()+1,n);
        if (!output_equal(x1,q,cn,"MVT x1") || !output_equal(x2,s,cn,"MVT x2")) return false;
    }
    return true;
}

int main() {
    const int sizes[]={1,2,3,7,8,9,16,31,64,129};
    for (int m:sizes) for (int n:sizes) for (int p=0;p<4;++p)
        if (!check_case(m,n,p)) {fprintf(stderr,"shape=%dx%d pattern=%d\n",m,n,p);return 1;}
    puts("PASS: 400 rectangular ATAX/BiCG cases and 40 square MVT cases; all output values bit-identical to full references; output guards intact");
    return 0;
}
