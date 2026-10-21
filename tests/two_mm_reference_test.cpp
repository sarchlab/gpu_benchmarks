
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <limits>
#include "../tier2/polybench_2mm/hip/cpu_reference.h"
int main() {
  int cases=0;
  const int sizes[]={1,2,3,15,16,17,31,32,33,64,65,96,128,129};
  for (int n:sizes) for (float alpha:{1.5f,0.0f,-0.5f,32412.0f}) for(float beta:{1.2f,0.0f,-0.75f}) {
    const size_t count=(size_t)n*n;
    std::vector<float>A(count),B(count),C(count),D(count);
    for(size_t i=0;i<count;++i) {A[i]=float((int)(i*7%17)-8)/8; B[i]=float((int)(i*11%19)-9)/8; C[i]=float((int)(i*13%23)-11)/8; D[i]=float((int)(i*3%13)-6)/8;}
    const auto original_A=A, original_B=B, original_C=C, original_D=D;
    // Qualified per-output double reference with no blocked traversal.
    std::vector<double> tmp(count,0),result(count,0);
    for(int i=0;i<n;++i)for(int j=0;j<n;++j)for(int k=0;k<n;++k)
      tmp[(size_t)i*n+j]+=alpha*(double)A[(size_t)i*n+k]*B[(size_t)k*n+j];
    for(int i=0;i<n;++i)for(int j=0;j<n;++j) {
      result[(size_t)i*n+j]=beta*(double)D[(size_t)i*n+j];
      for(int k=0;k<n;++k)result[(size_t)i*n+j]+=tmp[(size_t)i*n+k]*C[(size_t)k*n+j];
    }
    const auto actual=cpu_2mm(A.data(),B.data(),C.data(),D.data(),n,alpha,beta);
    if (A!=original_A || B!=original_B || C!=original_C || D!=original_D ||
        actual.tmp.size()!=count || actual.output.size()!=count) return 1;
    if (memcmp(tmp.data(),actual.tmp.data(),count*sizeof(double)) || memcmp(result.data(),actual.output.data(),count*sizeof(double))) {
      fprintf(stderr,"FAIL N%d alpha%g beta%g\n",n,alpha,beta);return 1;
    }
    ++cases;
  }
  if (!two_mm_value_matches(0,0) || !two_mm_value_matches(4e-4f,0) ||
      two_mm_value_matches(6e-4f,0) || !two_mm_value_matches(100.04f,100) ||
      two_mm_value_matches(100.06f,100) || !two_mm_value_matches(-100.04f,-100) ||
      two_mm_value_matches(-100.06f,-100)) return 1;
  for (float bad : {std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()})
      if (two_mm_value_matches(bad,0) || two_mm_value_matches(0,bad) ||
          two_mm_value_matches(bad,bad)) return 1;
  printf("PASS: %d complete two-product references bit-identical; original inputs preserved; finite/tolerance predicates checked\n",cases);
  return 0;
}
