#include "../tier2/polybench_jacobi2d/jacobi_reference.h"
#include "fixtures/polybench_jacobi_original.h"
#include <limits>
static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
int main(int argc,char**) {
    size_t input_values=0,output_values=0,trajectories=0,faults=0;
    std::vector<int> sizes={3,4,5,15,16,17,31,32,33,63,64,65,127};
    if(argc>1)sizes={64,128,192,256,384,512,640,768,1024,1280,1536,2048,2560,3072,3584,4096,5120,6144,7168,8192};
    for(int n:sizes) {
        const auto initial=jacobi::input(n);pinned_jacobi::Original oracle(n);
        require(initial.a==oracle.a && initial.b==oracle.b,"pinned complete input differs");input_values+=2*initial.a.size();
        for(int steps:{1,2,3,20}) {
            if(argc>1)break; // Full declared measured trajectories run on H100.
            oracle=pinned_jacobi::Original(n);oracle.solve(steps);
            const auto result=jacobi::reference({n,steps});
            require(result.a==oracle.a && result.b==oracle.b,"pinned complete trajectory differs");
            ++trajectories;output_values+=2*result.a.size();
            for(int i=0;i<n;++i)for(int j=0;j<n;++j) {
                const size_t k=static_cast<size_t>(i)*n+j;
                if(i==0 || j==0 || i==n-1 || j==n-1)
                    require(result.a[k]==initial.a[k] && result.b[k]==initial.b[k],"initialized boundary changed");
                else require(result.a[k]==result.b[k],"copy-back missing");
            }
        }
    }
    const int n=65;const auto expected=jacobi::reference({n,3});auto actual=expected;
    for(int array=0;array<2;++array) {
        auto& values=array==0?actual.a:actual.b;
        for(size_t i:{size_t(0),size_t(1),size_t(n-1),size_t(n),size_t(n+1),size_t(n*n/2),size_t(n*(n-1)),size_t(n*n-1)}) {
            const float saved=values[i];
            for(float bad:{std::nextafter(saved,std::numeric_limits<float>::infinity()),
                    std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()}) {
                values[i]=bad;require(jacobi::compare(actual,expected)==1,"corruption accepted");values[i]=saved;++faults;
            }
        }
    }
    // Twelve failures prove full counting after the ten diagnostic limit.
    for(size_t i=0;i<12;++i)actual.a[i]+=1.0f;
    require(jacobi::compare(actual,expected)==12,"mismatch count truncated");++faults;
    auto reference=expected;reference.b.back()=std::numeric_limits<float>::quiet_NaN();
    require(jacobi::compare(expected,reference)==1,"nonfinite reference accepted");++faults;
    bool threw=false;reference=expected;reference.b.pop_back();
    try{jacobi::compare(expected,reference);}catch(const std::invalid_argument&){threw=true;}
    require(threw,"dimension mismatch accepted");
    // Independent 3x3 hand calculation fixes the timestep unit and operation order.
    const auto tiny=jacobi::input(3);const auto first=jacobi::reference({3,1});
    const float center=0.2f*(tiny.a[4]+tiny.a[3]+tiny.a[5]+tiny.a[7]+tiny.a[1]);
    require(first.a[4]==center && first.b[4]==center,"one sweep plus copy-back differs");
    printf("PASS: %zu complete pinned input values; %zu complete pinned trajectories (%zu values); %zu corruptions rejected and shape guard passes\n",input_values,trajectories,output_values,faults);
}
