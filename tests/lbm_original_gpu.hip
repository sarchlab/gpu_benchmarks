#define main lbm_production_main
#include "../tier2/parboil_lbm/hip/parboil_lbm.hip"
#undef main
namespace pinned {
#include "upstream/lbm/layout_config.h"
#undef SIZE_X
#undef SIZE_Y
#undef SIZE_Z
#define SIZE_X LBM_TEST_N
#define SIZE_Y LBM_TEST_N
#define SIZE_Z LBM_TEST_N
#define DFL1 (1.0f/3.0f)
#define DFL2 (1.0f/18.0f)
#define DFL3 (1.0f/36.0f)
#include "upstream/lbm/lbm_kernel.cu"
#include "upstream/lbm/initialization.h"
}
int main(int argc,char** argv) {
    const int n=LBM_TEST_N,steps=argc>1?std::atoi(argv[1]):10;
    const lbm::Layout l(n);auto in=lbm::input(l);
    std::vector<float> original(l.entries,0),actual(l.entries),expected(l.entries);
    pinned::LBM_initializeGrid(original.data()+l.margin);
    pinned::LBM_initializeSpecialCellsForLDC(original.data()+l.margin);
    if(std::memcmp(in.data(),original.data(),l.bytes()))return 2;
    float *da,*db;HIP_CHECK(hipMalloc(&da,l.bytes()));HIP_CHECK(hipMalloc(&db,l.bytes()));
    for(int variant=0;variant<2;++variant) {
        HIP_CHECK(hipMemcpy(da,in.data(),l.bytes(),hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db,in.data(),l.bytes(),hipMemcpyHostToDevice));
        float *a=da+l.margin,*b=db+l.margin;
        for(int t=0;t<steps;++t) {
            if(variant)pinned::performStreamCollide_kernel<<<dim3(n,n),dim3(n)>>>(a,b);
            else lbm::lbm_collide_stream_kernel<<<dim3(n,n),dim3(n)>>>(a,b,l);
            HIP_CHECK(hipGetLastError());std::swap(a,b);
        }
        HIP_CHECK(hipMemcpy((variant?expected:actual).data(),a-l.margin,l.bytes(),hipMemcpyDeviceToHost));
    }
    HIP_CHECK(hipFree(da));HIP_CHECK(hipFree(db));
    size_t different=0;double maximum=0;
    for(size_t i=0;i<actual.size();++i) {
        if(actual[i]!=expected[i])++different;
        if(!std::isfinite(actual[i]) || !std::isfinite(expected[i]))return 3;
        maximum=std::max(maximum,std::fabs(double(actual[i])-expected[i]));
    }
    const auto reference=lbm::reference(l,steps,lbm::reference_workers(n));
    const auto a=lbm::compare(l,actual,reference),b=lbm::compare(l,expected,reference);
    printf("%s n=%d steps=%d different=%zu max=%.9g production_errors=%zu pinned_errors=%zu\n",
        different || a.errors || b.errors?"FAIL":"PASS",n,steps,different,maximum,a.errors,b.errors);
    return different || a.errors || b.errors ? 1:0;
}
