#define main hotspot3d_production_main
#include "../../tier2/rodinia_hotspot3d/cuda/rodinia_hotspot3d.cu"
#undef main
#include "../upstream/hotspot3d_cpu.h"
#include "upstream/hotspot3d_gpu.cu"
int main(int argc,char** argv)try {
    const auto p=hotspot3d::parameters(argc,argv);
    if(p.n<2 || p.ambient!=80.0f)throw std::invalid_argument("pinned GPU comparison requires n>=2 and ambient80");
    const auto c=hotspot3d::coefficients(p.n);const auto physics=pinned_hotspot3d::physics(p.n,p.n,p.n);
    if(c.dt!=physics.dt)throw std::invalid_argument("original GPU comparison requires an upstream-stable grid");
    auto in=hotspot3d::input(p);const size_t cells=in.temperature.size(),bytes=cells*sizeof(float);
    float *a,*b,*power;CUDA_CHECK(cudaMalloc(&a,bytes));CUDA_CHECK(cudaMalloc(&b,bytes));CUDA_CHECK(cudaMalloc(&power,bytes));
    CUDA_CHECK(cudaMemcpy(power,in.power.data(),bytes,cudaMemcpyHostToDevice));
    std::vector<float> actual(cells),expected(cells);
    for(int variant=0;variant<2;++variant) {
        CUDA_CHECK(cudaMemcpy(a,in.temperature.data(),bytes,cudaMemcpyHostToDevice));
        float* src=a;float* dst=b;
        // Independently compute pinned coefficients for the original kernel.
        const float sdc=physics.dt/physics.cap,ce=sdc/physics.rx,cn=sdc/physics.ry,ct=sdc/physics.rz;
        const float cc=1.0-(2.0*ce+2.0*cn+3.0*ct);
        for(int t=0;t<p.steps;++t) {
            if(variant==0) {
                const int extent=(p.n+p.block-1)/p.block;
                hotspot3d_kernel<<<dim3(extent,extent,extent),dim3(p.block,p.block,p.block)>>>(src,dst,power,p.n,c,p.ambient);
            } else {
                hotspotOpt1<<<dim3((p.n+63)/64,(p.n+3)/4),dim3(64,4)>>>(power,src,dst,sdc,p.n,p.n,p.n,ce,ce,cn,cn,ct,ct,cc);
            }
            CUDA_CHECK(cudaGetLastError());std::swap(src,dst);
        }
        CUDA_CHECK(cudaMemcpy((variant?expected:actual).data(),src,bytes,cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(a));CUDA_CHECK(cudaFree(b));CUDA_CHECK(cudaFree(power));
    // The mappings let NVCC contract different subsets of products into FMAs.
    // Require both complete grids to match the independent FP64 trajectory;
    // their mutual difference is then bounded by the sum of the two bounds.
    const auto reference=hotspot3d::reference(p,c,in);
    const double bound=hotspot3d::error_bound(p,c,in);
    size_t errors=hotspot3d::compare(actual,reference,bound)+hotspot3d::compare(expected,reference,bound);
    size_t different=0;double max_difference=0;
    for(size_t i=0;i<cells;++i) {
        const double difference=std::fabs(static_cast<double>(actual[i])-expected[i]);
        if(actual[i]!=expected[i])++different;
        max_difference=std::max(max_difference,difference);
        if(!std::isfinite(difference) || difference>2*bound)++errors;
    }
    // Diagnostic for a test-only build with FMA contraction disabled.
    if(std::getenv("HS3D_TEST_REQUIRE_EXACT") && different)++errors;
    printf("%s: %zu full final temperatures match pinned hotspotOpt1 within derived bounds; "
           "n=%d steps=%d block=%d different=%zu max_difference=%.9g bound=%.9g\n",
           errors?"FAIL":"PASS",cells,p.n,p.steps,p.block,different,max_difference,bound);
    return errors?1:0;
}catch(const std::exception& e){fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
