#define main hotspot_production_main
#include "../tier2/rodinia_hotspot/hip/rodinia_hotspot.hip"
#undef main
#include "upstream/hotspot_physics.h"
#include "upstream/hotspot_gpu.hip"
int main(int argc,char** argv)try {
    const auto p=hotspot::parameters(argc,argv);const auto g=hotspot::geometry(p);const auto c=hotspot::coefficients(p.n);
    const auto pin=pinned_hotspot::physics(p.n,p.n);
    if(c.cap!=pin.cap || c.rx!=pin.rx || c.ry!=pin.ry || c.rz!=pin.rz || c.upstream_dt!=pin.dt)
        throw std::runtime_error("production physical parameters differ from pinned source");
    auto in=hotspot::input(p);const size_t cells=g.cells,bytes=cells*sizeof(float);
    float *a,*b,*power;HIP_CHECK(hipMalloc(&a,bytes));HIP_CHECK(hipMalloc(&b,bytes));HIP_CHECK(hipMalloc(&power,bytes));
    HIP_CHECK(hipMemcpy(power,in.power.data(),bytes,hipMemcpyHostToDevice));
    std::vector<float> actual(cells),expected(cells);
    const dim3 grid(g.blocks,g.blocks),block(16,16);
    for(int variant=0;variant<2;++variant) {
        HIP_CHECK(hipMemcpy(a,in.temperature.data(),bytes,hipMemcpyHostToDevice));float* src=a;float* dst=b;
        for(int done=0;done<p.steps;) {
            const int it=std::min(p.pyramid,p.steps-done);
            if(variant==0)calculate_temp<<<grid,block>>>(it,power,src,dst,p.n,p.n,p.pyramid,p.pyramid,c.cap,c.rx,c.ry,c.rz,c.dt,0.001f);
            else pinned_calculate_temp<<<grid,block>>>(it,power,src,dst,p.n,p.n,p.pyramid,p.pyramid,pin.cap,pin.rx,pin.ry,pin.rz,c.dt,0.001f);
            HIP_CHECK(hipGetLastError());std::swap(src,dst);done+=it;
        }
        HIP_CHECK(hipMemcpy((variant?expected:actual).data(),src,bytes,hipMemcpyDeviceToHost));
    }
    HIP_CHECK(hipFree(a));HIP_CHECK(hipFree(b));HIP_CHECK(hipFree(power));
    size_t errors=0;
    for(size_t i=0;i<cells;++i)if(!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || actual[i]!=expected[i]) {
        if(errors<10)fprintf(stderr,"Pinned mismatch[%zu]: actual=%.9g pinned=%.9g\n",i,actual[i],expected[i]);++errors;
    }
    printf("%s: %zu complete temperatures exactly match pinned calculate_temp; n=%d steps=%d pyramid=%d limited=%d\n",errors?"FAIL":"PASS",cells,p.n,p.steps,p.pyramid,c.dt<c.upstream_dt);
    return errors?1:0;
}catch(const std::exception& e){fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
