// Compare every stage with the frozen pinned Altis kernels on identical inputs.
#define main cfd_production_main
#include "../tier2/altis_cfd/altis_cfd.hip"
#undef main
#include "fixtures/cfd_upstream.h"
#include <cstdint>

static void upstream_farfield() {
    auto ff=cfd::farfield();float pressure=1;float3 momentum=make_float3(ff[1],ff[2],ff[3]);
    float3 velocity=make_float3(1.2f,0,0),fx,fy,fz,fe;
    cfd_upstream::compute_flux_contribution(ff[0],momentum,ff[4],pressure,velocity,fx,fy,fz,fe);
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cfd_upstream::ff_variable),ff.data(),5*sizeof(float)));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cfd_upstream::ff_flux_contribution_momentum_x),&fx,sizeof(fx)));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cfd_upstream::ff_flux_contribution_momentum_y),&fy,sizeof(fy)));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cfd_upstream::ff_flux_contribution_momentum_z),&fz,sizeof(fz)));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cfd_upstream::ff_flux_contribution_density_energy),&fe,sizeof(fe)));
}
static uint64_t compare(const char* label,float* a,float* b,size_t count) {
    const size_t chunk=std::min<size_t>(count,1048576);std::vector<float> x(chunk),y(chunk);
    for(size_t off=0;off<count;off+=chunk) {
        const size_t n=std::min(chunk,count-off);
        HIP_CHECK(hipMemcpy(x.data(),a+off,n*sizeof(float),hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(y.data(),b+off,n*sizeof(float),hipMemcpyDeviceToHost));
        for(size_t i=0;i<n;++i)if(!std::isfinite(x[i])||!std::isfinite(y[i])||x[i]!=y[i]) {
            fprintf(stderr,"FAIL upstream %s[%zu]: %.9g != %.9g\n",label,off+i,x[i],y[i]);exit(1);
        }
    }
    return count;
}
int main(int argc,char** argv)try {
    const auto p=cfd::parameters(argc,argv);const int n=p.padded;auto in=cfd::input(p);size_t bytes=static_cast<size_t>(n)*sizeof(float);
    float *a,*b,*old_a,*old_b,*flux_a,*flux_b,*step_a,*step_b,*normals,*areas;int* neighbors;
    HIP_CHECK(hipMalloc(&a,5*bytes));HIP_CHECK(hipMalloc(&b,5*bytes));HIP_CHECK(hipMalloc(&old_a,5*bytes));HIP_CHECK(hipMalloc(&old_b,5*bytes));
    HIP_CHECK(hipMalloc(&flux_a,5*bytes));HIP_CHECK(hipMalloc(&flux_b,5*bytes));HIP_CHECK(hipMalloc(&step_a,bytes));HIP_CHECK(hipMalloc(&step_b,bytes));
    HIP_CHECK(hipMalloc(&normals,12*bytes));HIP_CHECK(hipMalloc(&areas,bytes));HIP_CHECK(hipMalloc(&neighbors,4*bytes));
    HIP_CHECK(hipMemcpy(a,in.initial.data(),5*bytes,hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(b,in.initial.data(),5*bytes,hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(normals,in.normals.data(),12*bytes,hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(areas,in.areas.data(),bytes,hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(neighbors,in.neighbors.data(),4*bytes,hipMemcpyHostToDevice));
    upload_farfield();upstream_farfield();dim3 grid(n/p.block),block(p.block);uint64_t checked=0;
    for(int t=0;t<10;++t) {
        copy_variables_kernel<<<grid,block>>>(n,a,old_a);
        HIP_CHECK(hipMemcpy(old_b,b,5*bytes,hipMemcpyDeviceToDevice));checked+=compare("old",old_a,old_b,5*static_cast<size_t>(n));
        cfd_device::compute_step_factor_kernel<<<grid,block>>>(n,a,areas,step_a);
        cfd_upstream::cuda_compute_step_factor<<<grid,block>>>(n,b,areas,step_b);checked+=compare("step",step_a,step_b,n);
        for(int stage=0;stage<3;++stage) {
            cfd_device::compute_flux_kernel<<<grid,block>>>(n,neighbors,normals,a,flux_a);
            cfd_upstream::cuda_compute_flux<<<grid,block>>>(n,neighbors,normals,b,flux_b);checked+=compare("flux",flux_a,flux_b,5*static_cast<size_t>(n));
            cfd_device::rk_update_kernel<<<grid,block>>>(stage,n,old_a,a,step_a,flux_a);
            cfd_upstream::cuda_time_step<<<grid,block>>>(stage,n,old_b,b,step_b,flux_b);checked+=compare("state",a,b,5*static_cast<size_t>(n));
        }
    }
    HIP_CHECK(hipFree(a));HIP_CHECK(hipFree(b));HIP_CHECK(hipFree(old_a));HIP_CHECK(hipFree(old_b));HIP_CHECK(hipFree(flux_a));HIP_CHECK(hipFree(flux_b));
    HIP_CHECK(hipFree(step_a));HIP_CHECK(hipFree(step_b));HIP_CHECK(hipFree(normals));HIP_CHECK(hipFree(areas));HIP_CHECK(hipFree(neighbors));
    printf("PASS: %llu complete stage values exactly match pinned upstream; cells=%d padded=%d block=%d\n",static_cast<unsigned long long>(checked),p.cells,p.padded,p.block);
    return 0;
} catch(const std::exception& e) {fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
