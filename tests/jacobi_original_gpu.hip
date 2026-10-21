// Compare the actual production kernels with frozen PolyBench/GPU kernels.
#define main jacobi_benchmark_main
#include "../tier2/polybench_jacobi2d/hip/polybench_jacobi2d.hip"
#undef main
#include "fixtures/polybench_jacobi_original.h"
#include "fixtures/polybench_jacobi_original_gpu.h"

int main(int argc,char** argv) try {
    const auto problem=jacobi::parameters(argc,argv);
    const int n=problem.n;
    auto actual=jacobi::input(n);
    pinned_jacobi::Original original(n);
    const size_t bytes=actual.a.size()*sizeof(float);
    if(std::memcmp(actual.a.data(),original.a.data(),bytes) ||
       std::memcmp(actual.b.data(),original.b.data(),bytes))
        throw std::runtime_error("production inputs differ from pinned original");
    float *a,*b,*oa,*ob;
    HIP_CHECK(hipMalloc(&a,bytes));HIP_CHECK(hipMalloc(&b,bytes));
    HIP_CHECK(hipMalloc(&oa,bytes));HIP_CHECK(hipMalloc(&ob,bytes));
    HIP_CHECK(hipMemcpy(a,actual.a.data(),bytes,hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b,actual.b.data(),bytes,hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(oa,original.a.data(),bytes,hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(ob,original.b.data(),bytes,hipMemcpyHostToDevice));
    const dim3 block(block_width,block_width),grid((n-2+15)/16,(n-2+15)/16);
    const dim3 original_block(32,8),original_grid((n+31)/32,(n+7)/8);
    for(int t=0;t<problem.steps;++t) {
        jacobi2d_kernel<<<grid,block>>>(a,b,n);HIP_CHECK(hipGetLastError());
        jacobi2d_copy_kernel<<<grid,block>>>(b,a,n);HIP_CHECK(hipGetLastError());
        pinned_jacobi_gpu::runJacobiCUDA_kernel1<<<original_grid,original_block>>>(n,oa,ob);
        HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());
        pinned_jacobi_gpu::runJacobiCUDA_kernel2<<<original_grid,original_block>>>(n,oa,ob);
        HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());
    }
    HIP_CHECK(hipMemcpy(actual.a.data(),a,bytes,hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(actual.b.data(),b,bytes,hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(original.a.data(),oa,bytes,hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(original.b.data(),ob,bytes,hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(a));HIP_CHECK(hipFree(b));HIP_CHECK(hipFree(oa));HIP_CHECK(hipFree(ob));
    if(std::memcmp(actual.a.data(),original.a.data(),bytes) ||
       std::memcmp(actual.b.data(),original.b.data(),bytes))
        throw std::runtime_error("production outputs differ from pinned original GPU kernels");
    printf("PASS: %zu input values and %zu complete output values byte-equal to pinned GPU; n=%d tsteps=%d\n",
           2*actual.a.size(),2*actual.a.size(),n,problem.steps);
    return 0;
} catch(const std::exception& error) {
    fprintf(stderr,"FAIL: %s\n",error.what());return 1;
}
