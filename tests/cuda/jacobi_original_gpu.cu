// Compare the actual production kernels with frozen PolyBench/GPU kernels.
#define main jacobi_benchmark_main
#include "../../tier2/polybench_jacobi2d/cuda/polybench_jacobi2d.cu"
#undef main
#include "../fixtures/polybench_jacobi_original.h"
#include "../fixtures/polybench_jacobi_original_gpu.h"

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
    CUDA_CHECK(cudaMalloc(&a,bytes));CUDA_CHECK(cudaMalloc(&b,bytes));
    CUDA_CHECK(cudaMalloc(&oa,bytes));CUDA_CHECK(cudaMalloc(&ob,bytes));
    CUDA_CHECK(cudaMemcpy(a,actual.a.data(),bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b,actual.b.data(),bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(oa,original.a.data(),bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ob,original.b.data(),bytes,cudaMemcpyHostToDevice));
    const dim3 block(block_width,block_width),grid((n-2+15)/16,(n-2+15)/16);
    const dim3 original_block(32,8),original_grid((n+31)/32,(n+7)/8);
    for(int t=0;t<problem.steps;++t) {
        jacobi2d_kernel<<<grid,block>>>(a,b,n);CUDA_CHECK(cudaGetLastError());
        jacobi2d_copy_kernel<<<grid,block>>>(b,a,n);CUDA_CHECK(cudaGetLastError());
        pinned_jacobi_gpu::runJacobiCUDA_kernel1<<<original_grid,original_block>>>(n,oa,ob);
        CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
        pinned_jacobi_gpu::runJacobiCUDA_kernel2<<<original_grid,original_block>>>(n,oa,ob);
        CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaMemcpy(actual.a.data(),a,bytes,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(actual.b.data(),b,bytes,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(original.a.data(),oa,bytes,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(original.b.data(),ob,bytes,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(a));CUDA_CHECK(cudaFree(b));CUDA_CHECK(cudaFree(oa));CUDA_CHECK(cudaFree(ob));
    if(std::memcmp(actual.a.data(),original.a.data(),bytes) ||
       std::memcmp(actual.b.data(),original.b.data(),bytes))
        throw std::runtime_error("production outputs differ from pinned original GPU kernels");
    printf("PASS: %zu input values and %zu complete output values byte-equal to pinned GPU; n=%d tsteps=%d\n",
           2*actual.a.size(),2*actual.a.size(),n,problem.steps);
    return 0;
} catch(const std::exception& error) {
    fprintf(stderr,"FAIL: %s\n",error.what());return 1;
}
