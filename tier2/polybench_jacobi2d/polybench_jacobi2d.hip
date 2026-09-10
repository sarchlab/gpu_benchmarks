// PolyBench/GPU 1.0 Jacobi-2D: stencil A->B, then interior copy B->A.
// Pin and source attribution: README.md and UPSTREAM_LICENSE.
#include "../../common/bench_timing.h"
#include "jacobi_reference.h"
#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

static constexpr int block_width=16;
__global__ void jacobi2d_kernel(const float* a,float* b,int n) {
    const int i=blockIdx.y*blockDim.y+threadIdx.y+1;
    const int j=blockIdx.x*blockDim.x+threadIdx.x+1;
    if(i>=n-1 || j>=n-1)return;
    const int k=i*n+j;
    // Preserve the pin's center, left, right, down, up addition order.
    b[k]=0.2f*(a[k]+a[k-1]+a[k+1]+a[k+n]+a[k-n]);
}
__global__ void jacobi2d_copy_kernel(const float* b,float* a,int n) {
    const int i=blockIdx.y*blockDim.y+threadIdx.y+1;
    const int j=blockIdx.x*blockDim.x+threadIdx.x+1;
    if(i<n-1 && j<n-1)a[i*n+j]=b[i*n+j];
}

int main(int argc,char** argv) try {
    setvbuf(stdout,nullptr,_IONBF,0);
    const jacobi::Problem problem=jacobi::parameters(argc,argv);
    const int n=problem.n,steps=problem.steps;
    const size_t cells=static_cast<size_t>(n)*n,bytes=cells*sizeof(float);
    const dim3 block(block_width,block_width);
    const dim3 grid((n-2+block_width-1)/block_width,(n-2+block_width-1)/block_width);
    int device;HIP_CHECK(hipGetDevice(&device));hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop,device));
    if(block.x>static_cast<unsigned>(prop.maxThreadsDim[0]) || block.y>static_cast<unsigned>(prop.maxThreadsDim[1]) ||
       block.x*block.y>static_cast<unsigned>(prop.maxThreadsPerBlock) ||
       grid.x>static_cast<unsigned>(prop.maxGridSize[0]) || grid.y>static_cast<unsigned>(prop.maxGridSize[1]))
        throw std::invalid_argument("Jacobi launch geometry exceeds device limits");
    fprintf(stderr,"Device: %s (id=%d)\nPolyBench/GPU Jacobi: %d x %d, %d stencil/copy timesteps\n",prop.name,device,n,n,steps);
    jacobi::Grids actual=jacobi::input(n);
    float *d_a,*d_b;HIP_CHECK(hipMalloc(&d_a,bytes));HIP_CHECK(hipMalloc(&d_b,bytes));
    HIP_CHECK(hipMemcpy(d_a,actual.a.data(),bytes,hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b,actual.b.data(),bytes,hipMemcpyHostToDevice));
    bench_timer_t timer;bench_timer_create(&timer);
    bench_lap_t stencil,copy;bench_lap_reset(&stencil);bench_lap_reset(&copy);
    bench_pass_warm();
    for(int t=0;t<steps;++t) {
        bench_lap_add(&stencil,bench_time_op(&timer,[&](){jacobi2d_kernel<<<grid,block>>>(d_a,d_b,n);}));
        bench_lap_add(&copy,bench_time_op(&timer,[&](){jacobi2d_copy_kernel<<<grid,block>>>(d_b,d_a,n);}));
    }
    const unsigned long long interior=static_cast<unsigned long long>(n-2)*(n-2);
    char params[768];snprintf(params,sizeof(params),
        "\"n\":%d,\"tsteps\":%d,\"block_x\":%u,\"block_y\":%u,\"grid_x\":%u,\"grid_y\":%u,"
        "\"threads_per_launch\":%llu,\"interior_cells\":%llu,\"kernel_launches\":%d,\"updated_cells\":%llu,"
        "\"stencil_sweeps_per_step\":1,\"copy_sweeps_per_step\":1,\"precision\":\"fp32\","
        "\"boundary\":\"fixed_initial_values\",\"input\":\"polybench_gpu_jacobi_v1\","
        "\"algorithm\":\"polybench_gpu_jacobi_copyback_v1\",\"verification\":\"full_cpu_exact_v1\"",
        n,steps,block.x,block.y,grid.x,grid.y,
        static_cast<unsigned long long>(grid.x)*grid.y*block.x*block.y,interior,steps,interior*steps);
    bench_emit_iter("jacobi2d_kernel",0,&stencil,params);
    bench_emit_iter("jacobi2d_copy_kernel",0,&copy,params);
    bench_emit_summary(stencil.gpu_ms+copy.gpu_ms);bench_timer_destroy(&timer);
    // Verify both complete actual measured grids, including their distinct borders.
    HIP_CHECK(hipMemcpy(actual.a.data(),d_a,bytes,hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(actual.b.data(),d_b,bytes,hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_a));HIP_CHECK(hipFree(d_b));
    const jacobi::Grids expected=jacobi::reference(problem);
    const size_t errors=jacobi::compare(actual,expected);
    fprintf(stderr,"%s (verified %zu values across both complete measured grids; %zu mismatches)\n",errors?"FAIL":"PASS",2*cells,errors);
    return errors?EXIT_FAILURE:EXIT_SUCCESS;
} catch(const std::exception& error) {
    fprintf(stderr,"FAIL: Jacobi setup/reference unavailable: %s\n",error.what());return EXIT_FAILURE;
}
