// Rodinia 3.1 HotSpot3D, portable one-cell mapping.
// See README.md for the pinned source and UPSTREAM_LICENSE for notices.
#include "../../../common_cuda/bench_timing.h"
#include "hotspot3d_reference.h"
#define CUDA_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

__global__ void hotspot3d_kernel(const float* __restrict__ src,float* __restrict__ dst,
                                 const float* __restrict__ power,int n,
                                 hotspot3d::Coefficients c,float ambient) {
    const int x=blockIdx.x*blockDim.x+threadIdx.x;
    const int y=blockIdx.y*blockDim.y+threadIdx.y;
    const int z=blockIdx.z*blockDim.z+threadIdx.z;
    if(x>=n || y>=n || z>=n)return;
    const int plane=n*n,i=z*plane+y*n+x;
    const int w=x?i-1:i,e=x+1<n?i+1:i;
    const int north=y?i-n:i,south=y+1<n?i+n:i;
    const int bottom=z?i-plane:i,top=z+1<n?i+plane:i;
    // Preserve the pinned hotspotOpt1 GPU sum order and all six clamped faces.
    dst[i]=c.cc*src[i]+c.cw*src[w]+c.ce*src[e]+c.cs*src[south]+c.cn*src[north]
           +c.cb*src[bottom]+c.ct*src[top]+c.sdc*power[i]+c.ct*ambient;
}

int main(int argc,char** argv)try {
    setvbuf(stdout,nullptr,_IONBF,0);
    const hotspot3d::Problem problem=hotspot3d::parameters(argc,argv);
    const hotspot3d::Coefficients coefficients=hotspot3d::coefficients(problem.n);
    hotspot3d::admissible(coefficients);
    const int n=problem.n,b=problem.block;
    const size_t cells=static_cast<size_t>(n)*n*n,bytes=cells*sizeof(float);
    const int extent=(n+b-1)/b;
    int device=0;CUDA_CHECK(cudaGetDevice(&device));cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop,device));
    if(b*b*b>prop.maxThreadsPerBlock || b>prop.maxThreadsDim[0] || b>prop.maxThreadsDim[1] || b>prop.maxThreadsDim[2]
       || extent>prop.maxGridSize[0] || extent>prop.maxGridSize[1] || extent>prop.maxGridSize[2])
        throw std::invalid_argument("HotSpot3D launch geometry exceeds device limits");
    fprintf(stderr,"Device: %s; grid=%d^3 steps=%d block=%d^3\n",prop.name,n,problem.steps,b);
    hotspot3d::Input in=hotspot3d::input(problem);
    const double bound=hotspot3d::error_bound(problem,coefficients,in);
    float *d_a=nullptr,*d_b=nullptr,*d_power=nullptr;
    CUDA_CHECK(cudaMalloc(&d_a,bytes));CUDA_CHECK(cudaMalloc(&d_b,bytes));CUDA_CHECK(cudaMalloc(&d_power,bytes));
    CUDA_CHECK(cudaMemcpy(d_a,in.temperature.data(),bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_power,in.power.data(),bytes,cudaMemcpyHostToDevice));
    const dim3 block(b,b,b),grid(extent,extent,extent);
    char params[1400];
    const int written=snprintf(params,sizeof(params),
        "\"grid_size\":%d,\"block_size\":%d,\"num_iterations\":%d,\"amb_temp\":%.17g,\"effective_amb_temp\":%.9g,"
        "\"block_x\":%d,\"block_y\":%d,\"block_z\":%d,\"grid_x\":%d,\"grid_y\":%d,\"grid_z\":%d,"
        "\"cells\":%zu,\"threads_per_launch\":%llu,\"kernel_launches\":%d,\"updated_cells\":%llu,"
        "\"dt\":%.9g,\"upstream_dt\":%.9g,\"timestep_limited\":%s,\"simulated_time_seconds\":%.17g,\"absolute_error_bound\":%.17g,\"precision\":\"fp32\",\"boundary\":\"clamped_six_faces\","
        "\"input\":\"libc_rand42_interleaved_v1\",\"algorithm\":\"rodinia31_hotspot3d_v1\","
        "\"verification\":\"full_cpu_fp64_bound_v1\"",
        n,b,problem.steps,problem.requested_ambient,problem.ambient,b,b,b,extent,extent,extent,cells,
        static_cast<unsigned long long>(extent)*extent*extent*b*b*b,problem.steps,
        static_cast<unsigned long long>(cells)*problem.steps,coefficients.dt,coefficients.upstream_dt,coefficients.dt<coefficients.upstream_dt?"true":"false",
        static_cast<double>(coefficients.dt)*problem.steps,bound);
    if(written<0 || static_cast<size_t>(written)>=sizeof(params))throw std::runtime_error("HotSpot3D metadata overflow");
    bench_timer_t timer;bench_timer_create(&timer);bench_lap_t elapsed;bench_lap_reset(&elapsed);
    float* src=d_a;float* dst=d_b;
    bench_pass_warm();
    for(int t=0;t<problem.steps;++t) {
        bench_lap_add(&elapsed,bench_time_op(&timer,[&]() {
            hotspot3d_kernel<<<grid,block>>>(src,dst,d_power,n,coefficients,problem.ambient);
        }));
        std::swap(src,dst);
    }
    bench_timer_destroy(&timer);
    bench_emit_iter("hotspot3d_kernel",0,&elapsed,params);bench_emit_summary(elapsed.gpu_ms);
    // Verify the complete actual final measured buffer after the last swap.
    std::vector<float> actual(cells);
    CUDA_CHECK(cudaMemcpy(actual.data(),src,bytes,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a));CUDA_CHECK(cudaFree(d_b));CUDA_CHECK(cudaFree(d_power));
    const std::vector<double> expected=hotspot3d::reference(problem,coefficients,in);
    const size_t errors=hotspot3d::compare(actual,expected,bound);
    fprintf(stderr,"%s (verified %zu values in the complete measured final grid; %zu mismatches; bound=%.9g)\n",
            errors?"FAIL":"PASS",cells,errors,bound);
    return errors?1:0;
} catch(const std::exception& e) {
    fprintf(stderr,"FAIL: HotSpot3D setup/reference unavailable: %s\n",e.what());return 1;
}
