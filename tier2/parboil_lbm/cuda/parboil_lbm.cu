// Parboil D3Q19 LDC scatter operator; see README.md and UPSTREAM_LICENSE.
#include "../../../common_cuda/bench_timing.h"
#include "lbm_kernel.h"
#include "lbm_reference.h"
#define CUDA_CHECK(cmd) BENCH_TIMING_CHECK(cmd)
int main(int argc,char** argv)try {
    (void)argc;(void)argv; // This benchmark's inputs are environment parameters.
    setvbuf(stdout,nullptr,_IONBF,0);
    const auto p=lbm::parameters();const lbm::Layout l(p.n);
    int device=0;CUDA_CHECK(cudaGetDevice(&device));cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop,device));
    if(p.n>prop.maxThreadsPerBlock || p.n>prop.maxThreadsDim[0] ||
       p.n>prop.maxGridSize[0] || p.n>prop.maxGridSize[1])
        throw std::invalid_argument("LBM upstream row launch exceeds device limits");
    fprintf(stderr,"Device: %s; LBM grid=%d steps=%d omega=1.95\n",prop.name,p.n,p.steps);
    auto initial=lbm::input(l);
    float *da,*db;CUDA_CHECK(cudaMalloc(&da,l.bytes()));CUDA_CHECK(cudaMalloc(&db,l.bytes()));
    CUDA_CHECK(cudaMemcpy(da,initial.data(),l.bytes(),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db,initial.data(),l.bytes(),cudaMemcpyHostToDevice));
    std::vector<float>().swap(initial);
    float *src=da+l.margin,*dst=db+l.margin;
    bench_timer_t timer;bench_timer_create(&timer);bench_lap_t lap;bench_lap_reset(&lap);
    bench_pass_warm();
    for(int t=0;t<p.steps;++t) {
        bench_lap_add(&lap,bench_time_op(&timer,[&](){
            lbm::lbm_collide_stream_kernel<<<dim3(p.n,p.n),dim3(p.n)>>>(src,dst,l);
        }));
        std::swap(src,dst);
    }
    bench_timer_destroy(&timer);
    const unsigned workers=lbm::reference_workers(p.n);
    char params[384];snprintf(params,sizeof(params),
        "\"grid_dim\":%d,\"num_timesteps\":%d,\"block_size\":%d,\"omega\":1.95,"
        "\"verification\":\"full_fp32_upstream_v1\",\"population_tolerance\":1e-5,"
        "\"velocity_tolerance\":1e-5,\"reference_workers\":%u",p.n,p.steps,p.n,workers);
    bench_emit_iter("lbm_collide_stream_kernel",0,&lap,params);bench_emit_summary(lap.gpu_ms);
    std::vector<float> actual(l.entries);
    CUDA_CHECK(cudaMemcpy(actual.data(),src-l.margin,l.bytes(),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(da));CUDA_CHECK(cudaFree(db));
    const auto expected=lbm::reference(l,p.steps,workers);
    const auto result=lbm::compare(l,actual,expected);
    fprintf(stderr,"%s: cells=%zu populations=%zu errors=%zu max_population=%.9g max_velocity=%.9g\n",
        result.errors?"FAIL":"PASS",result.cells,result.populations,result.errors,result.max_population,result.max_velocity);
    return result.errors?1:0;
}catch(const std::exception& e){fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
