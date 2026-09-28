// Pinned Altis/Rodinia GPU particle filter; see README.md and UPSTREAM_LICENSE.
#include "../../common/bench_timing.h"
#include "particlefilter_reference.h"
#include "particlefilter_kernels.h"
#include <type_traits>
#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

template<class T> struct Device {
    T* data=nullptr;size_t count;
    explicit Device(size_t n):count(n) {HIP_CHECK(hipMalloc(&data,n*sizeof(T)));}
    ~Device() {if(data)hipFree(data);}
    Device(const Device&)=delete;Device& operator=(const Device&)=delete;
    void upload(const T* host) {HIP_CHECK(hipMemcpy(data,host,count*sizeof(T),hipMemcpyHostToDevice));}
};
struct Verification {
    size_t checked=0,errors=0;
    template<class T,class Expected> void buffer(const char* name,const Device<T>& device,const Expected& expected,bool relative_only=false) {
        const size_t chunk=std::min<size_t>(device.count,131072);std::vector<T> actual(chunk);
        for(size_t start=0;start<device.count;start+=chunk) {
            const size_t count=std::min(chunk,device.count-start);
            HIP_CHECK(hipMemcpy(actual.data(),device.data+start,count*sizeof(T),hipMemcpyDeviceToHost));
            for(size_t j=0;j<count;++j) {
                const auto want=expected(start+j);const double a=double(actual[j]),b=double(want);
                const bool good=std::is_integral<T>::value?actual[j]==want:
                    relative_only?(std::isfinite(a)&&std::isfinite(b)&&std::fabs(a-b)<=4e-300+1e-10*std::fabs(b)):pf::matches(a,b);
                ++checked;if(!good) {if(errors++<10)std::fprintf(stderr,"Mismatch %s[%zu]: GPU=%.17g CPU=%.17g\n",name,start+j,a,b);}
            }
        }
    }
};
int main(int argc,char** argv)try {
    setvbuf(stdout,nullptr,_IONBF,0);
    const int n=pf::parameters(argc,argv),blocks=(n+pf::block-1)/pf::block;
    int device=0;HIP_CHECK(hipGetDevice(&device));hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop,device));
    if(prop.maxThreadsPerBlock<pf::block||prop.maxThreadsDim[0]<pf::block||blocks>prop.maxGridSize[0])
        throw std::runtime_error("Particle Filter launch exceeds selected device limits");
    size_t free_bytes=0,total_bytes=0;HIP_CHECK(hipMemGetInfo(&free_bytes,&total_bytes));
    const size_t bytes=pf::device_bytes(n),warmer=bench_vector_add_warmup()?BENCH_WARMER_BYTES:0;
    if(bytes+warmer>free_bytes)throw std::runtime_error("Particle Filter buffers plus warmer exceed available device memory");
    const pf::Input in=pf::input(n);
    Device<double> x(n),y(n),xj(n),yj(n),likelihood(n),weights(n),cdf(n),u(n),partial(blocks),u0(1);
    Device<int32_t> seeds(n);Device<int> indices(size_t(n)*pf::neighbors),offsets(2*pf::neighbors);
    Device<unsigned char> video(pf::video_size);
    std::vector<double> initial(n,pf::width/2);xj.upload(initial.data());yj.upload(initial.data());
    std::vector<double>().swap(initial);
    static_assert(sizeof(pf::Offset)==2*sizeof(int),"Offset upload layout");
    seeds.upload(in.seeds.data());offsets.upload(in.disk[0].data());video.upload(in.video.data());
    std::fprintf(stderr,"Device: %s (id=%d); Particle Filter particles=%d, video=128x128x11, updates=10, double precision\n",prop.name,device,n);
    bench_timer_t timer;bench_timer_create(&timer);bench_lap_t times[6];for(auto& t:times)bench_lap_reset(&t);
    bench_pass_warm();
    for(int frame=1;frame<pf::frames;++frame) {
        bench_lap_add(&times[0],bench_time_op(&timer,[&]() {pf_gpu::likelihood_kernel<<<blocks,pf::block>>>(n,frame,video.data,offsets.data,x.data,y.data,xj.data,yj.data,indices.data,likelihood.data,weights.data,seeds.data,partial.data);}));
        bench_lap_add(&times[1],bench_time_op(&timer,[&]() {pf_gpu::sum_kernel<<<blocks,pf::block>>>(partial.data,blocks);}));
        bench_lap_add(&times[2],bench_time_op(&timer,[&]() {pf_gpu::normalize_weights_kernel<<<blocks,pf::block>>>(weights.data,partial.data,n);}));
        bench_lap_add(&times[3],bench_time_op(&timer,[&]() {pf_gpu::cdf_kernel<<<1,pf::block>>>(weights.data,cdf.data,u0.data,seeds.data,n);}));
        bench_lap_add(&times[4],bench_time_op(&timer,[&]() {pf_gpu::thresholds_kernel<<<blocks,pf::block>>>(u0.data,u.data,n);}));
        bench_lap_add(&times[5],bench_time_op(&timer,[&]() {pf_gpu::find_index_kernel<<<blocks,pf::block>>>(x.data,y.data,cdf.data,u.data,xj.data,yj.data,n);}));
    }
    bench_timer_destroy(&timer);
    const char* names[]={"likelihood_kernel","sum_kernel","normalize_weights_kernel","cdf_kernel","thresholds_kernel","find_index_kernel"};
    char params[1200];int written=std::snprintf(params,sizeof(params),
        "\"size\":%d,\"seed\":12345,\"width\":128,\"height\":128,\"frames\":11,\"timesteps\":10,\"neighbors\":69,"
        "\"block_size\":512,\"kernel_launches\":10,\"particle_updates\":%llu,\"device_buffer_bytes\":%zu,"
        "\"algorithm\":\"altis_particlefilter_float_library_v1\",\"input\":\"fixed_video_safe_lcg_v1\","
        "\"precision\":\"fp64\",\"verification\":\"full_final_aux_cpu_fp64_v1\"",n,10ULL*n,bytes);
    if(written<0||size_t(written)>=sizeof(params))throw std::runtime_error("Particle Filter metadata overflow");
    double total=0;for(int k=0;k<6;++k) {bench_emit_iter(names[k],0,&times[k],params);total+=times[k].gpu_ms;}bench_emit_summary(total);
    // No GPU rerun, mid-trajectory readbacks, host scans or sampled validation.
    const pf::Reference expected=pf::reference(n,in);
    Verification result;
    result.buffer("x",x,[&](size_t i){return expected.x[i];});
    result.buffer("y",y,[&](size_t i){return expected.y[i];});
    result.buffer("xj",xj,[&](size_t i){return expected.xj[i];});
    result.buffer("yj",yj,[&](size_t i){return expected.yj[i];});
    result.buffer("likelihood",likelihood,[&](size_t i){return expected.likelihood[i];});
    result.buffer("weights",weights,[&](size_t i){return expected.weights[i];},true);
    result.buffer("cdf",cdf,[&](size_t i){return expected.cdf[i];});
    result.buffer("u",u,[&](size_t i){return expected.u[i];});
    result.buffer("partial",partial,[&](size_t i){return expected.partial[i];},true);
    result.buffer("u0",u0,[&](size_t){return expected.u0;});
    result.buffer("seeds",seeds,[&](size_t i){return expected.seeds[i];});
    result.buffer("indices",indices,[&](size_t i){return pf::gather_index(expected.x[i/pf::neighbors],expected.y[i/pf::neighbors],in.disk[i%pf::neighbors],pf::steps);});
    std::fprintf(stderr,"Particle Filter verification: %s (verified %d particles, 10 timesteps, %zu final and auxiliary values; %zu mismatches)\n",result.errors?"FAIL":"PASS",n,result.checked,result.errors);
    return result.errors?1:0;
} catch(const std::exception& e) {std::fprintf(stderr,"FAIL: Particle Filter setup/reference unavailable: %s\n",e.what());return 1;}
