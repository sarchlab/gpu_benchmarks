// Pinned Altis Euler solver with owner-approved synthetic tetrahedral input.
// See README.md and UPSTREAM_LICENSE for source, input and measurement details.
#include "../../../common/bench_timing.h"
#include "cfd_reference.h"
#include "cfd_kernels.h"
#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

__global__ void copy_variables_kernel(int n,const float* state,float* old) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    for(int k=0;k<5;++k)old[i+k*n]=state[i+k*n];
}

static void upload_farfield() {
    auto ff=cfd::farfield();float pressure=1.0f;
    float3 momentum=make_float3(ff[1],ff[2],ff[3]);
    float3 velocity=make_float3(1.2f,0.0f,0.0f),fx,fy,fz,fe;
    cfd_device::compute_flux_contribution(ff[0],momentum,ff[4],pressure,velocity,fx,fy,fz,fe);
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cfd_device::ff_variable),ff.data(),5*sizeof(float)));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cfd_device::ff_flux_contribution_momentum_x),&fx,sizeof(fx)));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cfd_device::ff_flux_contribution_momentum_y),&fy,sizeof(fy)));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cfd_device::ff_flux_contribution_momentum_z),&fz,sizeof(fz)));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cfd_device::ff_flux_contribution_density_energy),&fe,sizeof(fe)));
}

static size_t verify_buffer(const char* name,const float* device,const std::vector<double>& expected,size_t& printed) {
    const size_t chunk=std::min<size_t>(expected.size(),1048576);
    std::vector<float> actual(chunk);size_t errors=0;
    for(size_t start=0;start<expected.size();start+=chunk) {
        const size_t count=std::min(chunk,expected.size()-start);
        HIP_CHECK(hipMemcpy(actual.data(),device+start,count*sizeof(float),hipMemcpyDeviceToHost));
        for(size_t i=0;i<count;++i)if(!cfd::matches(actual[i],expected[start+i])) {
            ++errors;if(printed++<10)std::fprintf(stderr,"Mismatch %s[%zu]: GPU=%.9g CPU=%.17g\n",name,start+i,actual[i],expected[start+i]);
        }
    }
    return errors;
}

int main(int argc,char** argv)try {
    setvbuf(stdout,nullptr,_IONBF,0);
    const cfd::Problem p=cfd::parameters(argc,argv);const int n=p.padded;
    int device=0;HIP_CHECK(hipGetDevice(&device));hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop,device));
    if(p.block>prop.maxThreadsPerBlock||p.block>prop.maxThreadsDim[0]||n/p.block>prop.maxGridSize[0])
        throw std::invalid_argument("CFD launch exceeds selected device limits");
    const size_t bytes=static_cast<size_t>(n)*sizeof(float);
    const size_t device_bytes=33*bytes,warmer_bytes=bench_vector_add_warmup()?BENCH_WARMER_BYTES:0;
    size_t free_bytes=0,total_bytes=0;HIP_CHECK(hipMemGetInfo(&free_bytes,&total_bytes));
    if(device_bytes+warmer_bytes>free_bytes)throw std::runtime_error("CFD buffers plus warmer exceed available device memory");
    cfd::Input in=cfd::input(p);
    float *state=nullptr,*old=nullptr,*flux=nullptr,*factor=nullptr,*areas=nullptr,*normals=nullptr;int* neighbors=nullptr;
    HIP_CHECK(hipMalloc(&state,5*bytes));HIP_CHECK(hipMalloc(&old,5*bytes));HIP_CHECK(hipMalloc(&flux,5*bytes));
    HIP_CHECK(hipMalloc(&factor,bytes));HIP_CHECK(hipMalloc(&areas,bytes));HIP_CHECK(hipMalloc(&normals,12*bytes));HIP_CHECK(hipMalloc(&neighbors,4*bytes));
    HIP_CHECK(hipMemcpy(state,in.initial.data(),5*bytes,hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(areas,in.areas.data(),bytes,hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(normals,in.normals.data(),12*bytes,hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(neighbors,in.neighbors.data(),4*bytes,hipMemcpyHostToDevice));
    upload_farfield();
    std::fprintf(stderr,"Device: %s; CFD cells=%d padded=%d block=%d steps=10 RK=3\n",prop.name,p.cells,n,p.block);
    bench_timer_t timer;bench_timer_create(&timer);bench_lap_t times[4];for(auto& t:times)bench_lap_reset(&t);
    const dim3 grid(n/p.block),block(p.block);
    bench_pass_warm();
    for(int iteration=0;iteration<cfd::steps;++iteration) {
        bench_lap_add(&times[0],bench_time_op(&timer,[&]() { copy_variables_kernel<<<grid,block>>>(n,state,old); }));
        bench_lap_add(&times[1],bench_time_op(&timer,[&]() { cfd_device::compute_step_factor_kernel<<<grid,block>>>(n,state,areas,factor); }));
        for(int stage=0;stage<cfd::stages;++stage) {
            bench_lap_add(&times[2],bench_time_op(&timer,[&]() { cfd_device::compute_flux_kernel<<<grid,block>>>(n,neighbors,normals,state,flux); }));
            bench_lap_add(&times[3],bench_time_op(&timer,[&]() { cfd_device::rk_update_kernel<<<grid,block>>>(stage,n,old,state,factor,flux); }));
        }
    }
    bench_timer_destroy(&timer);
    const char* names[]={"copy_variables_kernel","compute_step_factor_kernel","compute_flux_kernel","rk_update_kernel"};
    double total=0;
    for(int k=0;k<4;++k) {
        char params[1200];const int launches=k<2?10:30;
        int written=std::snprintf(params,sizeof(params),
            "\"size\":%d,\"block_size\":%d,\"padded_cells\":%d,\"grid_size\":%d,\"mesh_cube_side\":%d,"
            "\"timesteps\":10,\"rk_stages\":3,\"kernel_launches\":%d,\"cell_updates\":%llu,"
            "\"device_buffer_bytes\":%zu,\"host_input_reference_bytes\":%zu,"
            "\"algorithm\":\"altis_euler3d_rk3_v1\",\"input\":\"tetrahedral_perturbed_flow_v1\","
            "\"boundary\":\"y0_slip_wall_other_farfield\",\"precision\":\"fp32\","
            "\"verification\":\"full_final_aux_cpu_fp64_v1\"",
            p.cells,p.block,n,n/p.block,in.side,launches,static_cast<unsigned long long>(n)*launches,
            device_bytes,54*bytes+std::min<size_t>(5*bytes,4*1024*1024));
        if(written<0||static_cast<size_t>(written)>=sizeof(params))throw std::runtime_error("CFD metadata overflow");
        bench_emit_iter(names[k],0,&times[k],params);total+=times[k].gpu_ms;
    }
    bench_emit_summary(total);
    // No downloads, CPU references or verification launches interrupt the measured algorithm.
    const cfd::Reference expected=cfd::reference(p,in);
    size_t errors=0,printed=0;
    errors+=verify_buffer("state",state,expected.state,printed);
    errors+=verify_buffer("old",old,expected.old,printed);
    errors+=verify_buffer("flux",flux,expected.flux,printed);
    errors+=verify_buffer("factor",factor,expected.factor,printed);
    HIP_CHECK(hipFree(state));HIP_CHECK(hipFree(old));HIP_CHECK(hipFree(flux));HIP_CHECK(hipFree(factor));
    HIP_CHECK(hipFree(areas));HIP_CHECK(hipFree(normals));HIP_CHECK(hipFree(neighbors));
    std::fprintf(stderr,"CFD verification: %s (verified %zu values: complete final state and last old/flux/factor buffers, including padding; %zu mismatches)\n",errors?"FAIL":"PASS",16*static_cast<size_t>(n),errors);
    return errors?1:0;
} catch(const std::exception& e) { std::fprintf(stderr,"FAIL: CFD setup/reference unavailable: %s\n",e.what());return 1; }
