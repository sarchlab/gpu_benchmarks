// NAS MG S/W/A/B/C. See README.md for the exact upstream and timing contract.
#include "../../../common/bench_timing.h"
#include "mg_reference.h"
#include "mg_kernels.h"
#include <cerrno>
#include <climits>
#include <cstring>
#include <cstdlib>
#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

static int integer(const char* value) {
    if(!value || !*value)throw std::invalid_argument("empty MG integer parameter");
    char* end=nullptr;errno=0;const long n=std::strtol(value,&end,10);
    if(errno || *end || n<INT_MIN || n>INT_MAX)throw std::invalid_argument("invalid MG integer parameter");
    return static_cast<int>(n);
}
static const nas_mg::Class* parameters(int argc,char** argv,int* block) {
    const char* name=std::getenv("BENCH_PARAM_npb_class");if(!name)name="A";
    *block=256;
    if(const char* value=std::getenv("BENCH_PARAM_block_size"))*block=integer(value);
    for(int i=1;i<argc;++i) {
        if(std::strcmp(argv[i],"--class")!=0 || i+1==argc)
            throw std::invalid_argument("expected --class S/W/A/B/C; obsolete --size is ambiguous for A/B");
        name=argv[++i];
    }
    for(const char* key:{"BENCH_PARAM_grid_size","BENCH_PARAM_num_vcycles","BENCH_PARAM_num_smooth_steps"})
        if(std::getenv(key))throw std::invalid_argument("obsolete MG controls: npb_class fixes grid, cycles and smoothing");
    if(std::strlen(name)!=1 || !nas_mg::problem(name[0]))throw std::invalid_argument("MG npb_class must be S/W/A/B/C");
    if(*block<1 || *block>1024 || (*block&(*block-1)))throw std::invalid_argument("MG block_size must be a power of two in 1..1024");
    return nas_mg::problem(name[0]);
}

int main(int argc,char** argv) try {
    setvbuf(stdout,nullptr,_IONBF,0);
    int block;const nas_mg::Class c=*parameters(argc,argv,&block);
    const nas_mg::Layout layout(c);
    const int top=layout.levels()-1,n=c.n+2;
    const unsigned workers=nas_mg::reference_workers(c.n);
    int device;HIP_CHECK(hipGetDevice(&device));hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop,device));
    // Restriction uses coarse_width-1 threads and interpolation coarse_width.
    const int required_block=std::max(block,c.n/2+2);
    const size_t shared=std::max(3*mg_max_width,2*block)*sizeof(double);
    const size_t max_grid=std::max(nas_mg::Layout::cube(c.n/2+2)/block+1,static_cast<size_t>(n));
    if(required_block>prop.maxThreadsPerBlock || required_block>prop.maxThreadsDim[0] ||
       max_grid>static_cast<size_t>(prop.maxGridSize[0]) || n>prop.maxGridSize[1] || shared>prop.sharedMemPerBlock)
        throw std::invalid_argument("MG launch geometry exceeds device limits");
    fprintf(stderr,"Device: %s (id=%d)\nNAS MG class %c: interior %d cubed, %d levels, %d cycles\n",prop.name,device,c.name,c.n,layout.levels(),c.iterations);
    nas_mg::Result actual(layout);actual.v=nas_mg::input(c);
    double *d_u,*d_r,*d_v,*d_a,*d_c,*d_norm;
    double a[]={-8.0/3,0.0,1.0/6,1.0/12};auto coefficients=nas_mg::smoother(c);
    const size_t hierarchy_bytes=layout.cells*sizeof(double),v_bytes=actual.v.size()*sizeof(double);
    std::vector<double> partial(static_cast<size_t>(2)*c.n*c.n);
    HIP_CHECK(hipMalloc(&d_u,hierarchy_bytes));HIP_CHECK(hipMalloc(&d_r,hierarchy_bytes));
    HIP_CHECK(hipMalloc(&d_v,v_bytes));HIP_CHECK(hipMalloc(&d_a,sizeof(a)));HIP_CHECK(hipMalloc(&d_c,sizeof(a)));
    HIP_CHECK(hipMalloc(&d_norm,partial.size()*sizeof(double)));
    HIP_CHECK(hipMemset(d_u,0,hierarchy_bytes));HIP_CHECK(hipMemset(d_r,0,hierarchy_bytes));
    HIP_CHECK(hipMemcpy(d_v,actual.v.data(),v_bytes,hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_a,a,sizeof(a),hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(d_c,coefficients.data(),sizeof(a),hipMemcpyHostToDevice));
    auto up=[&](int k){return d_u+layout.offset[k];};auto rp=[&](int k){return d_r+layout.offset[k];};
    enum { COMM1,COMM2,COMM3,INTERP,NORM,SMOOTH,RESID,RESTRICT,ZERO,KERNELS };
    const char* names[]={"comm3_gpu_kernel_1","comm3_gpu_kernel_2","comm3_gpu_kernel_3","interp_gpu_kernel","norm2u3_gpu_kernel","psinv_gpu_kernel","resid_gpu_kernel","rprj3_gpu_kernel","coarse_zero"};
    bench_lap_t laps[KERNELS];for(auto& lap:laps)bench_lap_reset(&lap);
    unsigned launches[KERNELS]={};unsigned long long threads[KERNELS]={};
    unsigned min_block[KERNELS]={},max_block[KERNELS]={};
    bench_timer_t timer;bench_timer_create(&timer);
    auto timed=[&](int kind,dim3 grid,int bs,auto op) {
        bench_lap_add(&laps[kind],bench_time_op(&timer,op));
        if(!launches[kind])min_block[kind]=bs;
        ++launches[kind];threads[kind]+=static_cast<unsigned long long>(grid.x)*grid.y*grid.z*bs;
        min_block[kind]=std::min(min_block[kind],static_cast<unsigned>(bs));
        max_block[kind]=std::max(max_block[kind],static_cast<unsigned>(bs));
    };
    auto comm=[&](double* data,int p) {
        dim3 g1((p-2+block-1)/block,p-2),g2((p+block-1)/block,p-2),g3(g2.x,p);
        timed(COMM1,g1,block,[&](){comm3_gpu_kernel_1<<<g1,block>>>(data,p,p,p,g1.x*block);});
        timed(COMM2,g2,block,[&](){comm3_gpu_kernel_2<<<g2,block>>>(data,p,p,p,g2.x*block);});
        timed(COMM3,g3,block,[&](){comm3_gpu_kernel_3<<<g3,block>>>(data,p,p,p,g3.x*block);});
    };
    auto resid=[&](int k,double* rhs) {
        const int p=layout.width[k],bs=std::min(p,block);dim3 grid(p-2,p-2);
        timed(RESID,grid,bs,[&](){resid_gpu_kernel<<<grid,bs,2*mg_max_width*sizeof(double)>>>(up(k),rhs,rp(k),d_a,p,p,p,(p-2)*bs);});
        comm(rp(k),p);
    };
    auto smooth=[&](int k) {
        const int p=layout.width[k],bs=std::min(p,block);dim3 grid(p-2,p-2);
        timed(SMOOTH,grid,bs,[&](){psinv_gpu_kernel<<<grid,bs,2*mg_max_width*sizeof(double)>>>(rp(k),up(k),d_c,p,p,p,(p-2)*bs);});
        comm(up(k),p);
    };
    auto restrict_level=[&](int k) {
        const int f=layout.width[k],p=layout.width[k-1],bs=p-1;dim3 grid(p-2,p-2);
        timed(RESTRICT,grid,bs,[&](){rprj3_gpu_kernel<<<grid,bs,2*mg_max_width*sizeof(double)>>>(rp(k),rp(k-1),f,f,f,p,p,p,1,1,1,(p-2)*bs);});
        comm(rp(k-1),p);
    };
    auto interpolate=[&](int k) {
        const int p=layout.width[k],q=layout.width[k-1];dim3 grid(q-1,q-1);
        timed(INTERP,grid,q,[&](){interp_gpu_kernel<<<grid,q,3*mg_max_width*sizeof(double)>>>(up(k-1),up(k),q,q,q,p,p,p,(q-1)*q);});
    };
    auto zero=[&](int k) {
        const int count=static_cast<int>(nas_mg::Layout::cube(layout.width[k]));dim3 grid((count+block-1)/block);
        timed(ZERO,grid,block,[&](){coarse_zero<<<grid,block>>>(up(k),count);});
    };
    auto norm=[&](int slot) {
        dim3 grid(c.n,c.n);const size_t count=static_cast<size_t>(c.n)*c.n;
        timed(NORM,grid,block,[&](){norm2u3_gpu_kernel<<<grid,block,2*block*sizeof(double)>>>(rp(top),n,n,n,d_norm,d_norm+count,c.n,c.n*block);});
        HIP_CHECK(hipMemcpy(partial.data(),d_norm,partial.size()*sizeof(double),hipMemcpyDeviceToHost));
        double sum=0.0,maximum=0.0;
        for(size_t i=0;i<count;++i) {
            if(!std::isfinite(partial[i]) || !std::isfinite(partial[count+i]))throw std::runtime_error("non-finite MG norm partial");
            sum+=partial[i];maximum=std::max(maximum,partial[count+i]);
        }
        actual.norms[slot]=std::sqrt(sum/(static_cast<double>(c.n)*c.n*c.n));actual.norms[slot+1]=maximum;
    };
    bench_pass_warm();
    resid(top,d_v);norm(0);
    for(int it=0;it<c.iterations;++it) {
        for(int k=top;k>0;--k)restrict_level(k);
        zero(0);smooth(0);
        for(int k=1;k<top;++k){zero(k);interpolate(k);resid(k,rp(k));smooth(k);}
        interpolate(top);resid(top,d_v);smooth(top);resid(top,d_v);
    }
    norm(2);
    double total=0.0;
    for(int k=0;k<KERNELS;++k) {
        char params[1024];snprintf(params,sizeof(params),
            "\"npb_class\":\"%c\",\"grid_size\":%d,\"padded_grid_size\":%d,\"levels\":%d,\"num_vcycles\":%d,"
            "\"block_size\":%d,\"min_block_threads\":%u,\"max_block_threads\":%u,\"kernel_launches\":%u,\"launched_threads\":%llu,"
            "\"hierarchy_cells\":%zu,\"input_seed\":314159265,\"input_multiplier\":1220703125,\"reference_workers\":%u,"
            "\"precision\":\"fp64\",\"boundary\":\"periodic\",\"algorithm\":\"npb_mg_vcycle_v1\",\"verification\":\"full_cpu_nas_l2_v1\"",
            c.name,c.n,n,layout.levels(),c.iterations,block,min_block[k],max_block[k],launches[k],threads[k],layout.cells,workers);
        bench_emit_iter(names[k],0,&laps[k],params);total+=laps[k].gpu_ms;
    }
    bench_emit_summary(total);bench_timer_destroy(&timer);
    // Download the entire measured hierarchy, including every periodic ghost.
    HIP_CHECK(hipMemcpy(actual.u.data(),d_u,hierarchy_bytes,hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(actual.r.data(),d_r,hierarchy_bytes,hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(actual.v.data(),d_v,v_bytes,hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_u));HIP_CHECK(hipFree(d_r));HIP_CHECK(hipFree(d_v));
    HIP_CHECK(hipFree(d_a));HIP_CHECK(hipFree(d_c));HIP_CHECK(hipFree(d_norm));
    const nas_mg::Result expected=nas_mg::reference(c,layout,workers);
    const size_t errors=nas_mg::compare(actual,expected);
    const bool nas_ok=nas_mg::published_l2(actual.norms[2],c);
    fprintf(stderr,"NAS class %c L2: GPU=%.17g expected=%.17g relative_error=%.9g (%s)\n",c.name,actual.norms[2],c.l2,std::fabs(actual.norms[2]-c.l2)/c.l2,nas_ok?"PASS":"FAIL");
    const bool ok=errors==0 && nas_ok;
    fprintf(stderr,"%s (verified %zu hierarchy/input values and 4 norms, %zu mismatches, %u CPU workers)\n",ok?"PASS":"FAIL",2*layout.cells+actual.v.size(),errors,workers);
    return ok?EXIT_SUCCESS:EXIT_FAILURE;
} catch(const std::exception& error) {
    fprintf(stderr,"FAIL: MG setup/reference/arithmetic unavailable: %s\n",error.what());return EXIT_FAILURE;
}
