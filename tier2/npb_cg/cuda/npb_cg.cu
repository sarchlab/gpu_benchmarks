// NAS CG inverse-power benchmark, standard classes S/W/A/B/C.
// Algorithm pin: GMAP/NPB-GPU 3f12d84920ee315ab00ef283717c1e74b68f4d00,
// CUDA/CG/cg.cu. The NAS problem and CPU reference are in nas_cg.h.
#include "../../../common_cuda/bench_timing.h"
#include "nas_cg.h"
#include <cerrno>
#include <climits>
#include <cstring>
#include <cstdlib>

#define CUDA_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

__global__ void cg_init_kernel(const double* x, double* z, double* p, double* q, double* r, int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) { z[i]=0.0; q[i]=0.0; r[i]=x[i]; p[i]=x[i]; }
}
__global__ void spmv_kernel(const int* offsets, const int* columns, const double* values,
                            const double* x, double* y, int n) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<n) {
        double sum=0.0;
        for(int k=offsets[row];k<offsets[row+1];++k) sum+=values[k]*x[columns[k]];
        y[row]=sum;
    }
}
__global__ void dot_product_kernel(const double* a, const double* b, double* partial, int n, bool difference) {
    extern __shared__ double scratch[];
    const int lane=threadIdx.x, i=blockIdx.x*blockDim.x+lane;
    double value=0.0;
    if(i<n) {
        if(difference) { const double d=a[i]-b[i]; value=d*d; }
        else value=a[i]*b[i];
    }
    scratch[lane]=value;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride/=2) {
        if(lane<stride) scratch[lane]+=scratch[lane+stride];
        __syncthreads();
    }
    if(lane==0) partial[blockIdx.x]=scratch[0];
}
__global__ void axpy_kernel(double alpha, const double* p, const double* q, double* z, double* r, int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) { z[i]+=alpha*p[i]; r[i]-=alpha*q[i]; }
}
__global__ void direction_kernel(double beta, const double* r, double* p, int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) p[i]=r[i]+beta*p[i];
}
__global__ void normalize_kernel(double inverse_norm, const double* z, double* x, int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) x[i]=inverse_norm*z[i];
}

// Host parameter admission. The legacy --size option retains its precedence
// over BENCH_PARAM_matrix_size; both now select a complete standard NAS class.
static int integer(const char* value) {
    if(!value || !*value) throw std::invalid_argument("empty CG integer parameter");
    char* end=nullptr; errno=0;
    const long parsed=std::strtol(value,&end,10);
    if(errno || *end || parsed<INT_MIN || parsed>INT_MAX)
        throw std::invalid_argument("invalid CG integer parameter");
    return static_cast<int>(parsed);
}
static const nas_cg::Class* parameters(int argc, char** argv, int* block) {
    int n=14000;
    *block=256;
    if(const char* value=std::getenv("BENCH_PARAM_matrix_size")) n=integer(value);
    if(const char* value=std::getenv("BENCH_PARAM_block_size")) *block=integer(value);
    for(int i=1;i<argc;++i) {
        if(std::strcmp(argv[i],"--size")!=0 || i+1==argc)
            throw std::invalid_argument("expected --size followed by a NAS class size");
        n=integer(argv[++i]);
    }
    if(std::getenv("BENCH_PARAM_max_cg_iterations") || std::getenv("BENCH_PARAM_nonzeros_per_row"))
        throw std::invalid_argument("obsolete CG controls: the selected NAS class fixes sparsity and iterations");
    const auto* c=nas_cg::problem(n);
    if(!c) throw std::invalid_argument("CG matrix_size must select NAS S/W/A/B/C: 1400,7000,14000,75000,150000");
    if(*block<=0 || *block>1024 || (*block&(*block-1))!=0)
        throw std::invalid_argument("CG block_size must be a power of two in 1..1024");
    return c;
}
static double ratio(double numerator, double denominator) {
    const double value=numerator/denominator;
    if(!std::isfinite(value)) throw std::runtime_error("non-finite CG scalar or zero denominator");
    return value;
}

int main(int argc, char** argv) try {
    setvbuf(stdout,nullptr,_IONBF,0);
    int block;
    const nas_cg::Class c=*parameters(argc,argv,&block);
    const int n=c.n, blocks=(n+block-1)/block;
    const size_t vector_bytes=static_cast<size_t>(n)*sizeof(double);
    const unsigned workers=nas_cg::reference_workers(n);
    int device=0; CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop,device));
    if(block>prop.maxThreadsPerBlock || block>prop.maxThreadsDim[0] || blocks>prop.maxGridSize[0] ||
       static_cast<size_t>(block)*sizeof(double)>prop.sharedMemPerBlock)
        throw std::invalid_argument("CG launch geometry exceeds device limits");
    fprintf(stderr,"Device: %s (id=%d)\nNAS CG class %c: N=%d NONZER=%d NITER=%d SHIFT=%.0f, 25 inner iterations\n",
            prop.name,device,c.name,n,c.nonzer,c.iterations,c.shift);
    const nas_cg::Matrix matrix=nas_cg::make_matrix(c);
    nas_cg::Result actual(c);
    std::vector<double> partial(blocks);
    int *d_offsets,*d_columns;
    double *d_values,*d_x,*d_z,*d_p,*d_q,*d_r,*d_partial;
    CUDA_CHECK(cudaMalloc(&d_offsets,matrix.offsets.size()*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_columns,matrix.columns.size()*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values,matrix.values.size()*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x,vector_bytes)); CUDA_CHECK(cudaMalloc(&d_z,vector_bytes));
    CUDA_CHECK(cudaMalloc(&d_p,vector_bytes)); CUDA_CHECK(cudaMalloc(&d_q,vector_bytes));
    CUDA_CHECK(cudaMalloc(&d_r,vector_bytes)); CUDA_CHECK(cudaMalloc(&d_partial,partial.size()*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_offsets,matrix.offsets.data(),matrix.offsets.size()*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_columns,matrix.columns.data(),matrix.columns.size()*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values,matrix.values.data(),matrix.values.size()*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x,actual.x.data(),vector_bytes,cudaMemcpyHostToDevice));

    enum { INIT, SPMV, DOT, AXPY, DIRECTION, NORMALIZE, KERNELS };
    const char* names[KERNELS]={"cg_init_kernel","spmv_kernel","dot_product_kernel","axpy_kernel","direction_kernel","normalize_kernel"};
    bench_lap_t laps[KERNELS];
    int launches[KERNELS]={};
    for(auto& lap:laps) bench_lap_reset(&lap);
    bench_timer_t timer; bench_timer_create(&timer);
    auto timed=[&](int kind, auto operation) {
        bench_lap_add(&laps[kind],bench_time_op(&timer,operation)); ++launches[kind];
    };
    auto dot=[&](const double* a,const double* b,bool difference=false) {
        timed(DOT,[&]() { dot_product_kernel<<<blocks,block,block*sizeof(double)>>>(a,b,d_partial,n,difference); });
        CUDA_CHECK(cudaMemcpy(partial.data(),d_partial,partial.size()*sizeof(double),cudaMemcpyDeviceToHost));
        double sum=0.0; for(double value:partial) sum+=value;
        return sum;
    };
    auto spmv=[&](const double* x,double* y) {
        timed(SPMV,[&]() { spmv_kernel<<<blocks,block>>>(d_offsets,d_columns,d_values,x,y,n); });
    };
    bench_pass_warm();
    for(int outer=0;outer<c.iterations;++outer) {
        timed(INIT,[&]() { cg_init_kernel<<<blocks,block>>>(d_x,d_z,d_p,d_q,d_r,n); });
        double rho=dot(d_r,d_r);
        for(int inner=0;inner<nas_cg::inner_iterations;++inner) {
            spmv(d_p,d_q);
            const double alpha=ratio(rho,dot(d_p,d_q)), previous=rho;
            timed(AXPY,[&]() { axpy_kernel<<<blocks,block>>>(alpha,d_p,d_q,d_z,d_r,n); });
            rho=dot(d_r,d_r);
            const double beta=ratio(rho,previous);
            timed(DIRECTION,[&]() { direction_kernel<<<blocks,block>>>(beta,d_r,d_p,n); });
        }
        spmv(d_z,d_r);
        actual.residual[outer]=std::sqrt(dot(d_x,d_r,true));
        actual.zeta[outer]=c.shift+ratio(1.0,dot(d_x,d_z));
        const double inverse_norm=ratio(1.0,std::sqrt(dot(d_z,d_z)));
        timed(NORMALIZE,[&]() { normalize_kernel<<<blocks,block>>>(inverse_norm,d_z,d_x,n); });
    }
    double total=0.0;
    for(int kind=0;kind<KERNELS;++kind) {
        char params[768];
        snprintf(params,sizeof(params),
            "\"matrix_size\":%d,\"block_size\":%d,\"num_blocks\":%d,\"actual_threads\":%d,"
            "\"npb_class\":\"%c\",\"npb_nonzer\":%d,\"outer_iterations\":%d,\"inner_iterations\":25,"
            "\"shift\":%.0f,\"rcond\":0.1,\"matrix_nonzeros\":%zu,\"kernel_launches\":%d,"
            "\"spmv_nonzero_products\":%llu,\"input_seed\":314159265,\"input_multiplier\":1220703125,"
            "\"reference_workers\":%u,\"precision\":\"fp64\",\"algorithm\":\"npb_cg_inverse_power_v1\","
            "\"verification\":\"full_cpu_nas_zeta_v1\"",
            n,block,blocks,blocks*block,c.name,c.nonzer,c.iterations,c.shift,matrix.values.size(),launches[kind],
            kind==SPMV ? static_cast<unsigned long long>(matrix.values.size())*launches[kind] : 0ULL,workers);
        bench_emit_iter(names[kind],0,&laps[kind],params); total+=laps[kind].gpu_ms;
    }
    bench_emit_summary(total); bench_timer_destroy(&timer);
    fprintf(stderr,"Total GPU time: %.3f ms\n",total);

    // Download every actual final vector after the complete measured algorithm.
    CUDA_CHECK(cudaMemcpy(actual.x.data(),d_x,vector_bytes,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(actual.z.data(),d_z,vector_bytes,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(actual.p.data(),d_p,vector_bytes,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(actual.q.data(),d_q,vector_bytes,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(actual.r.data(),d_r,vector_bytes,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_offsets)); CUDA_CHECK(cudaFree(d_columns)); CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_z)); CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_q)); CUDA_CHECK(cudaFree(d_r)); CUDA_CHECK(cudaFree(d_partial));
    const nas_cg::Result expected=nas_cg::reference(matrix,c,workers);
    const size_t errors=nas_cg::compare(actual,expected);
    const bool nas_ok=nas_cg::published_zeta(actual.zeta.back(),c);
    fprintf(stderr,"NAS class %c zeta: GPU=%.17g expected=%.17g relative_error=%.9g (%s)\n",
        c.name,actual.zeta.back(),c.zeta,std::fabs(actual.zeta.back()-c.zeta)/c.zeta,nas_ok?"PASS":"FAIL");
    const bool ok=errors==0 && nas_ok;
    fprintf(stderr,"%s (verified %zu vector values and %d outer scalars, %zu mismatches, %u CPU workers)\n",
        ok?"PASS":"FAIL",static_cast<size_t>(5)*n,2*c.iterations,errors,workers);
    return ok?EXIT_SUCCESS:EXIT_FAILURE;
} catch(const std::exception& error) {
    fprintf(stderr,"FAIL: CG setup/reference/arithmetic unavailable: %s\n",error.what());
    return EXIT_FAILURE;
}
