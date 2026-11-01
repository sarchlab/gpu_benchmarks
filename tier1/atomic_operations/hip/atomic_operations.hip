// Single-counter, fixed-attempt atomic microbenchmark. See README.md.
#include "../../../common_hip/bench_timing.h"
#include "atomic_reference.h"
#define HIP_CHECK(call) BENCH_TIMING_CHECK(call)
static_assert(sizeof(unsigned int)==sizeof(uint32_t),"32-bit unsigned atomic counter required");

__global__ void atomicAdd_kernel(unsigned int* counter,int ops) {
    for(int i=0;i<ops;++i)atomicAdd(counter,1u);
}
__global__ void atomicCAS_kernel(unsigned int* counter,int ops,unsigned int* successes) {
    unsigned int count=0;
    for(int i=0;i<ops;++i) {
        // A value-preserving atomic read; never mix ordinary loads with writers.
        const unsigned int old=atomicAdd(counter,0u);
        count+=atomicCAS(counter,old,old+1u)==old;
    }
    successes[static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x]=count;
}
__global__ void atomicMin_kernel(unsigned int* counter,int ops) {
    const unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x;
    for(int i=0;i<ops;++i)atomicMin(counter,tid+static_cast<unsigned int>(i));
}

int main(int argc,char** argv)try {
    setvbuf(stdout,nullptr,_IOLBF,0);
    const atomic_bench::Problem p=atomic_bench::parameters(argc,argv);
    const bool is_cas=atomic_bench::cas(p);
    const uint64_t num_threads=atomic_bench::threads(p),attempts=atomic_bench::attempts(p);
    int device=0;HIP_CHECK(hipGetDevice(&device));hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop,device));
    if(p.block>prop.maxThreadsPerBlock || p.block>prop.maxThreadsDim[0] || p.blocks>prop.maxGridSize[0])
        throw std::invalid_argument("atomic launch geometry exceeds device limits");
    fprintf(stderr,"Device: %s; %s; %llu threads; %llu fixed attempts\n",prop.name,p.operation,
            static_cast<unsigned long long>(num_threads),static_cast<unsigned long long>(attempts));
    std::vector<uint32_t> successes(is_cas?num_threads:0);
    unsigned int* d_counter=nullptr;unsigned int* d_successes=nullptr;
    HIP_CHECK(hipMalloc(&d_counter,sizeof(unsigned int)));
    if(is_cas)HIP_CHECK(hipMalloc(&d_successes,num_threads*sizeof(unsigned int)));
    if(is_cas)HIP_CHECK(hipMemset(d_successes,0xff,num_threads*sizeof(unsigned int)));
    unsigned int counter=0;
    HIP_CHECK(hipMemcpy(d_counter,&counter,sizeof(counter),hipMemcpyHostToDevice));
    // No copy or verification lies between the entry warmer and measured launch.
    char params[1000];
    const int written=snprintf(params,sizeof(params),
        "\"ops_per_thread\":%d,\"num_blocks\":%d,\"threads_per_block\":%d,\"atomic_op\":\"%s\","
        "\"num_threads\":%llu,\"attempted_operations\":%llu,\"atomic_read_operations\":%llu,"
        "\"total_atomic_operations\":%llu,\"success_count_stores\":%llu,\"counter_bits\":32,"
        "\"initial_counter\":0,\"workload\":\"fixed_attempts_v1\",\"verification\":\"full_counter_and_cas_counts_v1\"",
        p.ops,p.blocks,p.block,p.operation,static_cast<unsigned long long>(num_threads),
        static_cast<unsigned long long>(attempts),static_cast<unsigned long long>(is_cas?attempts:0),
        static_cast<unsigned long long>(attempts*(is_cas?2:1)),static_cast<unsigned long long>(is_cas?num_threads:0));
    if(written<0 || static_cast<size_t>(written)>=sizeof(params))throw std::runtime_error("atomic metadata overflow");
    const char* name=is_cas?"atomicCAS_kernel":std::strcmp(p.operation,"atomicAdd")==0?"atomicAdd_kernel":"atomicMin_kernel";
    const double elapsed=bench_run_once(name,params,[&]() {
        if(is_cas)atomicCAS_kernel<<<p.blocks,p.block>>>(d_counter,p.ops,d_successes);
        else if(std::strcmp(p.operation,"atomicAdd")==0)atomicAdd_kernel<<<p.blocks,p.block>>>(d_counter,p.ops);
        else atomicMin_kernel<<<p.blocks,p.block>>>(d_counter,p.ops);
    });
    bench_emit_summary(elapsed);
    // Verify the actual measured counter and every thread's CAS success count.
    HIP_CHECK(hipMemcpy(&counter,d_counter,sizeof(counter),hipMemcpyDeviceToHost));
    if(is_cas)HIP_CHECK(hipMemcpy(successes.data(),d_successes,num_threads*sizeof(unsigned int),hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_counter));if(is_cas)HIP_CHECK(hipFree(d_successes));
    const atomic_bench::Verification result=atomic_bench::verify(p,counter,successes);
    fprintf(stderr,"%s (counter=%u; verified %zu CAS counts; successful CAS=%llu; %zu mismatches)\n",
            result.errors?"FAIL":"PASS",counter,successes.size(),
            static_cast<unsigned long long>(result.successes),result.errors);
    return result.errors?1:0;
} catch(const std::exception& e) {
    fprintf(stderr,"FAIL: atomic setup/verification unavailable: %s\n",e.what());return 1;
}
