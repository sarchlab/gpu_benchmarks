// Local global-memory microbenchmarks; historical shoc_devicememory identifier.
// Preserve ordinary device malloc and the existing three grid-stride kernels.
#include "../../common/bench_timing.h"
#include "cpu_reference.h"
#include <memory>
#include <vector>
#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

__global__ void readOnly_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    float sum = 0.0f;
    for (int i = gid; i < N; i += totalThreads) {
        sum += input[i];
    }
    if (gid < N) {
        output[gid] = sum;
    }
}

__global__ void writeOnly_kernel(float* __restrict__ output,
                                 int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    for (int i = gid; i < N; i += totalThreads) {
        output[i] = (float)i * 0.001f;
    }
}

__global__ void readWrite_kernel(float* __restrict__ data,
                                 int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    for (int i = gid; i < N; i += totalThreads) {
        data[i] = data[i] * 1.0001f;
    }
}


static size_t verify_result(const char* name,const float* device,size_t count,
                            device_memory::Operation op,const device_memory::ReadReference* read,
                            std::vector<float>& scratch) {
    size_t errors=0;
    for(size_t offset=0;offset<count;) {
        const size_t chunk=std::min(scratch.size(),count-offset);
        HIP_CHECK(hipMemcpy(scratch.data(),device+offset,chunk*sizeof(float),hipMemcpyDeviceToHost));
        errors+=device_memory::compare(name,scratch.data(),offset,chunk,op,read,errors);
        offset+=chunk;
    }
    fprintf(stderr,"%s: %s (verified %zu measured values; %zu mismatches)\n",name,errors?"FAIL":"PASS",count,errors);
    return errors;
}
static void metadata(char* buffer,size_t capacity,const device_memory::Problem& p,
                     const device_memory::Layout& g,device_memory::Operation op) {
    const uint64_t reads=op==device_memory::Operation::write?0:g.elements;
    const uint64_t writes=op==device_memory::Operation::read?g.read_outputs:g.elements;
    const int used=snprintf(buffer,capacity,
        "\"array_size_mb\":%d,\"block_size\":%d,\"transfer_mode\":\"%s\","
        "\"elements\":%d,\"grid_size\":%d,\"launched_threads\":%d,\"output_elements\":%llu,"
        "\"logical_read_bytes\":%llu,\"logical_write_bytes\":%llu,"
        "\"algorithm\":\"local_grid_stride_v1\",\"input\":\"periodic_1000_fp32_v1\","
        "\"precision\":\"fp32\",\"allocation\":\"device_malloc\",\"verification\":\"full_cpu_exact_v1\"",
        p.mib,p.block,p.mode,g.elements,g.blocks,g.threads,
        static_cast<unsigned long long>(writes),static_cast<unsigned long long>(reads*4),
        static_cast<unsigned long long>(writes*4));
    if(used<0 || static_cast<size_t>(used)>=capacity) throw std::runtime_error("metadata overflow");
}
int main(int argc,char** argv) try {
    setvbuf(stdout,nullptr,_IONBF,0);
    const auto p=device_memory::parameters(argc,argv);
    const auto g=device_memory::layout(p);
    const size_t N=g.elements, bytes=g.bytes;
    const int block_size=p.block, num_blocks=g.blocks;
    const bool run_read=strcmp(p.mode,"write")!=0;
    const bool run_write=strcmp(p.mode,"read")!=0;
    const bool run_rw=strcmp(p.mode,"readwrite")==0;
    int device_id=0; HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop; HIP_CHECK(hipGetDeviceProperties(&prop,device_id));
    if(block_size>prop.maxThreadsPerBlock || block_size>prop.maxThreadsDim[0] || num_blocks>prop.maxGridSize[0])
        throw std::invalid_argument("launch geometry exceeds selected device limits");
    fprintf(stderr,"Device: %s; array=%d MiB block=%d mode=%s\n",prop.name,p.mib,block_size,p.mode);

    // Allocate the bounded verifier before any measurement. Production input
    // generation and ordinary device allocations retain the existing recipe.
    std::vector<float> scratch(std::min<size_t>(N,1024*1024));
    std::unique_ptr<device_memory::ReadReference> read_reference;
    if(run_read) read_reference.reset(new device_memory::ReadReference(g.elements,g.threads));
    float *d_input=nullptr,*d_output=nullptr;
    HIP_CHECK(hipMalloc(&d_input,bytes)); HIP_CHECK(hipMalloc(&d_output,bytes));
    {
        std::unique_ptr<float,decltype(&std::free)> h_init(static_cast<float*>(std::malloc(bytes)),&std::free);
        if(!h_init) throw std::bad_alloc();
        for(size_t i=0;i<N;++i) h_init.get()[i]=(float)(i%1000)*0.001f;
        HIP_CHECK(hipMemcpy(d_input,h_init.get(),bytes,hipMemcpyHostToDevice));
    }

    double total_gpu_ms=0; size_t errors=0; char params[768];
    if(run_read) {
        metadata(params,sizeof(params),p,g,device_memory::Operation::read);
        total_gpu_ms+=bench_run_once("readOnly_kernel",params,[&]() {
            readOnly_kernel<<<dim3(num_blocks),dim3(block_size)>>>(d_input,d_output,(int)N);
        });
        // Read sums must be checked before the write kernel overwrites them.
        errors+=verify_result("readOnly_kernel",d_output,g.read_outputs,device_memory::Operation::read,read_reference.get(),scratch);
    }
    if(run_write) {
        metadata(params,sizeof(params),p,g,device_memory::Operation::write);
        total_gpu_ms+=bench_run_once("writeOnly_kernel",params,[&]() {
            writeOnly_kernel<<<dim3(num_blocks),dim3(block_size)>>>(d_output,(int)N);
        });
        errors+=verify_result("writeOnly_kernel",d_output,N,device_memory::Operation::write,nullptr,scratch);
    }
    if(run_rw) {
        {
            std::unique_ptr<float,decltype(&std::free)> h_rw(static_cast<float*>(std::malloc(bytes)),&std::free);
            if(!h_rw) throw std::bad_alloc();
            for(size_t i=0;i<N;++i) h_rw.get()[i]=(float)(i%1000)*0.001f;
            HIP_CHECK(hipMemcpy(d_input,h_rw.get(),bytes,hipMemcpyHostToDevice));
        }
        metadata(params,sizeof(params),p,g,device_memory::Operation::readwrite);
        total_gpu_ms+=bench_run_once("readWrite_kernel",params,[&]() {
            readWrite_kernel<<<dim3(num_blocks),dim3(block_size)>>>(d_input,(int)N);
        });
        errors+=verify_result("readWrite_kernel",d_input,N,device_memory::Operation::readwrite,nullptr,scratch);
    }
    bench_emit_summary(total_gpu_ms);
    HIP_CHECK(hipFree(d_input)); HIP_CHECK(hipFree(d_output));
    fprintf(stderr,"%s: complete measured-output verification; %zu total mismatches\n",errors?"FAIL":"PASS",errors);
    return errors?1:0;
} catch(const std::exception& e) {
    fprintf(stderr,"FAIL: device-memory setup/reference unavailable: %s\n",e.what()); return 1;
}
