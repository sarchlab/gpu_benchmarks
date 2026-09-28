#define main devicememory_production_main
#include "../tier2/shoc_devicememory/shoc_devicememory.hip"
#undef main
#include "fixtures/devicememory_baseline.hip"
int main(int argc,char** argv)try {
    const auto p=device_memory::parameters(argc,argv);const auto g=device_memory::layout(p);
    const size_t n=g.elements,bytes=g.bytes;
    float *input,*a,*b;HIP_CHECK(hipMalloc(&input,bytes));HIP_CHECK(hipMalloc(&a,bytes));HIP_CHECK(hipMalloc(&b,bytes));
    std::vector<float> host(n);
    for(size_t i=0;i<n;++i)host[i]=static_cast<float>(i%1000)*0.001f;
    HIP_CHECK(hipMemcpy(input,host.data(),bytes,hipMemcpyHostToDevice));
    host.clear();host.shrink_to_fit();
    std::vector<float> left(std::min<size_t>(n,1024*1024)),right(left.size());
    size_t verified=0;
    for(int op=0;op<3;++op) {
        if(op==0) {
            readOnly_kernel<<<g.blocks,p.block>>>(input,a,g.elements);
            baseline_readOnly_kernel<<<g.blocks,p.block>>>(input,b,g.elements);
        }else if(op==1) {
            writeOnly_kernel<<<g.blocks,p.block>>>(a,g.elements);
            baseline_writeOnly_kernel<<<g.blocks,p.block>>>(b,g.elements);
        }else {
            HIP_CHECK(hipMemcpy(a,input,bytes,hipMemcpyDeviceToDevice));
            HIP_CHECK(hipMemcpy(b,input,bytes,hipMemcpyDeviceToDevice));
            readWrite_kernel<<<g.blocks,p.block>>>(a,g.elements);
            baseline_readWrite_kernel<<<g.blocks,p.block>>>(b,g.elements);
        }
        HIP_CHECK(hipGetLastError());
        const size_t count=op==0?g.read_outputs:n;
        for(size_t offset=0;offset<count;) {
            const size_t chunk=std::min(left.size(),count-offset);
            HIP_CHECK(hipMemcpy(left.data(),a+offset,chunk*sizeof(float),hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(right.data(),b+offset,chunk*sizeof(float),hipMemcpyDeviceToHost));
            for(size_t i=0;i<chunk;++i)if(!device_memory::equal_finite(left[i],right[i]))throw std::runtime_error("baseline output differs");
            verified+=chunk;offset+=chunk;
        }
    }
    HIP_CHECK(hipFree(input));HIP_CHECK(hipFree(a));HIP_CHECK(hipFree(b));
    printf("PASS: %zu complete output values exactly match frozen local kernels\n",verified);return 0;
}catch(const std::exception& e){fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
