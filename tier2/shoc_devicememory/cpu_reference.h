#pragma once
// CPU-only admission and exact references for the owner-selected local workload.
#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace device_memory {
enum class Operation { read, write, readwrite };
struct Problem { int mib=64, block=256; const char* mode="readwrite"; };
struct Layout { int elements, blocks, threads, read_outputs; size_t bytes; };
inline int positive_integer(const char* text) {
    if(!text || !*text) throw std::invalid_argument("empty integer parameter");
    errno=0; char* end=nullptr; const long value=std::strtol(text,&end,10);
    if(errno || *end || value<1 || value>INT_MAX)
        throw std::invalid_argument("expected positive 32-bit integer parameter");
    return static_cast<int>(value);
}
inline const char* mode(const char* text) {
    for(const char* known : {"read","write","readwrite"})
        if(std::strcmp(text,known)==0) return known;
    throw std::invalid_argument("transfer_mode must be read, write or readwrite");
}
inline Layout layout(const Problem& p) {
    if(p.mib<1 || p.block<1) throw std::invalid_argument("size and block must be positive");
    const uint64_t n=static_cast<uint64_t>(p.mib)*262144;
    const uint64_t blocks=std::min<uint64_t>((n+p.block-1)/p.block,65535);
    const uint64_t threads=blocks*p.block;
    // The unchanged signed-int kernel increments i even after its last access.
    if(n>INT_MAX || threads>INT_MAX || n-1+threads>INT_MAX)
        throw std::invalid_argument("grid-stride loop can exceed signed 32-bit range");
    if(n>SIZE_MAX/sizeof(float)) throw std::invalid_argument("array byte size overflows size_t");
    return {static_cast<int>(n),static_cast<int>(blocks),static_cast<int>(threads),
            static_cast<int>(std::min(n,threads)),static_cast<size_t>(n)*sizeof(float)};
}
inline Problem parameters(int argc,char** argv) {
    const char* names[]={"array_size_mb","block_size","transfer_mode"};
    const char* values[3]={nullptr,nullptr,nullptr};
    for(int i=1;i<argc;++i) {
        int key=0;
        for(;key<3;++key)
            if(std::strncmp(argv[i],"--",2)==0 && std::strcmp(argv[i]+2,names[key])==0) break;
        if(key==3 || i+1==argc) throw std::invalid_argument("unknown flag or missing parameter value");
        const char* value=argv[++i];
        if(key==2) mode(value); else positive_integer(value);
        if(!values[key]) values[key]=value;
    }
    for(int key=0;key<3;++key) {
        char name[64]; std::snprintf(name,sizeof(name),"BENCH_PARAM_%s",names[key]);
        if(const char* value=std::getenv(name)) values[key]=value;
    }
    Problem p;
    if(values[0]) p.mib=positive_integer(values[0]);
    if(values[1]) p.block=positive_integer(values[1]);
    if(values[2]) p.mode=mode(values[2]);
    layout(p); return p;
}
inline float input_value(size_t index) { return static_cast<float>(index%1000)*0.001f; }

// Input repeats every 1000 elements. For a grid stride S, a thread visits
// residues gid, gid+S, ... modulo 1000, with q or q+1 additions. Preserve the
// FP32 addition order; cache those 2000 sums, then check EVERY produced output.
// This is an exact reference reduction for this fixed input, not sampling.
class ReadReference {
    std::array<float,1000> shorter{},longer{};
    int remainder;
public:
    ReadReference(int n,int stride):remainder(0) {
        if(n<1 || stride<1) throw std::invalid_argument("invalid read reference geometry");
        const int q=n/stride, step=stride%1000; remainder=n%stride;
        for(int residue=0;residue<1000;++residue) {
            float sum=0; int position=residue;
            for(int j=0;j<q;++j) {
                sum+=input_value(position);
                position=(position+step)%1000;
            }
            shorter[residue]=sum;
            longer[residue]=sum+input_value(position);
        }
    }
    float operator()(size_t gid) const {
        return gid<static_cast<size_t>(remainder)?longer[gid%1000]:shorter[gid%1000];
    }
};
inline float expected_value(Operation op,size_t index,const ReadReference* read) {
    if(op==Operation::read) {
        if(!read) throw std::invalid_argument("missing read reference");
        return (*read)(index);
    }
    if(op==Operation::write) return static_cast<float>(index)*0.001f;
    return input_value(index)*1.0001f;
}
inline bool equal_finite(float actual,float expected) {
    return std::isfinite(actual) && std::isfinite(expected) && actual==expected;
}
inline size_t compare(const char* name,const float* actual,size_t offset,size_t count,
                      Operation op,const ReadReference* read,size_t errors_before=0) {
    size_t errors=0;
    for(size_t j=0;j<count;++j) {
        const size_t i=offset+j;
        const float expected=expected_value(op,i,read);
        if(!equal_finite(actual[j],expected)) {
            if(errors_before+errors<10)
                std::fprintf(stderr,"Mismatch %s[%zu]: GPU=%.9g CPU=%.9g\n",name,i,actual[j],expected);
            ++errors;
        }
    }
    return errors;
}
} // namespace device_memory
