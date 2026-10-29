#pragma once
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace atomic_bench {
struct Problem { int ops=1024,blocks=2048,block=256;const char* operation="atomicAdd"; };
inline int positive(const char* text) {
    if(!text || !*text)throw std::invalid_argument("empty atomic integer parameter");
    char* end=nullptr;errno=0;const long value=std::strtol(text,&end,10);
    if(errno || *end || value<1 || value>INT_MAX)
        throw std::invalid_argument("expected a positive 32-bit atomic integer parameter");
    return static_cast<int>(value);
}
inline uint64_t threads(const Problem& p) {return static_cast<uint64_t>(p.blocks)*p.block;}
inline uint64_t attempts(const Problem& p) {return threads(p)*p.ops;}
inline bool cas(const Problem& p) {return std::strcmp(p.operation,"atomicCAS")==0;}
inline Problem parameters(int argc,char** argv) {
    Problem p;
    if(argc>2)throw std::invalid_argument("usage: atomic_operations [ops_per_thread]");
    // Environment retains precedence over the optional positional argument.
    const char* v=std::getenv("BENCH_PARAM_ops_per_thread");
    if(v)p.ops=positive(v);else if(argc==2)p.ops=positive(argv[1]);
    if((v=std::getenv("BENCH_PARAM_num_blocks")))p.blocks=positive(v);
    if((v=std::getenv("BENCH_PARAM_threads_per_block")))p.block=positive(v);
    if((v=std::getenv("BENCH_PARAM_atomic_op")))p.operation=v;
    if(std::strcmp(p.operation,"atomicAdd") && std::strcmp(p.operation,"atomicMin") && !cas(p))
        throw std::invalid_argument("unsupported atomic_op");
    if(p.block>1024 || threads(p)>UINT32_MAX)
        throw std::invalid_argument("atomic launch exceeds 1024 threads/block or 32-bit thread indices");
    return p;
}
struct Verification { uint64_t successes=0;size_t errors=0; };
inline Verification verify(const Problem& p,uint32_t counter,const std::vector<uint32_t>& counts) {
    Verification result;
    uint32_t expected=0;
    if(cas(p)) {
        if(counts.size()!=threads(p))throw std::invalid_argument("CAS success array size mismatch");
        for(size_t i=0;i<counts.size();++i) {
            if(counts[i]>static_cast<uint32_t>(p.ops)) {
                if(result.errors<10)std::fprintf(stderr,"Invalid CAS successes[%zu]=%u\n",i,counts[i]);
                ++result.errors;
            }
            result.successes+=counts[i];
        }
        // One thread's ordered attempts each either update the counter or span
        // another successful update. Hence at least ops successes occur overall.
        if(result.successes<static_cast<uint64_t>(p.ops) || result.successes>attempts(p)) {
            std::fprintf(stderr,"Invalid CAS success total=%llu\n",static_cast<unsigned long long>(result.successes));
            ++result.errors;
        }
        expected=static_cast<uint32_t>(result.successes);
    } else {
        if(!counts.empty())throw std::invalid_argument("unexpected atomic success array");
        if(std::strcmp(p.operation,"atomicAdd")==0)expected=static_cast<uint32_t>(attempts(p));
    }
    if(counter!=expected) {
        std::fprintf(stderr,"Counter mismatch: actual=%u expected=%u (uint32 modulo arithmetic)\n",counter,expected);
        ++result.errors;
    }
    return result;
}
} // namespace atomic_bench
