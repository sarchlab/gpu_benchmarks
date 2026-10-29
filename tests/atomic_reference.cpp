#include "../tier1/atomic_operations/hip/atomic_reference.h"
#include <cassert>
int main() {
    using namespace atomic_bench;
    Problem p;p.blocks=2;p.block=2;p.ops=3;
    assert(attempts(p)==12 && threads(p)==4);
    assert(verify(p,12,{}).errors==0);assert(verify(p,11,{}).errors==1);
    p.blocks=2048;p.block=256;p.ops=8192;assert(attempts(p)==(1ull<<32));
    assert(verify(p,0,{}).errors==0);assert(verify(p,1,{}).errors==1);
    p.ops=65536;assert(attempts(p)==(1ull<<35));assert(verify(p,0,{}).errors==0);
    p.operation="atomicMin";assert(verify(p,0,{}).errors==0);assert(verify(p,1,{}).errors==1);
    p.operation="atomicCAS";p.blocks=2;p.block=2;p.ops=3;
    assert(verify(p,3,{0,1,0,2}).errors==0);
    assert(verify(p,12,{3,3,3,3}).errors==0);
    assert(verify(p,4,{0,1,0,2}).errors==1);
    assert(verify(p,0,{0,0,0,0}).errors==1);
    assert(verify(p,2,{0,1,0,1}).errors==1);
    assert(verify(p,4,{0,4,0,0}).errors==1);
    assert(verify(p,16,{4,4,4,4}).errors==5);
    p.ops=INT_MAX;
    assert(verify(p,0xfffffffcU,{INT_MAX,INT_MAX,INT_MAX,INT_MAX}).errors==0);
    assert(verify(p,0xfffffffbU,{INT_MAX,INT_MAX,INT_MAX,INT_MAX}).errors==1);
    bool caught=false;try{verify(p,0,{});}catch(const std::invalid_argument&){caught=true;}assert(caught);
    p.blocks=1;p.block=16;p.ops=3;
    assert(verify(p,64,std::vector<uint32_t>(16,4)).errors==17);
    puts("PASS: add/min exact counters, CAS all-count bounds, corruption, uint32 wrap and size rejection");
}
