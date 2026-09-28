#include "../tier2/shoc_devicememory/cpu_reference.h"
#include <cassert>
#include <vector>
#include <limits>
using namespace device_memory;
// Enumerate actual loop indices and materialize the input independently of the
// periodic reference. All active outputs are compared, including short tails.
static void check_read(int n,int stride) {
    std::vector<float> input(n),gold(std::min(n,stride));
    for(int i=0;i<n;++i) input[i]=static_cast<float>(i%1000)*0.001f;
    for(size_t gid=0;gid<gold.size();++gid)
        for(size_t i=gid;i<static_cast<size_t>(n);i+=stride) gold[gid]+=input[i];
    const ReadReference reference(n,stride);
    for(size_t gid=0;gid<gold.size();++gid) assert(gold[gid]==reference(gid));
    assert(compare("read",gold.data(),0,gold.size(),Operation::read,&reference)==0);
}
int main(int argc,char** argv) {
    if(argc>1) {
        try {const auto p=parameters(argc,argv);const auto g=layout(p);
            printf("%d,%d,%s,%d,%d,%d,%d\n",p.mib,p.block,p.mode,g.elements,g.blocks,g.threads,g.read_outputs);return 0;
        } catch(const std::exception& e) {fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
    }
    int cases=0;
    for(int n:{1,2,31,32,33,999,1000,1001,4095,4096,65537,262144})
        for(int stride:{1,7,31,32,255,256,999,1000,1001,65535,262145}) {check_read(n,stride);++cases;}
    uint32_t seed=123;
    for(int i=0;i<100;++i) {
        seed=1664525*seed+1013904223;int n=1+seed%100000;
        seed=1664525*seed+1013904223;int stride=1+seed%10000;check_read(n,stride);++cases;
    }
    // Force the production grid cap, including a partial final thread visit.
    check_read(64*262144,65535*256);++cases;
    printf("PASS: %d complete independent grid-stride reference cases\n",cases);
    std::vector<float> values(1007);
    for(Operation op:{Operation::write,Operation::readwrite}) {
        for(size_t offset:{size_t(0),size_t(999),size_t(1048573),size_t(16777210),size_t(1073740817)}) {
            for(size_t j=0;j<values.size();++j) {
                const size_t i=offset+j;
                const float pristine=static_cast<float>(i%1000)*0.001f;
                values[j]=op==Operation::write?static_cast<float>(i)*0.001f:pristine*1.0001f;
            }
            assert(compare("value",values.data(),offset,values.size(),op,nullptr)==0);
        }
    }
    for(float bad:{std::numeric_limits<float>::quiet_NaN(),INFINITY,-INFINITY}) {
        assert(!equal_finite(bad,1));assert(!equal_finite(1,bad));assert(!equal_finite(bad,bad));
    }
    assert(!equal_finite(1,std::nextafter(1.f,INFINITY)));
    assert(equal_finite(0,0) && equal_finite(1,1));
    ReadReference ref(1007,1007);
    for(size_t i=0;i<values.size();++i) values[i]=ref(i);
    for(size_t i:{size_t(0),size_t(500),size_t(1006)}) {
        const float original=values[i];
        for(float bad:{original+1,NAN,INFINITY,-INFINITY}) {
            values[i]=bad;assert(compare("fault",values.data(),0,values.size(),Operation::read,&ref)==1);
        }
        values[i]=original;
    }
    for(float& value:values) value+=1;
    assert(compare("all",values.data(),0,values.size(),Operation::read,&ref)==values.size());
    for(size_t offset=0;offset<values.size();) {
        const size_t count=std::min<size_t>(17,values.size()-offset);
        assert(compare("chunk",values.data()+offset,offset,count,Operation::read,&ref,100)==count);offset+=count;
    }
    for(auto pair:{std::pair<int,int>{0,1},{1,0},{-1,1}}) {
        bool caught=false;try{ReadReference bad(pair.first,pair.second);}catch(const std::invalid_argument&){caught=true;}assert(caught);
    }
    bool caught=false;try{expected_value(Operation::read,0,nullptr);}catch(const std::invalid_argument&){caught=true;}assert(caught);
    puts("PASS: full write/readwrite formulas, finite/exact comparisons, chunk offsets and diagnostics");
}
