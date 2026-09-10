#include "../tier2/npb_mg/mg_reference.h"
#include "fixtures/npb_mg_original.h"
#include <cstring>
#include <limits>
static void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(int argc,char**) {
    pinned_mg::Original original;
    uint64_t integer=314159265;double real=314159265;
    for(int i=0;i<30000;++i){integer=nas_mg::random_step(integer);original.randlc(&real,1220703125.0);require(static_cast<double>(integer)==real,"46-bit RNG differs");}
    size_t input_count=0;
    for(char name: {'S','W','A','B','C'}) {
        if(argc==1 && name!='S')continue;
        const auto c=*nas_mg::problem(name);const nas_mg::Layout layout(c);
        const auto input=nas_mg::input(c);const auto pinned=original.solve(c,true);
        require(input==pinned.v,"full pinned zran3 input differs");input_count+=input.size();
    }
    const auto c=*nas_mg::problem('S');const nas_mg::Layout layout(c);
    const auto pinned=original.solve(c);
    for(unsigned workers:{1,2,8}) {
        const auto result=nas_mg::reference(c,layout,workers);
        require(result.u==pinned.u && result.r==pinned.r && result.v==pinned.v && result.norms==pinned.norms,"full pinned trajectory differs");
        require(nas_mg::published_l2(result.norms[2],c),"published NAS L2 failed");
    }
    // Exercise every level including interior, periodic face, edge and corner,
    // both full measured arrays, binary input and all four actual norms.
    auto faulty=pinned;size_t faults=0;
    for(int array=0;array<4;++array) {
        std::vector<size_t> indices;
        if(array<2)for(int k=0;k<layout.levels();++k) {
            const size_t p=layout.width[k],o=layout.offset[k];
            for(size_t i:{size_t(0),p+1,p*p+p+1,p*p*p-1})indices.push_back(o+i);
        } else if(array==2)indices={0,1,pinned.v.size()/2,pinned.v.size()-1};
        else indices={0,1,2,3};
        double* values=array==0?faulty.u.data():array==1?faulty.r.data():array==2?faulty.v.data():faulty.norms.data();
        for(size_t i:indices)for(double delta:{0.01,std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
            const double saved=values[i];values[i]=saved+delta;
            require(nas_mg::compare(faulty,pinned)==1,"measured state corruption accepted");values[i]=saved;++faults;
        }
    }
    require(!nas_mg::published_l2(c.l2*1.000001,c),"NAS threshold too loose");
    require(!nas_mg::close(0,std::numeric_limits<double>::quiet_NaN()),"nonfinite reference accepted");
    printf("PASS: 30000 RNG states; %zu complete pinned input values; full Class S hierarchy/norm trajectory identical with 1/2/8 workers; %zu state/ghost/norm faults rejected\n",input_count,faults);
}
