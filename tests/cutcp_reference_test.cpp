#include "../tier2/parboil_cutcp/cpu_reference.h"
#include <climits>
#include <cstdio>
#include <cstring>

struct Atom { float x, y, z, w; };
static void require(bool ok) { if (!ok) std::abort(); }
static float point(const std::vector<Atom>& atoms, int i, int j, int k, float h, float cutoff) {
    const float x=i*h, y=j*h, z=k*h, cutoff2=cutoff*cutoff, inv=1.f/cutoff2;
    float ref=0.f;
    for (const auto& a:atoms) {
        const float dx=a.x-x, dy=a.y-y, dz=a.z-z;
        const float r2=dx*dx+dy*dy+dz*dz;
        if (r2<cutoff2) { const float s=1.f-r2*inv; ref+=a.w*(1.f/sqrtf(r2))*s*s; }
    }
    return ref;
}
static void compare(const std::vector<Atom>& a,int n,float h,float c,unsigned workers) {
    const auto saved=a;
    const auto result=cutcp_reference::potential(a,n,h,c,workers);
    require(result.size()==std::size_t(n)*n*n);
    require(a.empty() || !std::memcmp(a.data(),saved.data(),a.size()*sizeof(Atom)));
    for(int k=0;k<n;++k)for(int j=0;j<n;++j)for(int i=0;i<n;++i) {
        const float expected=point(a,i,j,k,h,c);
        require(!std::memcmp(&expected,&result[(std::size_t(k)*n+j)*n+i],sizeof(float)));
    }
}
int main() {
    // Independent analytical value: q=1, distance=1, cutoff=2 -> (1-1/4)^2.
    auto v=cutcp_reference::potential(std::vector<Atom>{{1,0,0,1}},1,1,2,1);
    require(v[0]==0.5625f);
    v=cutcp_reference::potential(std::vector<Atom>{{2,0,0,1}},1,1,2,1);
    require(v[0]==0.f); // Strict cutoff boundary.
    for(float x:{std::nextafter(2.f,0.f),2.f,std::nextafter(2.f,3.f),-2.f,1000.f,-1000.f})
        compare({{x,0.25f,0.5f,0.75f}},9,0.5f,2.f,16);
    compare({},3,.5f,1.f,16);
    // Alternating charges, cancellation, outside-box atoms, and uneven slabs.
    for(int n:{1,7,8,9,16,17})for(float h:{.5f,1.f})for(float c:{4.f,8.f,12.f}) {
        std::vector<Atom> a;
        for(int t=0;t<32;++t)
            a.push_back({float((t*13)%41)*.37f-.13f,float((t*19)%37)*.29f+.17f,
                         float((t*7)%43)*.41f-.23f,(t%2?-1.f:1.f)*(t+1)*.125f});
        for(unsigned workers:{1u,3u,16u,100u})compare(a,n,h,c,workers);
    }
    v=cutcp_reference::potential(std::vector<Atom>{{0,0,0,1}},1,1,2);
    require(!std::isfinite(v[0])); // Singular references must remain detectable.
    for(int n:{0,-1,INT_MAX}) {
        bool rejected=false;try{cutcp_reference::lattice_size(n);}catch(const std::exception&){rejected=true;}
        require(rejected);
    }
    for(float h:{0.f,-1.f,INFINITY,NAN}) {
        bool rejected=false;try{cutcp_reference::potential(std::vector<Atom>{},8,h,8);}catch(const std::exception&){rejected=true;}
        require(rejected);
    }
    for(float c:{0.f,-1.f,INFINITY,NAN,1e-30f,1e30f}) {
        bool rejected=false;try{cutcp_reference::potential(std::vector<Atom>{},8,1,c);}catch(const std::exception&){rejected=true;}
        require(rejected);
    }
    for(Atom a:std::vector<Atom>{{NAN,0,0,1},{0,INFINITY,0,1},{0,0,NAN,1},{0,0,0,NAN}}) {
        bool rejected=false;try{cutcp_reference::potential(std::vector<Atom>{a},8,1,8);}catch(const std::exception&){rejected=true;}
        require(rejected);
    }
    std::puts("PASS: analytical cutoff, complete naive equality, input immutability, slab equivalence and reference guards");
}
