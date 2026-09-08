#include <cstdio>
#include <limits>
#include "../tier2/rodinia_lavamd/cpu_reference.h"
struct Particle {double v,x,y,z;};
static bool close_reference(double actual,long double expected) {
    return std::isfinite(actual) && std::isfinite(expected) &&
        std::fabs((long double)actual-expected)<=1e-11L*std::max(1.0L,std::fabs(expected));
}
int main() {
    int cases=0;
    const int shapes[][2]={{1,1},{1,2},{1,100},{2,1},{2,7},{2,100},{3,1},{3,7},{4,1}};
    for (const auto& shape:shapes)
        for (double alpha:{0.0,0.25,0.5,1.0})
            for (bool signed_charges:{false,true}) {
                const int n=shape[0],p=shape[1];const size_t count=(size_t)n*n*n*p;
                std::vector<Particle> positions(count);
                std::vector<double> charges(count);
                for (size_t i=0;i<count;++i) {
                    positions[i]={double(i*3%11+1)/8,double(int(i*7%17)-8)/8,
                        double(int(i*11%19)-9)/8,double(int(i*13%23)-11)/8};
                    charges[i]=double(i*5%13+1)/16*(signed_charges && i%2?-1:1);
                }
                const auto original=positions;const auto original_charges=charges;
                const auto actual=cpu_lavamd(positions.data(),charges.data(),n,p,alpha);
                if (actual.size()!=count || charges!=original_charges) return 1;
                for (size_t i=0;i<count;++i) {
                    if(positions[i].v!=original[i].v || positions[i].x!=original[i].x ||
                       positions[i].y!=original[i].y || positions[i].z!=original[i].z) return 1;
                    const size_t box=i/p;const int x=box%n,y=(box/n)%n,z=box/(n*n);
                    long double energy=0,fx=0,fy=0,fz=0;
                    // Independent all-particle oracle: select adjacent boxes
                    // by coordinate distance, without the helper's neighbor list.
                    for (size_t j=0;j<count;++j) {
                        const size_t other=j/p;const int xx=other%n,yy=(other/n)%n,zz=other/(n*n);
                        if(std::abs(x-xx)>1 || std::abs(y-yy)>1 || std::abs(z-zz)>1)continue;
                        const auto& a=positions[i];const auto& b=positions[j];
                        const long double dot=(long double)a.x*b.x+(long double)a.y*b.y+(long double)a.z*b.z;
                        const long double value=std::exp(-2.0L*alpha*alpha*((long double)a.v+b.v-dot));
                        energy+=charges[j]*value;
                        fx+=2.0L*charges[j]*value*((long double)a.x-b.x);
                        fy+=2.0L*charges[j]*value*((long double)a.y-b.y);
                        fz+=2.0L*charges[j]*value*((long double)a.z-b.z);
                    }
                    if(!close_reference(actual[i].v,energy)||!close_reference(actual[i].x,fx)||
                       !close_reference(actual[i].y,fy)||!close_reference(actual[i].z,fz)) {
                        std::fprintf(stderr,"FAIL boxes=%d particles=%d alpha=%g index=%zu\n",n,p,alpha,i);return 1;
                    }
                }
                ++cases;
            }
    // Uniform zero positions make every pair's energy exactly its charge and
    // every force zero. This exposes home/face/edge/corner membership errors.
    for (int n:{1,2,3,4}) {
        const int p=3;const size_t count=(size_t)n*n*n*p;
        std::vector<Particle> positions(count);std::vector<double> charges(count,1);
        const auto actual=cpu_lavamd(positions.data(),charges.data(),n,p,0.5);
        for (size_t i=0;i<count;++i) {
            const int box=i/p,x=box%n,y=(box/n)%n,z=box/(n*n);
            const int nx=1+(x>0)+(x<n-1),ny=1+(y>0)+(y<n-1),nz=1+(z>0)+(z<n-1);
            if(actual[i].v!=nx*ny*nz*p || actual[i].x!=0 || actual[i].y!=0 || actual[i].z!=0)return 1;
        }
    }
    if(!lavamd_value_matches(0,0)||!lavamd_value_matches(4e-11,0)||lavamd_value_matches(2e-10,0)||
       !lavamd_value_matches(100+4e-9,100)||lavamd_value_matches(100+2e-8,100)||
       !lavamd_value_matches(-100-4e-9,-100)||lavamd_value_matches(-100-2e-8,-100))return 1;
    for(double bad:{std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()})
        if(lavamd_value_matches(bad,0)||lavamd_value_matches(0,bad)||lavamd_value_matches(bad,bad))return 1;
    std::printf("PASS: %d full long-double references, four closed-form geometry controls, input preservation and finite/tolerance checks\n",cases);
    return 0;
}
