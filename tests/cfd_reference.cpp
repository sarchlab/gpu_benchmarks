#include "../tier2/altis_cfd/hip/cfd_reference.h"
#include <map>
#include <set>
#include <cassert>
#include <limits>
// Execute the frozen upstream scalar bodies on a host for an independently
// arranged FP32 oracle. The production CPU reference is matrix-form FP64.
struct float3 {float x,y,z;};
struct Index {int x;};
static Index blockDim{1},blockIdx{0},threadIdx{0};
#define __device__
#define __host__
#define __global__
#define __constant__
#include "fixtures/cfd_upstream.h"
#include "../tier2/altis_cfd/hip/cfd_kernels.h"
#undef __device__
#undef __host__
#undef __global__
#undef __constant__

static void check_farfield() {
    const auto expected=cfd_upstream::farfield();auto actual=cfd::farfield();
    assert(std::memcmp(actual.data(),expected.variables,5*sizeof(float))==0);
    float pressure=1;float3 momentum{actual[1],actual[2],actual[3]},velocity{1.2f,0,0},x,y,z,e;
    cfd_device::compute_flux_contribution(actual[0],momentum,actual[4],pressure,velocity,x,y,z,e);
    assert(std::memcmp(&x,&expected.momentum_x,sizeof(x))==0);
    assert(std::memcmp(&y,&expected.momentum_y,sizeof(y))==0);
    assert(std::memcmp(&z,&expected.momentum_z,sizeof(z))==0);
    assert(std::memcmp(&e,&expected.energy,sizeof(e))==0);
}
static void check_mesh(int cells,int block) {
    const cfd::Problem p{cells,block,(cells+block-1)/block*block};const auto in=cfd::input(p);const int n=p.padded;
    using FaceKey=std::array<cfd::Point,3>;std::map<FaceKey,std::vector<std::pair<int,int>>> faces;
    const auto& topology=cfd::topology();
    for(int i=0;i<cells;++i) {
        const int cube=i/6;const cfd::Point origin{{cube%in.side,(cube/in.side)%in.side,cube/(in.side*in.side)}};
        const auto& tet=topology.tetra[i%6];
        const float q=float((11*origin[0]+7*origin[1]+3*origin[2]+i%6)%33-16)/256.0f;
        const float density=1.4f+q,pressure=1.0f+0.5f*q,speed=1.2f;
        const float energy=density*(0.5f*(speed*speed))+pressure/float(1.4f-1.0f);
        assert(in.initial[i]==density&&in.initial[i+n]==density*speed);
        assert(in.initial[i+2*n]==0&&in.initial[i+3*n]==0&&in.initial[i+4*n]==energy);
        int det=cfd::dot(cfd::subtract(tet[1],tet[0]),cfd::cross(cfd::subtract(tet[2],tet[0]),cfd::subtract(tet[3],tet[0])));
        assert(std::abs(det)==1 && in.areas[i]==1.0f/6.0f);
        for(int d=0;d<3;++d){float sum=0;for(int f=0;f<4;++f)sum+=in.normals[i+(f+4*d)*n];assert(sum==0);}
        for(int f=0;f<4;++f){faces[cfd::face(tet,f,origin)].push_back({i,f});int j=in.neighbors[i+f*n];assert(j>=-2&&j<cells&&j!=i);}
    }
    for(const auto& face:faces) {
        const auto& incident=face.second;assert(incident.size()==1||incident.size()==2);
        const auto a=incident[0];const int neighbor=in.neighbors[a.first+a.second*n];
        if(incident.size()==1) {
            const bool wall=face.first[0][1]==0&&face.first[1][1]==0&&face.first[2][1]==0;
            assert(neighbor==(wall?-1:-2));
        } else {
            const auto b=incident[1];assert(neighbor==b.first);assert(in.neighbors[b.first+b.second*n]==a.first);
            for(int d=0;d<3;++d)assert(in.normals[a.first+(a.second+4*d)*n]==-in.normals[b.first+(b.second+4*d)*n]);
        }
    }
    for(int i=cells;i<n;++i) {
        for(int k=0;k<4;++k)assert(in.neighbors[i+k*n]==in.neighbors[cells-1+k*n]);
        for(int k=0;k<12;++k)assert(in.normals[i+k*n]==in.normals[cells-1+k*n]);
        for(int k=0;k<5;++k)assert(in.initial[i+k*n]==in.initial[cells-1+k*n]);
    }
}
static void check_solver(int cells,int block) {
    cfd::Problem p{cells,block,(cells+block-1)/block*block};auto in=cfd::input(p);const size_t n=p.padded;
    const auto ff=cfd_upstream::farfield();for(int k=0;k<5;++k)cfd_upstream::ff_variable[k]=ff.variables[k];
    cfd_upstream::ff_flux_contribution_momentum_x[0]=ff.momentum_x;
    cfd_upstream::ff_flux_contribution_momentum_y[0]=ff.momentum_y;
    cfd_upstream::ff_flux_contribution_momentum_z[0]=ff.momentum_z;
    cfd_upstream::ff_flux_contribution_density_energy[0]=ff.energy;
    auto state=in.initial;std::vector<float> old(5*n),flux(5*n),factor(n);
    for(int t=0;t<10;++t) {
        old=state;
        for(size_t i=0;i<n;++i){blockIdx.x=i;cfd_upstream::cuda_compute_step_factor(n,state.data(),in.areas.data(),factor.data());}
        for(int s=0;s<3;++s) {
            for(size_t i=0;i<n;++i){blockIdx.x=i;cfd_upstream::cuda_compute_flux(n,in.neighbors.data(),in.normals.data(),state.data(),flux.data());}
            for(size_t i=0;i<n;++i){blockIdx.x=i;cfd_upstream::cuda_time_step(s,n,old.data(),state.data(),factor.data(),flux.data());}
        }
    }
    const auto expected=cfd::reference(p,in);
    for(size_t i=0;i<5*n;++i){assert(cfd::matches(state[i],expected.state[i]));assert(cfd::matches(old[i],expected.old[i]));assert(cfd::matches(flux[i],expected.flux[i]));}
    for(size_t i=0;i<n;++i){assert(cfd::matches(factor[i],expected.factor[i]));auto u=cfd::load(expected.state,n,i);assert(u[0]>0&&cfd::physical(u).pressure>0);}
    assert(!cfd::matches(std::numeric_limits<float>::quiet_NaN(),1));assert(!cfd::matches(1,std::numeric_limits<double>::infinity()));
    assert(!cfd::matches(1.01f,1));
}
int main() {
    check_farfield();
    int meshes=0,solves=0;
    for(int n=1;n<=83;++n)for(int b:{1,7,32,256}){check_mesh(n,b);++meshes;}
    for(int n:{125,127,128,129,500,503,1023,1024,1025,4096}){check_mesh(n,256);++meshes;}
    for(int n:{1,2,5,6,7,31,127,128,129,500,503,4097})for(int b:{1,7,256}){check_solver(n,b);++solves;}
    printf("PASS: %d independent geometric mesh checks; %d complete frozen-FP32 versus independent-FP64 trajectories and all final auxiliary arrays\n",meshes,solves);
}
