// Port of pinned Altis ex_particle_CUDA_float_seq.cu (double precision).
// Algorithm attribution and explicit safety exceptions: README.md, UPSTREAM_LICENSE.
#pragma once
#include "particlefilter_input.h"

namespace pf_gpu {
__device__ double uniform(int32_t& seed) {
    uint32_t bits=1103515245u*static_cast<uint32_t>(seed)+12345u;
    int64_t value=bits<=INT32_MAX?int64_t(bits):int64_t(bits)-4294967296LL;
    seed=static_cast<int32_t>(value)%INT_MAX;
    return fabs(double(seed)/double(INT_MAX));
}
__device__ double normal(int32_t& seed) {
    const double u=uniform(seed),v=uniform(seed);
    return sqrt(-2.0*log(u==0.0?1.0/INT_MAX:u))*cos(2.0*pf::pi*v);
}
__global__ void likelihood_kernel(int n,int frame,const unsigned char* video,const int* offsets,
        double* x,double* y,const double* xj,const double* yj,int* indices,double* likelihood,
        double* weights,int32_t* seeds,double* partial) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    __shared__ double buffer[pf::block];
    double weight=0;
    if(i<n) {
        x[i]=xj[i]+1.0+5.0*normal(seeds[i]);
        y[i]=yj[i]-2.0+2.0*normal(seeds[i]);
        for(int j=0;j<pf::neighbors;++j) {
            const int xx=static_cast<int>(x[i])+offsets[2*j+1];
            const int yy=static_cast<int>(y[i])+offsets[2*j];
            int index=abs(xx*pf::height*pf::frames+yy*pf::frames+frame);
            indices[i*pf::neighbors+j]=index>=int(pf::video_size)?0:index;
        }
        double sum=0;
        for(int j=0;j<pf::neighbors;++j) {
            const int pixel=video[indices[i*pf::neighbors+j]];
            sum+=(pow(double(pixel-100),2)-pow(double(pixel-228),2))/50.0;
        }
        likelihood[i]=sum/pf::neighbors;
        weight=(1.0/n)*exp(likelihood[i]);weights[i]=weight;
    }
    buffer[threadIdx.x]=weight;__syncthreads();
    for(unsigned stride=blockDim.x/2;stride;stride>>=1) {
        if(threadIdx.x<stride)buffer[threadIdx.x]+=buffer[threadIdx.x+stride];
        __syncthreads();
    }
    if(threadIdx.x==0)partial[blockIdx.x]=buffer[0];
}
__global__ void sum_kernel(double* partial,int blocks) {
    if(blockIdx.x==0&&threadIdx.x==0) {
        double total=0;for(int i=0;i<blocks;++i)total+=partial[i];partial[0]=total;
    }
}
__global__ void normalize_weights_kernel(double* weights,const double* partial,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)weights[i]/=partial[0];
}
// Kernel boundaries establish the missing device-wide happens-before edges.
__global__ void cdf_kernel(const double* weights,double* cdf,double* u0,int32_t* seeds,int n) {
    if(blockIdx.x==0&&threadIdx.x==0) {
        cdf[0]=weights[0];for(int i=1;i<n;++i)cdf[i]=weights[i]+cdf[i-1];
        *u0=(1.0/n)*uniform(seeds[0]);
    }
}
__global__ void thresholds_kernel(const double* u0,double* u,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)u[i]=*u0+double(i)/n;
}
__global__ void find_index_kernel(const double* x,const double* y,const double* cdf,
        const double* u,double* xj,double* yj,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) {
        int index=n-1;
        for(int j=0;j<n;++j)if(cdf[j]>=u[i]) {index=j;break;}
        xj[i]=x[index];yj[i]=y[index];
    }
}
} // namespace pf_gpu
