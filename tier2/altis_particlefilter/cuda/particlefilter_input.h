// Input contract for the pinned Altis/Rodinia particle filter. See UPSTREAM_LICENSE.
#pragma once
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace pf {
constexpr int width=128, height=128, frames=11, steps=10, block=512, neighbors=69;
constexpr uint32_t initial_seed=12345;
constexpr double pi=3.14159265358979323846;
constexpr size_t video_size=width*height*frames;
using Offset=std::array<int,2>;

inline int integer(const char* s) {
    if(!s||!*s)throw std::invalid_argument("Particle Filter expects a positive decimal size");
    for(const char* p=s;*p;++p)if(*p<'0'||*p>'9')throw std::invalid_argument("Particle Filter size must contain decimal digits only");
    errno=0;char* end=nullptr;long n=std::strtol(s,&end,10);
    // Retain the signed indexing contract of the upstream gather array.
    if(errno||*end||n<1||n>INT_MAX/neighbors)throw std::invalid_argument("Particle Filter size exceeds the signed gather index range");
    return static_cast<int>(n);
}
inline int parameters(int argc,char** argv) {
    int n=100000;
    for(int i=1;i<argc;++i) {
        if(std::strcmp(argv[i],"--size")||i+1==argc)throw std::invalid_argument("Particle Filter accepts --size N");
        n=integer(argv[++i]);
    }
    if(const char* s=std::getenv("BENCH_PARAM_size"))n=integer(s);
    return n;
}
// Specify the two's-complement wrap intended by the upstream signed LCG,
// without signed overflow or implementation-defined unsigned-to-signed casts.
inline int32_t signed_bits(uint32_t bits) {
    return static_cast<int32_t>(bits<=INT32_MAX?int64_t(bits):int64_t(bits)-4294967296LL);
}
inline double uniform(int32_t& seed) {
    seed=signed_bits(1103515245u*static_cast<uint32_t>(seed)+12345u)%INT_MAX;
    return std::fabs(double(seed)/double(INT_MAX));
}
inline double normal(int32_t& seed) {
    const double u=uniform(seed),v=uniform(seed);
    // Upstream log(0) can produce infinity. Preserve draws, repair only endpoint.
    return std::sqrt(-2.0*std::log(u==0.0?1.0/INT_MAX:u))*std::cos(2.0*pi*v);
}
inline std::array<Offset,neighbors> offsets() {
    std::array<Offset,neighbors> out{};int i=0;
    for(int x=-4;x<=4;++x)for(int y=-4;y<=4;++y)
        if(x*x+y*y<25)out[i++]={{y,x}};
    if(i!=neighbors)throw std::logic_error("Invalid disk support");
    return out;
}
struct Input { std::vector<unsigned char> video;std::vector<int32_t> seeds;std::array<Offset,neighbors> disk; };
inline Input input(int n) {
    Input in{std::vector<unsigned char>(video_size,100),std::vector<int32_t>(n),offsets()};
    for(int i=0;i<n;++i)in.seeds[i]=signed_bits(initial_seed*static_cast<uint32_t>(i));
    // The approved square video keeps the upstream's actual x/y motion and
    // radius-five dilation. Direct construction initializes every pixel.
    for(int k=0;k<frames;++k) {
        const int cx=k==0?width/2:width/2-2*(k-1);
        const int cy=k==0?height/2:height/2+k-1;
        for(int dx=-4;dx<=4;++dx)for(int dy=-4;dy<=4;++dy)
            if(dx*dx+dy*dy<25)in.video[((cx+dx)*height+cy+dy)*frames+k]=228;
    }
    for(auto& pixel:in.video) {
        // Convert finite Gaussian noise to signed int before byte wrapping;
        // a negative floating-point value cannot be cast directly to uchar.
        const int noise=static_cast<int>(5.0*normal(in.seeds[0]));
        pixel=static_cast<unsigned char>(int(pixel)+noise);
    }
    return in;
}
inline int gather_index(double x,double y,Offset offset,int frame) {
    // roundDouble in the pinned source actually truncates (postincrement return).
    const int xx=static_cast<int>(x)+offset[1],yy=static_cast<int>(y)+offset[0];
    const int index=std::abs(xx*height*frames+yy*frames+frame);
    return index>=int(video_size)?0:index;
}
inline size_t device_bytes(int n) {
    return size_t(n)*(8*sizeof(double)+(neighbors+1)*sizeof(int32_t))+
        size_t((n+block-1)/block)*sizeof(double)+sizeof(double)+video_size+2*neighbors*sizeof(int);
}
} // namespace pf
