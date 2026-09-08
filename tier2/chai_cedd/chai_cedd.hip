/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba,
 *        University of Illinois nor the names of its contributors may be used
 *        to endorse or promote products derived from this Software without
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

// Chai CUDA-D/CEDD, pinned at 61db988662cb993fa108429a13759dbdc83ab9be.
// Preserve the four byte-valued operators and A/B ping-pong dependencies.
// Port changes: bounded cooperative halo loads, complete partial tiles, defined
// scratch borders, shared timing protocol and full measured-output verification.
#include "../../common/bench_timing.h"
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

static constexpr int CEDD_TILE = 16;

static size_t cedd_shared_bytes(dim3 block) {
    return (size_t)(block.x+2)*(block.y+2)*sizeof(int);
}

// All threads participate, including inactive output threads in partial tiles.
// Callers use square 16x16 blocks, with a one-pixel halo in shared memory.
__device__ void load_cedd_tile(const unsigned char* data, int* tile, int rows, int cols) {
    const int pitch = blockDim.x + 2;
    for (int t=threadIdx.y*blockDim.x+threadIdx.x; t<pitch*pitch;
         t+=blockDim.x*blockDim.y) {
        int row=blockIdx.y*blockDim.y+t/pitch;
        int col=blockIdx.x*blockDim.x+t%pitch;
        tile[t]=(row<rows && col<cols) ? data[(size_t)row*cols+col] : 0;
    }
}

__constant__ float gaus[3][3] = {{0.0625f, 0.125f, 0.0625f}, {0.1250f, 0.250f, 0.1250f}, {0.0625f, 0.125f, 0.0625f}};
__constant__ int   sobx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
__constant__ int   soby[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

// https://github.com/smskelley/canny-opencl
// Gaussian Kernel
// data: image input data with each pixel taking up 1 byte (8Bit 1Channel)
// out: image output data (8B1C)
__global__ void gaussian_kernel(unsigned char *data, unsigned char *out, int rows, int cols) {

    extern __shared__ int l_mem[];
    int* l_data = l_mem;

    const int L_SIZE = blockDim.x;
    int sum         = 0;
    const int g_row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    const int g_col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int l_row = threadIdx.y + 1;
    const int l_col = threadIdx.x + 1;

    load_cedd_tile(data, l_data, rows, cols);
    __syncthreads();
    if (g_row >= rows-1 || g_col >= cols-1) return;
    const int pos = g_row * cols + g_col;

    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            sum += gaus[i][j] * l_data[(i + l_row - 1) * (L_SIZE + 2) + j + l_col - 1];
        }
    }

    out[pos] = min(255, max(0, sum));
}

// Upstream float magnitude/angle quantization, shared with the domain test.
__device__ void cedd_gradient(float sumx, float sumy, unsigned char* magnitude,
                              unsigned char* direction) {
    const float PI = 3.14159265f;
    float angle = 0;
    // The output is now the square root of their squares, but they are
    // constrained to 0 <= value <= 255. Note that hypot is a built in function
    // defined as: hypot(x,y) = sqrt(x*x, y*y).
    *magnitude = min(255, max(0, (int)hypot(sumx, sumy)));

    // Compute the direction angle theta in radians
    // atan2 has a range of (-PI, PI) degrees
    angle = atan2(sumy, sumx);

    // If the angle is negative,
    // shift the range to (0, 2PI) by adding 2PI to the angle,
    // then perform modulo operation of 2PI
    if(angle < 0) {
        angle = fmod((angle + 2 * PI), (2 * PI));
    }

    // Round the angle to one of four possibilities: 0, 45, 90, 135 degrees
    // then store it in the theta buffer at the proper position
    //*direction = ((int)(degrees(angle * (PI/8) + PI/8-0.0001) / 45) * 45) % 180;
    if(angle <= PI / 8)
        *direction = 0;
    else if(angle <= 3 * PI / 8)
        *direction = 45;
    else if(angle <= 5 * PI / 8)
        *direction = 90;
    else if(angle <= 7 * PI / 8)
        *direction = 135;
    else if(angle <= 9 * PI / 8)
        *direction = 0;
    else if(angle <= 11 * PI / 8)
        *direction = 45;
    else if(angle <= 13 * PI / 8)
        *direction = 90;
    else if(angle <= 15 * PI / 8)
        *direction = 135;
    else
        *direction = 0; // (angle <= 16*PI/8)
}

// Sobel kernel. Apply sobx and soby separately, then find the sqrt of their
//               squares.
// data:  image input data with each pixel taking up 1 byte (8Bit 1Channel)
// out:   image output data (8B1C)
// theta: angle output data
__global__ void sobel_kernel(unsigned char *data, unsigned char *out, unsigned char *theta,
    int rows, int cols) {

    extern __shared__ int l_mem[];
    int* l_data = l_mem;

    // collect sums separately. we're storing them into floats because that
    // is what hypot and atan2 will expect.
    const int L_SIZE = blockDim.x;
    const int   g_row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    const int   g_col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int   l_row = threadIdx.y + 1;
    const int   l_col = threadIdx.x + 1;

    load_cedd_tile(data, l_data, rows, cols);
    __syncthreads();
    if (g_row >= rows-1 || g_col >= cols-1) return;
    const int pos = g_row * cols + g_col;

    float sumx = 0, sumy = 0;
    // find x and y derivatives
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            sumx += sobx[i][j] * l_data[(i + l_row - 1) * (L_SIZE + 2) + j + l_col - 1];
            sumy += soby[i][j] * l_data[(i + l_row - 1) * (L_SIZE + 2) + j + l_col - 1];
        }
    }

    cedd_gradient(sumx, sumy, out+pos, theta+pos);
}

// Non-maximum Supression Kernel
// data: image input data with each pixel taking up 1 byte (8Bit 1Channel)
// out: image output data (8B1C)
// theta: angle input data
__global__ void non_max_supp_kernel(unsigned char *data, unsigned char *out,
    unsigned char *theta, int rows, int cols) {

    extern __shared__ int l_mem[];
    int* l_data = l_mem;

    // These variables are offset by one to avoid seg. fault errors
    // As such, this kernel ignores the outside ring of pixels
    const int L_SIZE = blockDim.x;
    const int g_row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    const int g_col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int l_row = threadIdx.y + 1;
    const int l_col = threadIdx.x + 1;

    load_cedd_tile(data, l_data, rows, cols);
    __syncthreads();
    if (g_row >= rows-1 || g_col >= cols-1) return;
    const int pos = g_row * cols + g_col;

    unsigned char my_magnitude = l_data[l_row * (L_SIZE + 2) + l_col];

    // The following variables are used to address the matrices more easily
    switch(theta[pos]) {
    // A gradient angle of 0 degrees = an edge that is North/South
    // Check neighbors to the East and West
    case 0:
        // supress me if my neighbor has larger magnitude
        if(my_magnitude <= l_data[l_row * (L_SIZE + 2) + l_col + 1] || // east
            my_magnitude <= l_data[l_row * (L_SIZE + 2) + l_col - 1]) // west
        {
            out[pos] = 0;
        }
        // otherwise, copy my value to the output buffer
        else {
            out[pos] = my_magnitude;
        }
        break;

    // A gradient angle of 45 degrees = an edge that is NW/SE
    // Check neighbors to the NE and SW
    case 45:
        // supress me if my neighbor has larger magnitude
        if(my_magnitude <= l_data[(l_row - 1) * (L_SIZE + 2) + l_col + 1] || // north east
            my_magnitude <= l_data[(l_row + 1) * (L_SIZE + 2) + l_col - 1]) // south west
        {
            out[pos] = 0;
        }
        // otherwise, copy my value to the output buffer
        else {
            out[pos] = my_magnitude;
        }
        break;

    // A gradient angle of 90 degrees = an edge that is E/W
    // Check neighbors to the North and South
    case 90:
        // supress me if my neighbor has larger magnitude
        if(my_magnitude <= l_data[(l_row - 1) * (L_SIZE + 2) + l_col] || // north
            my_magnitude <= l_data[(l_row + 1) * (L_SIZE + 2) + l_col]) // south
        {
            out[pos] = 0;
        }
        // otherwise, copy my value to the output buffer
        else {
            out[pos] = my_magnitude;
        }
        break;

    // A gradient angle of 135 degrees = an edge that is NE/SW
    // Check neighbors to the NW and SE
    case 135:
        // supress me if my neighbor has larger magnitude
        if(my_magnitude <= l_data[(l_row - 1) * (L_SIZE + 2) + l_col - 1] || // north west
            my_magnitude <= l_data[(l_row + 1) * (L_SIZE + 2) + l_col + 1]) // south east
        {
            out[pos] = 0;
        }
        // otherwise, copy my value to the output buffer
        else {
            out[pos] = my_magnitude;
        }
        break;

    default: out[pos] = my_magnitude; break;
    }
}

// Hysteresis Threshold Kernel
// data: image input data with each pixel taking up 1 byte (8Bit 1Channel)
// out: image output data (8B1C)
__global__ void hyst_kernel(unsigned char *data, unsigned char *out, int rows, int cols) {
    // Establish our high and low thresholds as floats
    float lowThresh  = 10;
    float highThresh = 70;

    // These variables are offset by one to avoid seg. fault errors
    // As such, this kernel ignores the outside ring of pixels
    const int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    const int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (row >= rows-1 || col >= cols-1) return;
    const int pos = row * cols + col;

    const unsigned char EDGE = 255;

    unsigned char magnitude = data[pos];

    if(magnitude >= highThresh)
        out[pos] = EDGE;
    else if(magnitude <= lowThresh)
        out[pos] = 0;
    else {
        float med = (highThresh + lowThresh) / 2;

        if(magnitude >= med)
            out[pos] = EDGE;
        else
            out[pos] = 0;
    }
}


struct CeddLaps { bench_lap_t gaussian, sobel, nms, hysteresis; };

// Input A contains the image; B and theta have zero borders before entry.
// Each interior is overwritten by its defining stage, without intervening resets.
static CeddLaps run_cedd_pass(unsigned char* a, unsigned char* b,
                              unsigned char* theta, int width, int height) {
    dim3 block(CEDD_TILE,CEDD_TILE);
    dim3 grid((width-2+CEDD_TILE-1)/CEDD_TILE,(height-2+CEDD_TILE-1)/CEDD_TILE);
    const size_t shared_bytes=cedd_shared_bytes(block);
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_pass_warm();
    CeddLaps laps;
    laps.gaussian=bench_time_op(&timer,[&]() {
        gaussian_kernel<<<grid,block,shared_bytes>>>(a,b,height,width);
    });
    laps.sobel=bench_time_op(&timer,[&]() {
        sobel_kernel<<<grid,block,shared_bytes>>>(b,a,theta,height,width);
    });
    laps.nms=bench_time_op(&timer,[&]() {
        non_max_supp_kernel<<<grid,block,shared_bytes>>>(a,b,theta,height,width);
    });
    laps.hysteresis=bench_time_op(&timer,[&]() {
        hyst_kernel<<<grid,block>>>(b,a,height,width);
    });
    bench_timer_destroy(&timer);
    return laps;
}

// Independent integer/double CPU evaluation, with the same defined boundaries.
// Gaussian truncates EACH positive term because upstream uses an int accumulator.
static std::vector<unsigned char> reference_cedd(const std::vector<unsigned char>& input,
                                                int width, int height) {
    std::vector<unsigned char> a=input, b(input.size(),0), theta(input.size(),0);
    const int weights[3][3]={{1,2,1},{2,4,2},{1,2,1}};
    for (int y=1;y<height-1;++y) for (int x=1;x<width-1;++x) {
        int sum=0;
        for (int dy=-1;dy<=1;++dy) for (int dx=-1;dx<=1;++dx)
            sum += (weights[dy+1][dx+1]*input[(size_t)(y+dy)*width+x+dx])/16;
        b[(size_t)y*width+x]=(unsigned char)sum;
    }
    for (int y=1;y<height-1;++y) for (int x=1;x<width-1;++x) {
        size_t p=(size_t)y*width+x;
        int gx=b[p-width+1]+2*b[p+1]+b[p+width+1]
              -b[p-width-1]-2*b[p-1]-b[p+width-1];
        int gy=b[p+width-1]+2*b[p+width]+b[p+width+1]
              -b[p-width-1]-2*b[p-width]-b[p-width+1];
        a[p]=(unsigned char)std::min(255,(int)std::sqrt((double)gx*gx+(double)gy*gy));
        // Integer gradients never sit on the irrational 22.5/67.5-degree ties.
        const double tangent=std::sqrt(2.0)-1.0;
        if (std::abs(gy)<=std::abs(gx)*tangent) theta[p]=0;
        else if (std::abs(gx)<=std::abs(gy)*tangent) theta[p]=90;
        else theta[p]=((gx<0)==(gy<0))?45:135;
    }
    for (int y=1;y<height-1;++y) for (int x=1;x<width-1;++x) {
        size_t p=(size_t)y*width+x;
        int offset=theta[p]==0?1:theta[p]==45?width-1:theta[p]==90?width:width+1;
        b[p]=(a[p]>a[p-offset] && a[p]>a[p+offset])?a[p]:0;
    }
    for (int y=1;y<height-1;++y) for (int x=1;x<width-1;++x) {
        size_t p=(size_t)y*width+x;
        // Pinned low=10/high=70 branches reduce to this per-pixel midpoint.
        a[p]=b[p]>=40?255:0;
    }
    return a;
}

static bool verify_cedd(const std::vector<unsigned char>& input,
                        const std::vector<unsigned char>& actual, int width, int height) {
    std::vector<unsigned char> expected=reference_cedd(input,width,height);
    if (actual.size()!=expected.size()) {
        fprintf(stderr,"FAIL: incomplete measured CEDD output\n"); return false;
    }
    for (size_t i=0;i<expected.size();++i) if (actual[i]!=expected[i]) {
        fprintf(stderr,"FAIL: measured CEDD pixel[%zu]: got %u expected %u\n",
                i,(unsigned)actual[i],(unsigned)expected[i]); return false;
    }
    fprintf(stderr,"PASS: verified all %zu measured pixels exactly (full_cpu_byte_v1)\n",actual.size());
    return true;
}

static bool read_dimension(const char* name,const char* text,int* value) {
    errno=0;
    char* end=nullptr;
    long parsed=strtol(text,&end,10);
    if (errno || end==text || *end || parsed<3 || parsed>INT_MAX) {
        fprintf(stderr,"Invalid %s: %s (requires integer >=3)\n",name,text); return false;
    }
    *value=(int)parsed; return true;
}

int main(int argc,char** argv) {
    setvbuf(stdout,nullptr,_IONBF,0);
    int width=2048,height=2048;
    if (getenv("BENCH_PARAM_threshold_low") || getenv("BENCH_PARAM_threshold_high")) {
        fprintf(stderr,"Chai CEDD has fixed thresholds 10/70; threshold overrides are unsupported\n"); return 1;
    }
    // Retain W=/H= command-line inputs and environment precedence.
    for (int i=1;i<argc;++i) {
        if (!strncmp(argv[i],"W=",2)) { if (!read_dimension("width",argv[i]+2,&width)) return 1; }
        else if (!strncmp(argv[i],"H=",2)) { if (!read_dimension("height",argv[i]+2,&height)) return 1; }
        else { fprintf(stderr,"Unknown argument: %s; use W= and H= only\n",argv[i]); return 1; }
    }
    const char* text=getenv("BENCH_PARAM_width");
    if (text && !read_dimension("width",text,&width)) return 1;
    text=getenv("BENCH_PARAM_height");
    if (text && !read_dimension("height",text,&height)) return 1;
    if (width>INT_MAX/height) { fprintf(stderr,"width*height exceeds signed kernel indexing\n"); return 1; }
    const size_t pixels=(size_t)width*height;
    int device=0;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop,device));
    fprintf(stderr,"Device: %s (id=%d), Chai CEDD %dx%d\n",prop.name,device,width,height);
    std::vector<unsigned char> input(pixels), output(pixels);
    srand(42);
    for (size_t i=0;i<pixels;++i) input[i]=(unsigned char)(rand()%256);
    unsigned char *d_a,*d_b,*d_theta;
    HIP_CHECK(hipMalloc(&d_a,pixels));
    HIP_CHECK(hipMalloc(&d_b,pixels));
    HIP_CHECK(hipMalloc(&d_theta,pixels));
    HIP_CHECK(hipMemcpy(d_a,input.data(),pixels,hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_b,0,pixels));
    HIP_CHECK(hipMemset(d_theta,0,pixels));
    CeddLaps laps=run_cedd_pass(d_a,d_b,d_theta,width,height);
    char params[320];
    snprintf(params,sizeof(params),"\"width\":%d,\"height\":%d,\"operator\":\"chai_61db988\",\"boundary\":\"input_a_zero_scratch_v1\",\"input_generator\":\"libc_rand42_mod256\",\"verification\":\"full_cpu_byte_v1\"",width,height);
    bench_emit_iter("gaussian_kernel",0,&laps.gaussian,params);
    bench_emit_iter("sobel_kernel",0,&laps.sobel,params);
    bench_emit_iter("non_max_supp_kernel",0,&laps.nms,params);
    bench_emit_iter("hyst_kernel",0,&laps.hysteresis,params);
    bench_emit_summary(laps.gaussian.gpu_ms+laps.sobel.gpu_ms+laps.nms.gpu_ms+laps.hysteresis.gpu_ms);
    HIP_CHECK(hipMemcpy(output.data(),d_a,pixels,hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_theta));
    return verify_cedd(input,output,width,height)?EXIT_SUCCESS:EXIT_FAILURE;
}
