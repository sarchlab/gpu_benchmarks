#include <stdio.h>
#include <stdlib.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#define GPU_CHECK(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess){fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));exit(1);} } while(0)
typedef cudaEvent_t GpuEvent;
#define EVENT_CREATE(e) GPU_CHECK(cudaEventCreate(&e))
#define EVENT_RECORD(e) GPU_CHECK(cudaEventRecord(e))
#define EVENT_SYNC(e)   GPU_CHECK(cudaEventSynchronize(e))
#define EVENT_ELAPSED(ms,s,e) GPU_CHECK(cudaEventElapsedTime(&ms,s,e))
#define DEVICE_SYNC()   GPU_CHECK(cudaDeviceSynchronize())
#define MALLOC(p,sz)    GPU_CHECK(cudaMalloc(p,sz))
#define FREE(p)         GPU_CHECK(cudaFree(p))
#define MEMCPY_H2D(dst,src,sz) GPU_CHECK(cudaMemcpy(dst,src,sz,cudaMemcpyHostToDevice))
#define MEMCPY_D2H(dst,src,sz) GPU_CHECK(cudaMemcpy(dst,src,sz,cudaMemcpyDeviceToHost))
#elif defined(USE_HIP)
#include <hip/hip_runtime.h>
#define GPU_CHECK(x) do { hipError_t _e=(x); if(_e!=hipSuccess){fprintf(stderr,"HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(_e));exit(1);} } while(0)
typedef hipEvent_t GpuEvent;
#define EVENT_CREATE(e) GPU_CHECK(hipEventCreate(&e))
#define EVENT_RECORD(e) GPU_CHECK(hipEventRecord(e,0))
#define EVENT_SYNC(e)   GPU_CHECK(hipEventSynchronize(e))
#define EVENT_ELAPSED(ms,s,e) GPU_CHECK(hipEventElapsedTime(&ms,s,e))
#define DEVICE_SYNC()   GPU_CHECK(hipDeviceSynchronize())
#define MALLOC(p,sz)    GPU_CHECK(hipMalloc(p,sz))
#define FREE(p)         GPU_CHECK(hipFree(p))
#define MEMCPY_H2D(dst,src,sz) GPU_CHECK(hipMemcpy(dst,src,sz,hipMemcpyHostToDevice))
#define MEMCPY_D2H(dst,src,sz) GPU_CHECK(hipMemcpy(dst,src,sz,hipMemcpyDeviceToHost))
#else
#error "Define USE_CUDA or USE_HIP"
#endif

#define N (32*1024*1024)
#define WARMUP 5
#define ITERS  5

__global__ void copy_kernel(const float* __restrict__ a, float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i];
}

__global__ void scale_kernel(const float* __restrict__ c, float* __restrict__ b, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = s * c[i];
}

__global__ void add_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void triad_kernel(const float* __restrict__ b, const float* __restrict__ c, float* __restrict__ a, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = b[i] + s * c[i];
}

__global__ void dot_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ partial, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float val = (i < n) ? a[i] * b[i] : 0.0f;
    sdata[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

int main() {
    float *da, *db, *dc;
    MALLOC((void**)&da, N * sizeof(float));
    MALLOC((void**)&db, N * sizeof(float));
    MALLOC((void**)&dc, N * sizeof(float));

    /* Initialize on host then copy */
    float *ha = (float*)malloc(N * sizeof(float));
    float *hb = (float*)malloc(N * sizeof(float));
    float *hc = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) { ha[i] = 1.0f; hb[i] = 2.0f; hc[i] = 0.0f; }
    MEMCPY_H2D(da, ha, N * sizeof(float));
    MEMCPY_H2D(db, hb, N * sizeof(float));
    MEMCPY_H2D(dc, hc, N * sizeof(float));

    const float s = 2.0f;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    /* Dot reduction buffer */
    float *d_partial;
    MALLOC((void**)&d_partial, blocks * sizeof(float));
    float *h_partial = (float*)malloc(blocks * sizeof(float));

    GpuEvent ev_start, ev_stop;
    EVENT_CREATE(ev_start);
    EVENT_CREATE(ev_stop);

    /* --- COPY --- */
    for (int i = 0; i < WARMUP; i++) copy_kernel<<<blocks, threads>>>(da, dc, N);
    DEVICE_SYNC();
    EVENT_RECORD(ev_start);
    for (int i = 0; i < ITERS; i++) copy_kernel<<<blocks, threads>>>(da, dc, N);
    EVENT_RECORD(ev_stop);
    EVENT_SYNC(ev_stop);
    { float ms = 0; EVENT_ELAPSED(ms, ev_start, ev_stop); ms /= ITERS;
      double gbs = 2.0 * N * sizeof(float) / (ms / 1000.0) / 1e9;
      printf("babelstream_copy,%d,%.3f,%.2f\n", N, ms, gbs); }

    /* --- SCALE --- */
    for (int i = 0; i < WARMUP; i++) scale_kernel<<<blocks, threads>>>(dc, db, s, N);
    DEVICE_SYNC();
    EVENT_RECORD(ev_start);
    for (int i = 0; i < ITERS; i++) scale_kernel<<<blocks, threads>>>(dc, db, s, N);
    EVENT_RECORD(ev_stop);
    EVENT_SYNC(ev_stop);
    { float ms = 0; EVENT_ELAPSED(ms, ev_start, ev_stop); ms /= ITERS;
      double gbs = 2.0 * N * sizeof(float) / (ms / 1000.0) / 1e9;
      printf("babelstream_scale,%d,%.3f,%.2f\n", N, ms, gbs); }

    /* --- ADD --- */
    for (int i = 0; i < WARMUP; i++) add_kernel<<<blocks, threads>>>(da, db, dc, N);
    DEVICE_SYNC();
    EVENT_RECORD(ev_start);
    for (int i = 0; i < ITERS; i++) add_kernel<<<blocks, threads>>>(da, db, dc, N);
    EVENT_RECORD(ev_stop);
    EVENT_SYNC(ev_stop);
    { float ms = 0; EVENT_ELAPSED(ms, ev_start, ev_stop); ms /= ITERS;
      double gbs = 3.0 * N * sizeof(float) / (ms / 1000.0) / 1e9;
      printf("babelstream_add,%d,%.3f,%.2f\n", N, ms, gbs); }

    /* --- TRIAD --- */
    for (int i = 0; i < WARMUP; i++) triad_kernel<<<blocks, threads>>>(db, dc, da, s, N);
    DEVICE_SYNC();
    EVENT_RECORD(ev_start);
    for (int i = 0; i < ITERS; i++) triad_kernel<<<blocks, threads>>>(db, dc, da, s, N);
    EVENT_RECORD(ev_stop);
    EVENT_SYNC(ev_stop);
    { float ms = 0; EVENT_ELAPSED(ms, ev_start, ev_stop); ms /= ITERS;
      double gbs = 3.0 * N * sizeof(float) / (ms / 1000.0) / 1e9;
      printf("babelstream_triad,%d,%.3f,%.2f\n", N, ms, gbs); }

    /* --- DOT --- */
    for (int i = 0; i < WARMUP; i++) {
        dot_kernel<<<blocks, threads, threads * sizeof(float)>>>(da, db, d_partial, N);
        MEMCPY_D2H(h_partial, d_partial, blocks * sizeof(float));
        float sum = 0; for (int j = 0; j < blocks; j++) sum += h_partial[j];
        (void)sum;
    }
    DEVICE_SYNC();
    EVENT_RECORD(ev_start);
    for (int i = 0; i < ITERS; i++) {
        dot_kernel<<<blocks, threads, threads * sizeof(float)>>>(da, db, d_partial, N);
    }
    EVENT_RECORD(ev_stop);
    EVENT_SYNC(ev_stop);
    { float ms = 0; EVENT_ELAPSED(ms, ev_start, ev_stop); ms /= ITERS;
      double gbs = 2.0 * N * sizeof(float) / (ms / 1000.0) / 1e9;
      printf("babelstream_dot,%d,%.3f,%.2f\n", N, ms, gbs); }

    FREE(da); FREE(db); FREE(dc); FREE(d_partial);
    free(ha); free(hb); free(hc); free(h_partial);
    return 0;
}
