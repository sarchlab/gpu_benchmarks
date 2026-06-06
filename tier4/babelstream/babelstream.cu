#include <stdio.h>
#include <stdlib.h>

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

static int WARMUP = 5;

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
    setvbuf(stdout, NULL, _IONBF, 0);

    int n = 32 * 1024 * 1024;
    int threads_per_block = 256;
    float scalar = 2.0f;
    int num_times = 100;

    /* Read parameters from environment variables */
    const char* env_val;
    env_val = getenv("BENCH_PARAM_n");
    if (env_val) n = atoi(env_val);
    env_val = getenv("BENCH_PARAM_threads_per_block");
    if (env_val) threads_per_block = atoi(env_val);
    env_val = getenv("BENCH_PARAM_scalar");
    if (env_val) scalar = (float)atof(env_val);
    env_val = getenv("BENCH_PARAM_num_times");
    if (env_val) num_times = atoi(env_val);

    float *da, *db, *dc;
    MALLOC((void**)&da, (size_t)n * sizeof(float));
    MALLOC((void**)&db, (size_t)n * sizeof(float));
    MALLOC((void**)&dc, (size_t)n * sizeof(float));

    /* Initialize on host then copy */
    float *ha = (float*)malloc((size_t)n * sizeof(float));
    float *hb = (float*)malloc((size_t)n * sizeof(float));
    float *hc = (float*)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) { ha[i] = 1.0f; hb[i] = 2.0f; hc[i] = 0.0f; }
    MEMCPY_H2D(da, ha, (size_t)n * sizeof(float));
    MEMCPY_H2D(db, hb, (size_t)n * sizeof(float));
    MEMCPY_H2D(dc, hc, (size_t)n * sizeof(float));

    int blocks = (n + threads_per_block - 1) / threads_per_block;

    /* Dot reduction buffer */
    float *d_partial;
    MALLOC((void**)&d_partial, (size_t)blocks * sizeof(float));
    float *h_partial = (float*)malloc((size_t)blocks * sizeof(float));

    GpuEvent ev_start, ev_stop;
    EVENT_CREATE(ev_start);
    EVENT_CREATE(ev_stop);

    double total_time_all = 0.0;

    /* --- COPY --- */
    for (int i = 0; i < WARMUP; i++) copy_kernel<<<blocks, threads_per_block>>>(da, dc, n);
    DEVICE_SYNC();
    EVENT_RECORD(ev_start);
    for (int i = 0; i < num_times; i++) copy_kernel<<<blocks, threads_per_block>>>(da, dc, n);
    EVENT_RECORD(ev_stop);
    EVENT_SYNC(ev_stop);
    { float ms = 0; EVENT_ELAPSED(ms, ev_start, ev_stop); ms /= num_times;
      double gbs = 2.0 * n * sizeof(float) / (ms / 1000.0) / 1e9;
      total_time_all += ms * num_times;
      printf("{\"type\":\"kernel\",\"name\":\"copy_kernel\",\"time_ms\":%.6f,"
             "\"params\":{\"n\":%d,\"threads_per_block\":%d,\"scalar\":%.6f,\"num_times\":%d}}\n",
             (double)ms, n, threads_per_block, (double)scalar, num_times);
      fprintf(stderr, "copy_kernel: %.3f ms, %.2f GB/s\n", ms, gbs); }

    /* --- SCALE --- */
    for (int i = 0; i < WARMUP; i++) scale_kernel<<<blocks, threads_per_block>>>(dc, db, scalar, n);
    DEVICE_SYNC();
    EVENT_RECORD(ev_start);
    for (int i = 0; i < num_times; i++) scale_kernel<<<blocks, threads_per_block>>>(dc, db, scalar, n);
    EVENT_RECORD(ev_stop);
    EVENT_SYNC(ev_stop);
    { float ms = 0; EVENT_ELAPSED(ms, ev_start, ev_stop); ms /= num_times;
      double gbs = 2.0 * n * sizeof(float) / (ms / 1000.0) / 1e9;
      total_time_all += ms * num_times;
      printf("{\"type\":\"kernel\",\"name\":\"scale_kernel\",\"time_ms\":%.6f,"
             "\"params\":{\"n\":%d,\"threads_per_block\":%d,\"scalar\":%.6f,\"num_times\":%d}}\n",
             (double)ms, n, threads_per_block, (double)scalar, num_times);
      fprintf(stderr, "scale_kernel: %.3f ms, %.2f GB/s\n", ms, gbs); }

    /* --- ADD --- */
    for (int i = 0; i < WARMUP; i++) add_kernel<<<blocks, threads_per_block>>>(da, db, dc, n);
    DEVICE_SYNC();
    EVENT_RECORD(ev_start);
    for (int i = 0; i < num_times; i++) add_kernel<<<blocks, threads_per_block>>>(da, db, dc, n);
    EVENT_RECORD(ev_stop);
    EVENT_SYNC(ev_stop);
    { float ms = 0; EVENT_ELAPSED(ms, ev_start, ev_stop); ms /= num_times;
      double gbs = 3.0 * n * sizeof(float) / (ms / 1000.0) / 1e9;
      total_time_all += ms * num_times;
      printf("{\"type\":\"kernel\",\"name\":\"add_kernel\",\"time_ms\":%.6f,"
             "\"params\":{\"n\":%d,\"threads_per_block\":%d,\"scalar\":%.6f,\"num_times\":%d}}\n",
             (double)ms, n, threads_per_block, (double)scalar, num_times);
      fprintf(stderr, "add_kernel: %.3f ms, %.2f GB/s\n", ms, gbs); }

    /* --- TRIAD --- */
    for (int i = 0; i < WARMUP; i++) triad_kernel<<<blocks, threads_per_block>>>(db, dc, da, scalar, n);
    DEVICE_SYNC();
    EVENT_RECORD(ev_start);
    for (int i = 0; i < num_times; i++) triad_kernel<<<blocks, threads_per_block>>>(db, dc, da, scalar, n);
    EVENT_RECORD(ev_stop);
    EVENT_SYNC(ev_stop);
    { float ms = 0; EVENT_ELAPSED(ms, ev_start, ev_stop); ms /= num_times;
      double gbs = 3.0 * n * sizeof(float) / (ms / 1000.0) / 1e9;
      total_time_all += ms * num_times;
      printf("{\"type\":\"kernel\",\"name\":\"triad_kernel\",\"time_ms\":%.6f,"
             "\"params\":{\"n\":%d,\"threads_per_block\":%d,\"scalar\":%.6f,\"num_times\":%d}}\n",
             (double)ms, n, threads_per_block, (double)scalar, num_times);
      fprintf(stderr, "triad_kernel: %.3f ms, %.2f GB/s\n", ms, gbs); }

    /* --- DOT --- */
    for (int i = 0; i < WARMUP; i++) {
        dot_kernel<<<blocks, threads_per_block, threads_per_block * sizeof(float)>>>(da, db, d_partial, n);
        MEMCPY_D2H(h_partial, d_partial, (size_t)blocks * sizeof(float));
        float sum = 0; for (int j = 0; j < blocks; j++) sum += h_partial[j];
        (void)sum;
    }
    DEVICE_SYNC();
    EVENT_RECORD(ev_start);
    for (int i = 0; i < num_times; i++) {
        dot_kernel<<<blocks, threads_per_block, threads_per_block * sizeof(float)>>>(da, db, d_partial, n);
    }
    EVENT_RECORD(ev_stop);
    EVENT_SYNC(ev_stop);
    { float ms = 0; EVENT_ELAPSED(ms, ev_start, ev_stop); ms /= num_times;
      double gbs = 2.0 * n * sizeof(float) / (ms / 1000.0) / 1e9;
      total_time_all += ms * num_times;
      printf("{\"type\":\"kernel\",\"name\":\"dot_kernel\",\"time_ms\":%.6f,"
             "\"params\":{\"n\":%d,\"threads_per_block\":%d,\"scalar\":%.6f,\"num_times\":%d}}\n",
             (double)ms, n, threads_per_block, (double)scalar, num_times);
      fprintf(stderr, "dot_kernel: %.3f ms, %.2f GB/s\n", ms, gbs); }

    /* Summary */
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[]}\n",
           total_time_all);

    FREE(da); FREE(db); FREE(dc); FREE(d_partial);
    free(ha); free(hb); free(hc); free(h_partial);
    return 0;
}
