/**
 * pcie_bandwidth.hip
 *
 * HIP port of the pcie_bandwidth microbenchmark.
 * Compiles for both AMD (ROCm) and NVIDIA (HIP-on-CUDA) targets.
 *
 * Microbenchmark for PCIe transfer bandwidth between host and device.
 *
 * Three tests are performed:
 *   1. Unidirectional H2D — pinned host → device, reports GB/s
 *   2. Unidirectional D2H — device → pinned host, reports GB/s
 *   3. Bidirectional       — concurrent H2D + D2H on two HIP streams,
 *                            reports aggregate (sum) GB/s
 *
 * Pinned (page-locked) host memory is allocated with hipHostMalloc() to
 * bypass OS paging and saturate the PCIe link.  Each test is preceded by a
 * warm-up pass and then timed using HIP
 * events.  Results are reported in SI GB/s (1 GB = 10^9 bytes).
 *
 * Usage:
 *   ./pcie_bandwidth [buffer_size_mb]
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_buffer_size_mb
 *   BENCH_PARAM_transfer_type     (h2d, d2h, bidir, all)
 *   BENCH_PARAM_use_pinned_memory (true, false)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__NVCC__) && !__has_include(<hip/hip_runtime.h>)
#include <cuda_runtime.h>
#define hipMalloc               cudaMalloc
#define hipFree                 cudaFree
#define hipMemcpy               cudaMemcpy
#define hipMemcpyAsync          cudaMemcpyAsync
#define hipMemset               cudaMemset
#define hipMemsetAsync          cudaMemsetAsync
#define hipMemcpyHostToDevice   cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost   cudaMemcpyDeviceToHost
#define hipMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define hipMemcpyToSymbol       cudaMemcpyToSymbol
#define hipDeviceSynchronize    cudaDeviceSynchronize
#define hipGetDevice            cudaGetDevice
#define hipGetDeviceProperties  cudaGetDeviceProperties
#define hipGetLastError         cudaGetLastError
#define hipGetErrorString       cudaGetErrorString
#define hipEvent_t              cudaEvent_t
#define hipEventCreate          cudaEventCreate
#define hipEventDestroy         cudaEventDestroy
#define hipEventRecord          cudaEventRecord
#define hipEventSynchronize     cudaEventSynchronize
#define hipEventElapsedTime     cudaEventElapsedTime
#define hipStream_t             cudaStream_t
#define hipStreamCreate         cudaStreamCreate
#define hipStreamDestroy        cudaStreamDestroy
#define hipStreamSynchronize    cudaStreamSynchronize
#define hipDeviceProp_t         cudaDeviceProp
#define hipError_t              cudaError_t
#define hipSuccess              cudaSuccess
#define hipHostMalloc           cudaMallocHost
#define hipHostFree             cudaFreeHost
#define hipHostMallocDefault    cudaHostAllocDefault
#define hipBlockIdx_x           blockIdx.x
#define hipBlockIdx_y           blockIdx.y
#define hipBlockIdx_z           blockIdx.z
#define hipThreadIdx_x          threadIdx.x
#define hipThreadIdx_y          threadIdx.y
#define hipThreadIdx_z          threadIdx.z
#define hipBlockDim_x           blockDim.x
#define hipBlockDim_y           blockDim.y
#define hipBlockDim_z           blockDim.z
#define hipGridDim_x            gridDim.x
#define hipGridDim_y            gridDim.y
#define hipGridDim_z            gridDim.z
#define __HIP_PLATFORM_AMD__    0
#define hipLaunchKernelGGL(kernel, grid, block, smem, stream, ...) \
    kernel<<<grid, block, smem, stream>>>(__VA_ARGS__)
#else
#include <hip/hip_runtime.h>
#endif

// ---------------------------------------------------------------------------
// Error-checking macro
// ---------------------------------------------------------------------------

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t _err = (call);                                              \
        if (_err != hipSuccess) {                                              \
            fprintf(stderr, "HIP error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, hipGetErrorString(_err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Constants (defaults)
// ---------------------------------------------------------------------------

static const int    DEFAULT_BUFFER_MB      = 256;
static const int    DEFAULT_NUM_WARMUP     = 5;
// Report the peak bandwidth over several timed iterations. A single timed copy
// is unreliable on some platforms (e.g. MI300A charges a large one-time DMA
// registration cost to the first transfer that touches a freshly-allocated
// pinned buffer), which collapses the measured D2H bandwidth. Taking the best
// of many iterations filters that one-time cost out, matching rocm-bandwidth-test.
static const int    DEFAULT_NUM_ITERS      = 20;
static const double BYTES_PER_GB           = 1e9;

// ---------------------------------------------------------------------------
// Result struct
// ---------------------------------------------------------------------------

struct BenchResult {
    double bandwidth_gbps;
    double total_ms;
};

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------

static float hip_elapsed_ms(hipEvent_t start, hipEvent_t stop)
{
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    return ms;
}

// ---------------------------------------------------------------------------
// Benchmark: unidirectional Host → Device
// ---------------------------------------------------------------------------

static BenchResult benchmark_h2d(const void *h_buf, void *d_buf,
                                  size_t bytes,
                                  int num_warmup)
{
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for (int w = 0; w < num_warmup; ++w) {
        HIP_CHECK(hipMemcpyAsync(d_buf, h_buf, bytes, hipMemcpyHostToDevice, 0));
    }
    HIP_CHECK(hipDeviceSynchronize());

    float  total_ms = -1.0f;
    for (int it = 0; it < DEFAULT_NUM_ITERS; ++it) {
        HIP_CHECK(hipEventRecord(start, 0));
        HIP_CHECK(hipMemcpyAsync(d_buf, h_buf, bytes, hipMemcpyHostToDevice, 0));
        HIP_CHECK(hipEventRecord(stop, 0));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = hip_elapsed_ms(start, stop);
        if (total_ms < 0.0f || ms < total_ms) total_ms = ms;
    }
    double bw_GBs   = static_cast<double>(bytes)
                      / (total_ms * 1e-3) / BYTES_PER_GB;

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return {bw_GBs, (double)total_ms};
}

// ---------------------------------------------------------------------------
// Benchmark: unidirectional Device → Host
// ---------------------------------------------------------------------------

static BenchResult benchmark_d2h(const void *d_buf, void *h_buf,
                                  size_t bytes,
                                  int num_warmup)
{
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for (int w = 0; w < num_warmup; ++w) {
        HIP_CHECK(hipMemcpyAsync(h_buf, d_buf, bytes, hipMemcpyDeviceToHost, 0));
    }
    HIP_CHECK(hipDeviceSynchronize());

    float  total_ms = -1.0f;
    for (int it = 0; it < DEFAULT_NUM_ITERS; ++it) {
        HIP_CHECK(hipEventRecord(start, 0));
        HIP_CHECK(hipMemcpyAsync(h_buf, d_buf, bytes, hipMemcpyDeviceToHost, 0));
        HIP_CHECK(hipEventRecord(stop, 0));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = hip_elapsed_ms(start, stop);
        if (total_ms < 0.0f || ms < total_ms) total_ms = ms;
    }
    double bw_GBs   = static_cast<double>(bytes)
                      / (total_ms * 1e-3) / BYTES_PER_GB;

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return {bw_GBs, (double)total_ms};
}

// ---------------------------------------------------------------------------
// Benchmark: bidirectional (concurrent H2D + D2H)
// ---------------------------------------------------------------------------

static BenchResult benchmark_bidir(const void *h_src, void *h_dst,
                                    const void *d_src, void *d_dst,
                                    size_t bytes,
                                    int num_warmup)
{
    hipStream_t stream_h2d, stream_d2h;
    HIP_CHECK(hipStreamCreate(&stream_h2d));
    HIP_CHECK(hipStreamCreate(&stream_d2h));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    /* Warm-up. */
    for (int w = 0; w < num_warmup; ++w) {
        HIP_CHECK(hipMemcpyAsync(d_dst, h_src, bytes,
                                 hipMemcpyHostToDevice, stream_h2d));
        HIP_CHECK(hipMemcpyAsync(h_dst, d_src, bytes,
                                 hipMemcpyDeviceToHost, stream_d2h));
        HIP_CHECK(hipStreamSynchronize(stream_h2d));
        HIP_CHECK(hipStreamSynchronize(stream_d2h));
    }

    float  total_ms = -1.0f;
    for (int it = 0; it < DEFAULT_NUM_ITERS; ++it) {
        HIP_CHECK(hipEventRecord(start, 0));
        HIP_CHECK(hipMemcpyAsync(d_dst, h_src, bytes,
                                 hipMemcpyHostToDevice, stream_h2d));
        HIP_CHECK(hipMemcpyAsync(h_dst, d_src, bytes,
                                 hipMemcpyDeviceToHost, stream_d2h));
        HIP_CHECK(hipStreamSynchronize(stream_h2d));
        HIP_CHECK(hipStreamSynchronize(stream_d2h));
        HIP_CHECK(hipEventRecord(stop, 0));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = hip_elapsed_ms(start, stop);
        if (total_ms < 0.0f || ms < total_ms) total_ms = ms;
    }
    double total_bytes = 2.0 * static_cast<double>(bytes);
    double bw_GBs      = total_bytes / (total_ms * 1e-3) / BYTES_PER_GB;

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipStreamDestroy(stream_h2d));
    HIP_CHECK(hipStreamDestroy(stream_d2h));
    return {bw_GBs, (double)total_ms};
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    setvbuf(stdout, NULL, _IOLBF, 0);

    const char* env_val;

    int buffer_mb = DEFAULT_BUFFER_MB;
    env_val = getenv("BENCH_PARAM_buffer_size_mb");
    if (env_val) {
        buffer_mb = atoi(env_val);
    } else if (argc >= 2) {
        buffer_mb = atoi(argv[1]);
    }
    if (buffer_mb <= 0) {
        fprintf(stderr, "Invalid buffer size\n");
        return EXIT_FAILURE;
    }


    const char* transfer_type = "all";
    env_val = getenv("BENCH_PARAM_transfer_type");
    if (env_val) transfer_type = env_val;

    const char* use_pinned_memory = "true";
    env_val = getenv("BENCH_PARAM_use_pinned_memory");
    if (env_val) use_pinned_memory = env_val;

    int num_warmup = DEFAULT_NUM_WARMUP;

    bool pinned   = (strcmp(use_pinned_memory, "false") != 0);
    bool run_h2d  = (strcmp(transfer_type, "all") == 0 || strcmp(transfer_type, "h2d") == 0);
    bool run_d2h  = (strcmp(transfer_type, "all") == 0 || strcmp(transfer_type, "d2h") == 0);
    bool run_bidir = (strcmp(transfer_type, "all") == 0 || strcmp(transfer_type, "bidir") == 0);

    size_t bytes = static_cast<size_t>(buffer_mb) * 1024ULL * 1024ULL;

    /* Print device info to stderr. */
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device          : %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Transfer size   : %d MiB (%zu bytes)\n", buffer_mb, bytes);
    fprintf(stderr, "Transfer type   : %s\n", transfer_type);
    fprintf(stderr, "Pinned memory   : %s\n", use_pinned_memory);
    fprintf(stderr, "Warmup iters    : %d\n\n", num_warmup);

    /* Allocate host memory. */
    void *h_src = nullptr;
    void *h_dst = nullptr;
    if (pinned) {
        HIP_CHECK(hipHostMalloc(&h_src, bytes, hipHostMallocDefault));
        HIP_CHECK(hipHostMalloc(&h_dst, bytes, hipHostMallocDefault));
    } else {
        h_src = malloc(bytes);
        h_dst = malloc(bytes);
        if (!h_src || !h_dst) {
            fprintf(stderr, "Host malloc failed\n");
            return EXIT_FAILURE;
        }
    }

    memset(h_src, 0xAB, bytes);
    memset(h_dst, 0x00, bytes);

    void *d_src = nullptr;
    void *d_dst = nullptr;
    HIP_CHECK(hipMalloc(&d_src, bytes));
    HIP_CHECK(hipMalloc(&d_dst, bytes));

    HIP_CHECK(hipMemcpy(d_src, h_src, bytes, hipMemcpyHostToDevice));

    double total_time_ms = 0.0;
    double h2d_bw = 0.0, d2h_bw = 0.0, bidir_bw = 0.0;

    /* ----- H2D ----- */
    if (run_h2d) {
        fprintf(stderr, "%-30s", "H2D (unidirectional)");
        BenchResult r = benchmark_h2d(h_src, d_dst, bytes, num_warmup);
        h2d_bw = r.bandwidth_gbps;
        
        total_time_ms += r.total_ms;
        fprintf(stderr, ": %8.2f GB/s\n", h2d_bw);

        printf("{\"type\":\"kernel\",\"name\":\"memcpy_h2d\",\"time_ms\":%.6f,"
               "\"params\":{\"buffer_size_mb\":%d,"
               "\"transfer_type\":\"%s\",\"use_pinned_memory\":\"%s\"}}\n",
               r.total_ms, buffer_mb, transfer_type, use_pinned_memory);
    }

    /* ----- D2H ----- */
    if (run_d2h) {
        fprintf(stderr, "%-30s", "D2H (unidirectional)");
        BenchResult r = benchmark_d2h(d_src, h_dst, bytes, num_warmup);
        d2h_bw = r.bandwidth_gbps;
        
        total_time_ms += r.total_ms;
        fprintf(stderr, ": %8.2f GB/s\n", d2h_bw);

        printf("{\"type\":\"kernel\",\"name\":\"memcpy_d2h\",\"time_ms\":%.6f,"
               "\"params\":{\"buffer_size_mb\":%d,"
               "\"transfer_type\":\"%s\",\"use_pinned_memory\":\"%s\"}}\n",
               r.total_ms, buffer_mb, transfer_type, use_pinned_memory);
    }

    /* ----- Bidirectional ----- */
    if (run_bidir) {
        fprintf(stderr, "%-30s", "Bidirectional (H2D + D2H)");
        BenchResult r = benchmark_bidir(h_src, h_dst, d_src, d_dst,
                                         bytes, num_warmup);
        bidir_bw = r.bandwidth_gbps;
        
        total_time_ms += r.total_ms;
        fprintf(stderr, ": %8.2f GB/s  (aggregate)\n", bidir_bw);

        printf("{\"type\":\"kernel\",\"name\":\"memcpy_bidir\",\"time_ms\":%.6f,"
               "\"params\":{\"buffer_size_mb\":%d,"
               "\"transfer_type\":\"%s\",\"use_pinned_memory\":\"%s\"}}\n",
               r.total_ms, buffer_mb, transfer_type, use_pinned_memory);
    }

    /* Summary. */
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,\"metrics\":[", total_time_ms);
    bool first_metric = true;
    if (run_h2d) {
        if (!first_metric) printf(",");
        printf("{\"name\":\"h2d_bandwidth_gbps\",\"value\":%.2f}", h2d_bw);
        first_metric = false;
    }
    if (run_d2h) {
        if (!first_metric) printf(",");
        printf("{\"name\":\"d2h_bandwidth_gbps\",\"value\":%.2f}", d2h_bw);
        first_metric = false;
    }
    if (run_bidir) {
        if (!first_metric) printf(",");
        printf("{\"name\":\"bidir_bandwidth_gbps\",\"value\":%.2f}", bidir_bw);
        first_metric = false;
    }
    printf("]}\n");

    /* Clean up. */
    if (pinned) {
        HIP_CHECK(hipHostFree(h_src));
        HIP_CHECK(hipHostFree(h_dst));
    } else {
        free(h_src);
        free(h_dst);
    }
    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
