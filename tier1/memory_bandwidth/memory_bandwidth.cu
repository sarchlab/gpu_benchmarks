/**
 * memory_bandwidth.cu
 *
 * CUDA native version of the memory_bandwidth microbenchmark.
 * CUDA native implementation for NVIDIA GPUs.
 *
 * Measures GPU memory bandwidth in three directions:
 *   - Host-to-Device (H2D)
 *   - Device-to-Host (D2H)
 *   - Device-to-Device (D2D)
 *
 * Each transfer is timed with CUDA events and repeated multiple times to
 * obtain a stable average.  Results are reported in GB/s (10^9 bytes/s).
 *
 * Usage:
 *   ./memory_bandwidth [buffer_size_mb]
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_buffer_size_mb
 *   BENCH_PARAM_transfer_type     (h2d, d2h, d2d, all)
 *   BENCH_PARAM_use_pinned_memory (true, false)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error-checking helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                        \
    do {                                                                       \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Constants (defaults)
// ---------------------------------------------------------------------------

static const int    DEFAULT_BUFFER_MB      = 256;
static const int    DEFAULT_NUM_WARMUP     = 5;
static const double BYTES_PER_GB           = 1e9;

// ---------------------------------------------------------------------------
// Result struct
// ---------------------------------------------------------------------------

struct BenchResult {
    double bandwidth_gbps;
    double total_ms;
};

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

static float cuda_elapsed_ms(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

// ---------------------------------------------------------------------------
// Benchmark functions
// ---------------------------------------------------------------------------

static BenchResult benchmark_h2d(const void *h_buf, void *d_buf,
                                  size_t bytes,
                                  int num_warmup)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int w = 0; w < num_warmup; ++w) {
        CUDA_CHECK(cudaMemcpy(d_buf, h_buf, bytes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(d_buf, h_buf, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = cuda_elapsed_ms(start, stop);
    double bw_GBs  = static_cast<double>(bytes)
                     / (total_ms * 1e-3) / BYTES_PER_GB;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return {bw_GBs, (double)total_ms};
}

static BenchResult benchmark_d2h(const void *d_buf, void *h_buf,
                                  size_t bytes,
                                  int num_warmup)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int w = 0; w < num_warmup; ++w) {
        CUDA_CHECK(cudaMemcpy(h_buf, d_buf, bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(h_buf, d_buf, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = cuda_elapsed_ms(start, stop);
    double bw_GBs  = static_cast<double>(bytes)
                     / (total_ms * 1e-3) / BYTES_PER_GB;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return {bw_GBs, (double)total_ms};
}

static BenchResult benchmark_d2d(const void *d_src, void *d_dst,
                                  size_t bytes,
                                  int num_warmup)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int w = 0; w < num_warmup; ++w) {
        CUDA_CHECK(cudaMemcpy(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = cuda_elapsed_ms(start, stop);
    double bw_GBs  = static_cast<double>(bytes)
                     / (total_ms * 1e-3) / BYTES_PER_GB;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return {bw_GBs, (double)total_ms};
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    setvbuf(stdout, NULL, _IOLBF, 0);

    const char* env_val;

    /* Read parameters: env vars take priority, then argv, then defaults. */
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

    bool pinned = (strcmp(use_pinned_memory, "false") != 0);
    bool run_h2d = (strcmp(transfer_type, "all") == 0 || strcmp(transfer_type, "h2d") == 0);
    bool run_d2h = (strcmp(transfer_type, "all") == 0 || strcmp(transfer_type, "d2h") == 0);
    bool run_d2d = (strcmp(transfer_type, "all") == 0 || strcmp(transfer_type, "d2d") == 0);

    size_t bytes = static_cast<size_t>(buffer_mb) * 1024ULL * 1024ULL;

    /* Print device info to stderr. */
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Transfer size: %d MiB (%zu bytes)\n", buffer_mb, bytes);
    fprintf(stderr, "Transfer type: %s\n", transfer_type);
    fprintf(stderr, "Pinned memory: %s\n", use_pinned_memory);
    fprintf(stderr, "Warmup iters: %d\n\n", num_warmup);

    /* Allocate host memory. */
    void *h_src = nullptr;
    void *h_dst = nullptr;
    if (pinned) {
        CUDA_CHECK(cudaHostAlloc(&h_src, bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_dst, bytes, cudaHostAllocDefault));
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

    /* Allocate device memory. */
    void *d_src = nullptr;
    void *d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));

    CUDA_CHECK(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));

    double total_time_ms = 0.0;
    double h2d_bw = 0.0, d2h_bw = 0.0, d2d_bw = 0.0;

    /* ----- H2D ----- */
    if (run_h2d) {
        BenchResult r = benchmark_h2d(h_src, d_src, bytes, num_warmup);
        h2d_bw = r.bandwidth_gbps;
        
        total_time_ms += r.total_ms;
        fprintf(stderr, "H2D bandwidth : %8.2f GB/s\n", h2d_bw);

        printf("{\"type\":\"kernel\",\"name\":\"memcpy_h2d\",\"time_ms\":%.6f,"
               "\"params\":{\"buffer_size_mb\":%d,"
               "\"transfer_type\":\"%s\",\"use_pinned_memory\":\"%s\"}}\n",
               r.total_ms, buffer_mb, transfer_type, use_pinned_memory);
    }

    /* ----- D2H ----- */
    if (run_d2h) {
        BenchResult r = benchmark_d2h(d_src, h_dst, bytes, num_warmup);
        d2h_bw = r.bandwidth_gbps;
        
        total_time_ms += r.total_ms;
        fprintf(stderr, "D2H bandwidth : %8.2f GB/s\n", d2h_bw);

        printf("{\"type\":\"kernel\",\"name\":\"memcpy_d2h\",\"time_ms\":%.6f,"
               "\"params\":{\"buffer_size_mb\":%d,"
               "\"transfer_type\":\"%s\",\"use_pinned_memory\":\"%s\"}}\n",
               r.total_ms, buffer_mb, transfer_type, use_pinned_memory);
    }

    /* ----- D2D ----- */
    if (run_d2d) {
        BenchResult r = benchmark_d2d(d_src, d_dst, bytes, num_warmup);
        d2d_bw = r.bandwidth_gbps;
        
        total_time_ms += r.total_ms;
        fprintf(stderr, "D2D bandwidth : %8.2f GB/s\n", d2d_bw);

        printf("{\"type\":\"kernel\",\"name\":\"memcpy_d2d\",\"time_ms\":%.6f,"
               "\"params\":{\"buffer_size_mb\":%d,"
               "\"transfer_type\":\"%s\",\"use_pinned_memory\":\"%s\"}}\n",
               r.total_ms, buffer_mb, transfer_type, use_pinned_memory);
    }

    /* Summary with bandwidth metrics for each direction tested. */
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
    if (run_d2d) {
        if (!first_metric) printf(",");
        printf("{\"name\":\"d2d_bandwidth_gbps\",\"value\":%.2f}", d2d_bw);
        first_metric = false;
    }
    printf("]}\n");

    /* Clean up. */
    if (pinned) {
        CUDA_CHECK(cudaFreeHost(h_src));
        CUDA_CHECK(cudaFreeHost(h_dst));
    } else {
        free(h_src);
        free(h_dst);
    }
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
