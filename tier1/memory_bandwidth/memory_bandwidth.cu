/**
 * memory_bandwidth.cu
 *
 * Microbenchmark for GPU memory bandwidth using cudaMemcpy in three
 * directions:
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
 *   buffer_size_mb  – optional transfer size in MiB (default 256)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error-checking helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_err));             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static const int    DEFAULT_BUFFER_MB = 256;   // default transfer size
static const int    NUM_ITERATIONS    = 20;    // repetitions for stable avg
static const double BYTES_PER_GB      = 1e9;   // SI gigabyte

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

/** Return elapsed milliseconds between two CUDA events. */
static float elapsed_ms(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

// ---------------------------------------------------------------------------
// Benchmark functions
// ---------------------------------------------------------------------------

/**
 * Benchmark host-to-device bandwidth.
 *
 * @param h_buf   Pinned host source buffer
 * @param d_buf   Device destination buffer
 * @param bytes   Transfer size in bytes
 * @return        Average bandwidth in GB/s
 */
static double benchmark_h2d(const void *h_buf, void *d_buf, size_t bytes)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up
    CUDA_CHECK(cudaMemcpy(d_buf, h_buf, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        CUDA_CHECK(cudaMemcpy(d_buf, h_buf, bytes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = elapsed_ms(start, stop);
    double bw_GBs  = (static_cast<double>(bytes) * NUM_ITERATIONS)
                     / (total_ms * 1e-3) / BYTES_PER_GB;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return bw_GBs;
}

/**
 * Benchmark device-to-host bandwidth.
 *
 * @param d_buf   Device source buffer
 * @param h_buf   Pinned host destination buffer
 * @param bytes   Transfer size in bytes
 * @return        Average bandwidth in GB/s
 */
static double benchmark_d2h(const void *d_buf, void *h_buf, size_t bytes)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up
    CUDA_CHECK(cudaMemcpy(h_buf, d_buf, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        CUDA_CHECK(cudaMemcpy(h_buf, d_buf, bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = elapsed_ms(start, stop);
    double bw_GBs  = (static_cast<double>(bytes) * NUM_ITERATIONS)
                     / (total_ms * 1e-3) / BYTES_PER_GB;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return bw_GBs;
}

/**
 * Benchmark device-to-device bandwidth.
 *
 * @param d_src   Device source buffer
 * @param d_dst   Device destination buffer
 * @param bytes   Transfer size in bytes
 * @return        Average bandwidth in GB/s
 */
static double benchmark_d2d(const void *d_src, void *d_dst, size_t bytes)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up
    CUDA_CHECK(cudaMemcpy(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        CUDA_CHECK(cudaMemcpy(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = elapsed_ms(start, stop);
    double bw_GBs  = (static_cast<double>(bytes) * NUM_ITERATIONS)
                     / (total_ms * 1e-3) / BYTES_PER_GB;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return bw_GBs;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    // Parse optional buffer size argument
    int buffer_mb = DEFAULT_BUFFER_MB;
    if (argc >= 2) {
        buffer_mb = atoi(argv[1]);
        if (buffer_mb <= 0) {
            fprintf(stderr, "Invalid buffer size: %s\n", argv[1]);
            return EXIT_FAILURE;
        }
    }

    size_t bytes = static_cast<size_t>(buffer_mb) * 1024ULL * 1024ULL;

    // Print device info
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    printf("Device: %s (id=%d)\n", prop.name, device_id);
    printf("Transfer size: %d MiB (%zu bytes)\n", buffer_mb, bytes);
    printf("Iterations per test: %d\n\n", NUM_ITERATIONS);

    // Allocate pinned host memory
    void *h_src = nullptr;
    void *h_dst = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_src, bytes));
    CUDA_CHECK(cudaMallocHost(&h_dst, bytes));

    // Initialize host source with a recognizable pattern
    memset(h_src, 0xAB, bytes);
    memset(h_dst, 0x00, bytes);

    // Allocate device memory
    void *d_src = nullptr;
    void *d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));

    // Seed d_src so D2D has real data to copy
    CUDA_CHECK(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));

    // ----- H2D -----
    double h2d_bw = benchmark_h2d(h_src, d_src, bytes);
    printf("H2D bandwidth : %8.2f GB/s\n", h2d_bw);

    // ----- D2H -----
    double d2h_bw = benchmark_d2h(d_src, h_dst, bytes);
    printf("D2H bandwidth : %8.2f GB/s\n", d2h_bw);

    // ----- D2D -----
    double d2d_bw = benchmark_d2d(d_src, d_dst, bytes);
    printf("D2D bandwidth : %8.2f GB/s\n", d2d_bw);

    // Clean up
    CUDA_CHECK(cudaFreeHost(h_src));
    CUDA_CHECK(cudaFreeHost(h_dst));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));

    printf("\nDone.\n");
    return EXIT_SUCCESS;
}
