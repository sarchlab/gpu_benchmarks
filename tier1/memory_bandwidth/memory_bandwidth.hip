/**
 * memory_bandwidth.hip
 *
 * HIP port of the memory_bandwidth microbenchmark.
 * Compiles for both AMD (ROCm) and NVIDIA (HIP-on-CUDA) targets.
 *
 * Measures GPU memory bandwidth in three directions:
 *   - Host-to-Device (H2D)
 *   - Device-to-Host (D2H)
 *   - Device-to-Device (D2D)
 *
 * Each transfer is timed with HIP events and repeated multiple times to
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

#include <hip/hip_runtime.h>

// ---------------------------------------------------------------------------
// Error-checking helpers
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
// Constants
// ---------------------------------------------------------------------------

static const int    DEFAULT_BUFFER_MB = 256;   // default transfer size
static const int    NUM_ITERATIONS    = 20;    // repetitions for stable avg
static const double BYTES_PER_GB      = 1e9;   // SI gigabyte

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

/** Return elapsed milliseconds between two HIP events. */
static float elapsed_ms(hipEvent_t start, hipEvent_t stop)
{
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
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
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Warm-up
    HIP_CHECK(hipMemcpy(d_buf, h_buf, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipEventRecord(start, 0));
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        HIP_CHECK(hipMemcpy(d_buf, h_buf, bytes, hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));

    float total_ms = elapsed_ms(start, stop);
    double bw_GBs  = (static_cast<double>(bytes) * NUM_ITERATIONS)
                     / (total_ms * 1e-3) / BYTES_PER_GB;

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
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
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Warm-up
    HIP_CHECK(hipMemcpy(h_buf, d_buf, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipEventRecord(start, 0));
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        HIP_CHECK(hipMemcpy(h_buf, d_buf, bytes, hipMemcpyDeviceToHost));
    }
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));

    float total_ms = elapsed_ms(start, stop);
    double bw_GBs  = (static_cast<double>(bytes) * NUM_ITERATIONS)
                     / (total_ms * 1e-3) / BYTES_PER_GB;

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
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
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Warm-up
    HIP_CHECK(hipMemcpy(d_dst, d_src, bytes, hipMemcpyDeviceToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipEventRecord(start, 0));
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        HIP_CHECK(hipMemcpy(d_dst, d_src, bytes, hipMemcpyDeviceToDevice));
    }
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));

    float total_ms = elapsed_ms(start, stop);
    double bw_GBs  = (static_cast<double>(bytes) * NUM_ITERATIONS)
                     / (total_ms * 1e-3) / BYTES_PER_GB;

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
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
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    printf("Device: %s (id=%d)\n", prop.name, device_id);
    printf("Transfer size: %d MiB (%zu bytes)\n", buffer_mb, bytes);
    printf("Iterations per test: %d\n\n", NUM_ITERATIONS);

    // Allocate pinned host memory
    void *h_src = nullptr;
    void *h_dst = nullptr;
    HIP_CHECK(hipHostMalloc(&h_src, bytes, hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(&h_dst, bytes, hipHostMallocDefault));

    // Initialize host source with a recognizable pattern
    memset(h_src, 0xAB, bytes);
    memset(h_dst, 0x00, bytes);

    // Allocate device memory
    void *d_src = nullptr;
    void *d_dst = nullptr;
    HIP_CHECK(hipMalloc(&d_src, bytes));
    HIP_CHECK(hipMalloc(&d_dst, bytes));

    // Seed d_src so D2D has real data to copy
    HIP_CHECK(hipMemcpy(d_src, h_src, bytes, hipMemcpyHostToDevice));

    // ----- H2D -----
    double h2d_bw = benchmark_h2d(h_src, d_src, bytes);
    printf("H2D bandwidth : %8.2f GB/s\n", h2d_bw);

    // ----- D2H -----
    double d2h_bw = benchmark_d2h(d_src, h_dst, bytes);
    printf("D2H bandwidth : %8.2f GB/s\n", d2h_bw);

    // ----- D2D -----
    double d2d_bw = benchmark_d2d(d_src, d_dst, bytes);
    printf("D2D bandwidth : %8.2f GB/s\n", d2d_bw);

    // CSV output (machine-parseable): benchmark_name,size_mb,time_ms,gbs
    double h2d_time_ms = (static_cast<double>(bytes) / (h2d_bw * 1e9)) * 1e3;
    double d2h_time_ms = (static_cast<double>(bytes) / (d2h_bw * 1e9)) * 1e3;
    double d2d_time_ms = (static_cast<double>(bytes) / (d2d_bw * 1e9)) * 1e3;
    printf("memory_bandwidth_h2d,%d,%.3f,%.2f\n", buffer_mb, h2d_time_ms, h2d_bw);
    printf("memory_bandwidth_d2h,%d,%.3f,%.2f\n", buffer_mb, d2h_time_ms, d2h_bw);
    printf("memory_bandwidth_d2d,%d,%.3f,%.2f\n", buffer_mb, d2d_time_ms, d2d_bw);

    // Clean up
    HIP_CHECK(hipHostFree(h_src));
    HIP_CHECK(hipHostFree(h_dst));
    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));

    printf("\nDone.\n");
    return EXIT_SUCCESS;
}
