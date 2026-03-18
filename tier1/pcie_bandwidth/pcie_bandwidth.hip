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
 * warm-up pass and then timed over NUM_ITERATIONS repetitions using HIP
 * events.  Results are reported in SI GB/s (1 GB = 10^9 bytes).
 *
 * Usage:
 *   ./pcie_bandwidth [buffer_size_mb]
 *
 *   buffer_size_mb  – optional transfer size in MiB (default 256)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <hip/hip_runtime.h>

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
// Constants
// ---------------------------------------------------------------------------

static const int    DEFAULT_BUFFER_MB = 256;   // default transfer size in MiB
static const int    NUM_ITERATIONS    = 20;    // repetitions for stable average
static const double BYTES_PER_GB      = 1e9;   // SI gigabyte

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------

/** Return elapsed milliseconds between two recorded HIP events. */
static float elapsed_ms(hipEvent_t start, hipEvent_t stop)
{
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    return ms;
}

// ---------------------------------------------------------------------------
// Benchmark: unidirectional Host → Device
// ---------------------------------------------------------------------------

/**
 * Measure PCIe H2D (Host-to-Device) bandwidth.
 *
 * @param h_buf   Pinned host source buffer
 * @param d_buf   Device destination buffer
 * @param bytes   Transfer size in bytes
 * @return        Average H2D bandwidth in GB/s
 */
static double benchmark_h2d(const void *h_buf, void *d_buf, size_t bytes)
{
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Warm-up: one transfer to prime TLB entries and PCIe link
    HIP_CHECK(hipMemcpy(d_buf, h_buf, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    // Timed transfers
    HIP_CHECK(hipEventRecord(start, 0));
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        HIP_CHECK(hipMemcpy(d_buf, h_buf, bytes, hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));

    float  total_ms = elapsed_ms(start, stop);
    double bw_GBs   = (static_cast<double>(bytes) * NUM_ITERATIONS)
                      / (total_ms * 1e-3) / BYTES_PER_GB;

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return bw_GBs;
}

// ---------------------------------------------------------------------------
// Benchmark: unidirectional Device → Host
// ---------------------------------------------------------------------------

/**
 * Measure PCIe D2H (Device-to-Host) bandwidth.
 *
 * @param d_buf   Device source buffer
 * @param h_buf   Pinned host destination buffer
 * @param bytes   Transfer size in bytes
 * @return        Average D2H bandwidth in GB/s
 */
static double benchmark_d2h(const void *d_buf, void *h_buf, size_t bytes)
{
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Warm-up
    HIP_CHECK(hipMemcpy(h_buf, d_buf, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Timed transfers
    HIP_CHECK(hipEventRecord(start, 0));
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        HIP_CHECK(hipMemcpy(h_buf, d_buf, bytes, hipMemcpyDeviceToHost));
    }
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));

    float  total_ms = elapsed_ms(start, stop);
    double bw_GBs   = (static_cast<double>(bytes) * NUM_ITERATIONS)
                      / (total_ms * 1e-3) / BYTES_PER_GB;

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return bw_GBs;
}

// ---------------------------------------------------------------------------
// Benchmark: bidirectional (concurrent H2D + D2H)
// ---------------------------------------------------------------------------

/**
 * Measure aggregate bidirectional PCIe bandwidth.
 *
 * Concurrent H2D and D2H transfers are issued on two separate HIP streams.
 * Both streams are launched between a pair of HIP events, so the measured
 * wall time covers the full overlap window.  The reported aggregate bandwidth
 * is (total bytes moved) / elapsed_time, where total bytes = H2D + D2H bytes
 * per iteration.
 *
 * @param h_src   Pinned host buffer used as H2D source
 * @param h_dst   Pinned host buffer used as D2H destination
 * @param d_src   Device buffer used as D2H source
 * @param d_dst   Device buffer used as H2D destination
 * @param bytes   Transfer size per direction in bytes
 * @return        Aggregate bidirectional bandwidth in GB/s
 */
static double benchmark_bidir(const void *h_src, void *h_dst,
                               const void *d_src, void *d_dst,
                               size_t bytes)
{
    hipStream_t stream_h2d, stream_d2h;
    HIP_CHECK(hipStreamCreate(&stream_h2d));
    HIP_CHECK(hipStreamCreate(&stream_d2h));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Warm-up: one concurrent pass on both streams
    HIP_CHECK(hipMemcpyAsync(d_dst, h_src, bytes,
                             hipMemcpyHostToDevice, stream_h2d));
    HIP_CHECK(hipMemcpyAsync(h_dst, d_src, bytes,
                             hipMemcpyDeviceToHost, stream_d2h));
    HIP_CHECK(hipStreamSynchronize(stream_h2d));
    HIP_CHECK(hipStreamSynchronize(stream_d2h));

    // Record start, launch both async copies, then record stop
    HIP_CHECK(hipEventRecord(start, 0));
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        HIP_CHECK(hipMemcpyAsync(d_dst, h_src, bytes,
                                 hipMemcpyHostToDevice, stream_h2d));
        HIP_CHECK(hipMemcpyAsync(h_dst, d_src, bytes,
                                 hipMemcpyDeviceToHost, stream_d2h));
    }
    // Synchronize both streams before recording stop
    HIP_CHECK(hipStreamSynchronize(stream_h2d));
    HIP_CHECK(hipStreamSynchronize(stream_d2h));
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));

    float  total_ms    = elapsed_ms(start, stop);
    // Each iteration moves 'bytes' in each direction → 2*bytes total
    double total_bytes = 2.0 * static_cast<double>(bytes) * NUM_ITERATIONS;
    double bw_GBs      = total_bytes / (total_ms * 1e-3) / BYTES_PER_GB;

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipStreamDestroy(stream_h2d));
    HIP_CHECK(hipStreamDestroy(stream_d2h));
    return bw_GBs;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    // Parse optional buffer size argument (in MiB)
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
    printf("Device          : %s (id=%d)\n", prop.name, device_id);
    printf("Transfer size   : %d MiB (%zu bytes)\n", buffer_mb, bytes);
    printf("Iterations/test : %d\n\n", NUM_ITERATIONS);

    // ---- Allocate pinned (page-locked) host memory ----
    // hipHostMalloc() prevents the OS from paging these pages and maps them
    // into device address space, giving the DMA engine direct access for
    // maximum PCIe throughput.
    void *h_src = nullptr;
    void *h_dst = nullptr;
    HIP_CHECK(hipHostMalloc(&h_src, bytes, hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(&h_dst, bytes, hipHostMallocDefault));

    // Initialize host buffers
    memset(h_src, 0xAB, bytes);
    memset(h_dst, 0x00, bytes);

    // ---- Allocate device memory ----
    void *d_src = nullptr;
    void *d_dst = nullptr;
    HIP_CHECK(hipMalloc(&d_src, bytes));
    HIP_CHECK(hipMalloc(&d_dst, bytes));

    // Seed d_src so D2H has real data to transfer
    HIP_CHECK(hipMemcpy(d_src, h_src, bytes, hipMemcpyHostToDevice));

    // ---- Run benchmarks ----

    printf("%-30s", "H2D (unidirectional)");
    fflush(stdout);
    double h2d_bw = benchmark_h2d(h_src, d_dst, bytes);
    printf(": %8.2f GB/s\n", h2d_bw);

    printf("%-30s", "D2H (unidirectional)");
    fflush(stdout);
    double d2h_bw = benchmark_d2h(d_src, h_dst, bytes);
    printf(": %8.2f GB/s\n", d2h_bw);

    printf("%-30s", "Bidirectional (H2D + D2H)");
    fflush(stdout);
    double bidir_bw = benchmark_bidir(h_src, h_dst, d_src, d_dst, bytes);
    printf(": %8.2f GB/s  (aggregate)\n", bidir_bw);

    // CSV output (machine-parseable): benchmark_name,size_mb,time_ms,gbs
    double h2d_time_ms   = (static_cast<double>(bytes) / (h2d_bw   * 1e9)) * 1e3;
    double d2h_time_ms   = (static_cast<double>(bytes) / (d2h_bw   * 1e9)) * 1e3;
    double bidir_time_ms = (2.0 * static_cast<double>(bytes) / (bidir_bw * 1e9)) * 1e3;
    printf("pcie_bandwidth_h2d,%d,%.3f,%.2f\n",   buffer_mb, h2d_time_ms,   h2d_bw);
    printf("pcie_bandwidth_d2h,%d,%.3f,%.2f\n",   buffer_mb, d2h_time_ms,   d2h_bw);
    printf("pcie_bandwidth_bidir,%d,%.3f,%.2f\n", buffer_mb, bidir_time_ms, bidir_bw);

    // ---- Clean up ----
    HIP_CHECK(hipHostFree(h_src));
    HIP_CHECK(hipHostFree(h_dst));
    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));

    printf("\nDone.\n");
    return EXIT_SUCCESS;
}
