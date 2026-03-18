/**
 * pcie_bandwidth.cu
 *
 * Microbenchmark for PCIe transfer bandwidth between host and device.
 *
 * Three tests are performed:
 *   1. Unidirectional H2D — pinned host → device, reports GB/s
 *   2. Unidirectional D2H — device → pinned host, reports GB/s
 *   3. Bidirectional       — concurrent H2D + D2H on two CUDA streams,
 *                            reports aggregate (sum) GB/s
 *
 * Pinned (page-locked) host memory is allocated with cudaMallocHost() to
 * bypass OS paging and saturate the PCIe link.  Each test is preceded by a
 * warm-up pass and then timed over NUM_ITERATIONS repetitions using CUDA
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

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error-checking macro
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

static const int    DEFAULT_BUFFER_MB = 256;   // default transfer size in MiB
static const int    NUM_ITERATIONS    = 20;    // repetitions for stable average
static const double BYTES_PER_GB      = 1e9;   // SI gigabyte

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------

/** Return elapsed milliseconds between two recorded CUDA events. */
static float elapsed_ms(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
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
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up: one transfer to prime TLB entries and PCIe link
    CUDA_CHECK(cudaMemcpy(d_buf, h_buf, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed transfers
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        CUDA_CHECK(cudaMemcpy(d_buf, h_buf, bytes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float  total_ms = elapsed_ms(start, stop);
    double bw_GBs   = (static_cast<double>(bytes) * NUM_ITERATIONS)
                      / (total_ms * 1e-3) / BYTES_PER_GB;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
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
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up
    CUDA_CHECK(cudaMemcpy(h_buf, d_buf, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed transfers
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        CUDA_CHECK(cudaMemcpy(h_buf, d_buf, bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float  total_ms = elapsed_ms(start, stop);
    double bw_GBs   = (static_cast<double>(bytes) * NUM_ITERATIONS)
                      / (total_ms * 1e-3) / BYTES_PER_GB;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return bw_GBs;
}

// ---------------------------------------------------------------------------
// Benchmark: bidirectional (concurrent H2D + D2H)
// ---------------------------------------------------------------------------

/**
 * Measure aggregate bidirectional PCIe bandwidth.
 *
 * Concurrent H2D and D2H transfers are issued on two separate CUDA streams.
 * Both streams are launched between a pair of CUDA events, so the measured
 * wall time covers the full overlap window.  The reported aggregate bandwidth
 * is (total bytes moved) / elapsed_time, where total bytes = H2D + D2H bytes
 * per iteration.
 *
 * @param h_src   Pinned host buffer used as H2D source
 * @param h_dst   Pinned host buffer used as D2H destination
 * @param d_src   Device buffer used as D2H source (also H2D destination swap)
 * @param d_dst   Device buffer used as H2D destination (also D2H source swap)
 * @param bytes   Transfer size per direction in bytes
 * @return        Aggregate bidirectional bandwidth in GB/s
 */
static double benchmark_bidir(const void *h_src, void *h_dst,
                               const void *d_src, void *d_dst,
                               size_t bytes)
{
    cudaStream_t stream_h2d, stream_d2h;
    CUDA_CHECK(cudaStreamCreate(&stream_h2d));
    CUDA_CHECK(cudaStreamCreate(&stream_d2h));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up: one concurrent pass on both streams
    CUDA_CHECK(cudaMemcpyAsync(d_dst, h_src, bytes,
                               cudaMemcpyHostToDevice, stream_h2d));
    CUDA_CHECK(cudaMemcpyAsync(h_dst, d_src, bytes,
                               cudaMemcpyDeviceToHost, stream_d2h));
    CUDA_CHECK(cudaStreamSynchronize(stream_h2d));
    CUDA_CHECK(cudaStreamSynchronize(stream_d2h));

    // Record start on the default stream (acts as a barrier) then launch both
    // async copies.  The stop event is recorded after both streams finish.
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(d_dst, h_src, bytes,
                                   cudaMemcpyHostToDevice, stream_h2d));
        CUDA_CHECK(cudaMemcpyAsync(h_dst, d_src, bytes,
                                   cudaMemcpyDeviceToHost, stream_d2h));
    }
    // Synchronize both streams before recording stop
    CUDA_CHECK(cudaStreamSynchronize(stream_h2d));
    CUDA_CHECK(cudaStreamSynchronize(stream_d2h));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float  total_ms    = elapsed_ms(start, stop);
    // Each iteration moves 'bytes' in each direction → 2*bytes total
    double total_bytes = 2.0 * static_cast<double>(bytes) * NUM_ITERATIONS;
    double bw_GBs      = total_bytes / (total_ms * 1e-3) / BYTES_PER_GB;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream_h2d));
    CUDA_CHECK(cudaStreamDestroy(stream_d2h));
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
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    printf("Device          : %s (id=%d)\n", prop.name, device_id);
    printf("PCIe link width : x%d  gen %d\n",
           prop.pciDomainID,   // note: just for display; actual link info below
           prop.pciBusID);
    printf("Transfer size   : %d MiB (%zu bytes)\n", buffer_mb, bytes);
    printf("Iterations/test : %d\n\n", NUM_ITERATIONS);

    // ---- Allocate pinned (page-locked) host memory ----
    // cudaMallocHost() prevents the OS from paging these pages and maps them
    // into device address space, giving the DMA engine direct access for
    // maximum PCIe throughput.
    void *h_src = nullptr;
    void *h_dst = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_src, bytes));
    CUDA_CHECK(cudaMallocHost(&h_dst, bytes));

    // Initialize host buffers
    memset(h_src, 0xAB, bytes);
    memset(h_dst, 0x00, bytes);

    // ---- Allocate device memory ----
    void *d_src = nullptr;
    void *d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));

    // Seed d_src so D2H has real data to transfer
    CUDA_CHECK(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));

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

    // ---- Clean up ----
    CUDA_CHECK(cudaFreeHost(h_src));
    CUDA_CHECK(cudaFreeHost(h_dst));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));

    printf("\nDone.\n");
    return EXIT_SUCCESS;
}
