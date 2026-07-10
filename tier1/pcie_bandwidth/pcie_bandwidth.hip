/**
 * pcie_bandwidth.hip
 *
 * HIP port of the pcie_bandwidth microbenchmark.
 * Compiles for both AMD (ROCm) and NVIDIA (HIP-on-CUDA) targets.
 *
 * Microbenchmark for PCIe transfer time between host and device.
 *
 * Three tests are performed:
 *   1. Unidirectional H2D — pinned host → device
 *   2. Unidirectional D2H — device → pinned host
 *   3. Bidirectional       — concurrent H2D + D2H on two HIP streams
 *
 * Pinned (page-locked) host memory is allocated with hipHostMalloc() to
 * bypass OS paging and saturate the PCIe link. Warmup copies absorb
 * one-time costs (e.g. MI300A charges a large DMA-registration cost to the
 * first transfer touching a fresh pinned buffer); every timed iteration is
 * then recorded individually.
 *
 * Usage:
 *   ./pcie_bandwidth [buffer_size_mb]
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_buffer_size_mb
 *   BENCH_PARAM_transfer_type     (h2d, d2h, bidir, all)
 *   BENCH_PARAM_use_pinned_memory (true, false)
 *
 * Output: one {"type":"kernel",...} JSON line per timed iteration carrying
 * the GPU event time and host wall-clock time, then a summary whose
 * total_time_ms is the SUM of all recorded iterations' GPU event times.
 * Bandwidth is derived downstream from buffer_size_mb and time_ms (bidir
 * moves 2x buffer_size_mb per iteration).
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../../common/bench_timing.h"

// ---------------------------------------------------------------------------
// Error-checking macro
// ---------------------------------------------------------------------------

#define HIP_CHECK(call) BENCH_TIMING_CHECK(call)

// ---------------------------------------------------------------------------
// Constants (defaults)
// ---------------------------------------------------------------------------

static const int DEFAULT_BUFFER_MB = 256;

// ---------------------------------------------------------------------------
// Bidirectional runner
// ---------------------------------------------------------------------------

/** Bidirectional: concurrent H2D + D2H on two blocking streams.
 *
 *  The timing events are recorded on the null stream. Streams created with
 *  hipStreamCreate are blocking (legacy) streams, so the stop event does not
 *  execute until both copies have drained — the event pair therefore spans
 *  exactly the concurrent transfer, with no host-side synchronization inside
 *  the timed bracket. */
static double benchmark_bidir(const void *h_src, void *h_dst,
                              const void *d_src, void *d_dst,
                              size_t bytes, const char *params)
{
    hipStream_t stream_h2d, stream_d2h;
    HIP_CHECK(hipStreamCreate(&stream_h2d));
    HIP_CHECK(hipStreamCreate(&stream_d2h));

    double total_gpu_ms =
        bench_run_once("memcpy_bidir", params, [&]() {
            HIP_CHECK(hipMemcpyAsync(d_dst, h_src, bytes,
                                     hipMemcpyHostToDevice, stream_h2d));
            HIP_CHECK(hipMemcpyAsync(h_dst, d_src, bytes,
                                     hipMemcpyDeviceToHost, stream_d2h));
        });

    HIP_CHECK(hipStreamDestroy(stream_h2d));
    HIP_CHECK(hipStreamDestroy(stream_d2h));
    return total_gpu_ms;
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

    /* Validate string params against their known sets, so the values echoed
     * into the JSON records are always canonical literals (no escaping
     * needed and no malformed output from arbitrary env content). */
    const char* transfer_type = "all";
    env_val = getenv("BENCH_PARAM_transfer_type");
    if (env_val) transfer_type = env_val;
    if (strcmp(transfer_type, "all") != 0 && strcmp(transfer_type, "h2d") != 0 &&
        strcmp(transfer_type, "d2h") != 0 && strcmp(transfer_type, "bidir") != 0) {
        fprintf(stderr, "Invalid transfer_type '%s' (use h2d, d2h, bidir, all)\n", transfer_type);
        return EXIT_FAILURE;
    }

    const char* use_pinned_memory = "true";
    env_val = getenv("BENCH_PARAM_use_pinned_memory");
    if (env_val) use_pinned_memory = env_val;

    bool pinned    = (strcmp(use_pinned_memory, "false") != 0);
    bool run_h2d   = (strcmp(transfer_type, "all") == 0 || strcmp(transfer_type, "h2d") == 0);
    bool run_d2h   = (strcmp(transfer_type, "all") == 0 || strcmp(transfer_type, "d2h") == 0);
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

    /* Setup copy — outside any timed region. */
    HIP_CHECK(hipMemcpy(d_src, h_src, bytes, hipMemcpyHostToDevice));

    char params[192];
    snprintf(params, sizeof(params),
             "\"buffer_size_mb\":%d,\"transfer_type\":\"%s\","
             "\"use_pinned_memory\":\"%s\"",
             buffer_mb, transfer_type, pinned ? "true" : "false");

    double total_gpu_ms = 0.0;
    if (run_h2d) {
        total_gpu_ms += bench_run_memcpy("memcpy_h2d", d_dst, h_src, bytes,
                                         hipMemcpyHostToDevice, params);
    }
    if (run_d2h) {
        total_gpu_ms += bench_run_memcpy("memcpy_d2h", h_dst, d_src, bytes,
                                         hipMemcpyDeviceToHost, params);
    }
    if (run_bidir) {
        total_gpu_ms += benchmark_bidir(h_src, h_dst, d_src, d_dst, bytes,
                                        params);
    }

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

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
