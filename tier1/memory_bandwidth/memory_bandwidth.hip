/**
 * memory_bandwidth.hip
 *
 * HIP port of the memory_bandwidth microbenchmark.
 * Compiles for both AMD (ROCm) and NVIDIA (HIP-on-CUDA) targets.
 *
 * Measures GPU memory copy time in three directions:
 *   - Host-to-Device (H2D)
 *   - Device-to-Host (D2H)
 *   - Device-to-Device (D2D)
 *
 * Each hipMemcpyAsync is timed with HIP events (device time) and host
 * wall-clock. Warmup copies absorb one-time costs (e.g. MI300A charges a
 * large DMA-registration cost to the first transfer touching a fresh pinned
 * buffer); every timed iteration is then recorded individually, so any
 * residual first-iteration effect stays visible in the data.
 *
 * Usage:
 *   ./memory_bandwidth [buffer_size_mb]
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_buffer_size_mb
 *   BENCH_PARAM_transfer_type     (h2d, d2h, d2d, all)
 *   BENCH_PARAM_use_pinned_memory (true, false)
 *   BENCH_PARAM_num_warmup        (warmup copies per direction, default 3)
 *   BENCH_PARAM_num_iters         (timed copies per direction, each recorded, default 7)
 *
 * Output: one {"type":"kernel",...} JSON line per timed copy carrying the
 * GPU event time and host wall-clock time, then a summary whose
 * total_time_ms is the SUM of all recorded copies' GPU event times.
 * Bandwidth is derived downstream from buffer_size_mb and time_ms.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../../common/bench_timing.h"

// ---------------------------------------------------------------------------
// Error-checking helpers
// ---------------------------------------------------------------------------

#define HIP_CHECK(call) BENCH_TIMING_CHECK(call)

// ---------------------------------------------------------------------------
// Constants (defaults)
// ---------------------------------------------------------------------------

static const int DEFAULT_BUFFER_MB = 256;

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

    /* Validate string params against their known sets, so the values echoed
     * into the JSON records are always canonical literals (no escaping
     * needed and no malformed output from arbitrary env content). */
    const char* transfer_type = "all";
    env_val = getenv("BENCH_PARAM_transfer_type");
    if (env_val) transfer_type = env_val;
    if (strcmp(transfer_type, "all") != 0 && strcmp(transfer_type, "h2d") != 0 &&
        strcmp(transfer_type, "d2h") != 0 && strcmp(transfer_type, "d2d") != 0) {
        fprintf(stderr, "Invalid transfer_type '%s' (use h2d, d2h, d2d, all)\n", transfer_type);
        return EXIT_FAILURE;
    }

    const char* use_pinned_memory = "true";
    env_val = getenv("BENCH_PARAM_use_pinned_memory");
    if (env_val) use_pinned_memory = env_val;

    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

    bool pinned = (strcmp(use_pinned_memory, "false") != 0);
    bool run_h2d = (strcmp(transfer_type, "all") == 0 || strcmp(transfer_type, "h2d") == 0);
    bool run_d2h = (strcmp(transfer_type, "all") == 0 || strcmp(transfer_type, "d2h") == 0);
    bool run_d2d = (strcmp(transfer_type, "all") == 0 || strcmp(transfer_type, "d2d") == 0);

    size_t bytes = static_cast<size_t>(buffer_mb) * 1024ULL * 1024ULL;

    /* Print device info to stderr. */
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Transfer size: %d MiB (%zu bytes)\n", buffer_mb, bytes);
    fprintf(stderr, "Transfer type: %s\n", transfer_type);
    fprintf(stderr, "Pinned memory: %s\n", use_pinned_memory);
    fprintf(stderr, "Warmup / iters: %d / %d per direction\n\n", num_warmup, num_iters);

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

    /* Allocate device memory. */
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
        total_gpu_ms += bench_run_memcpy("memcpy_h2d", d_src, h_src, bytes,
                                         hipMemcpyHostToDevice,
                                         num_warmup, num_iters, params);
    }
    if (run_d2h) {
        total_gpu_ms += bench_run_memcpy("memcpy_d2h", h_dst, d_src, bytes,
                                         hipMemcpyDeviceToHost,
                                         num_warmup, num_iters, params);
    }
    if (run_d2d) {
        total_gpu_ms += bench_run_memcpy("memcpy_d2d", d_dst, d_src, bytes,
                                         hipMemcpyDeviceToDevice,
                                         num_warmup, num_iters, params);
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
