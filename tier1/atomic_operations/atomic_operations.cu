/**
 * atomic_operations.cu
 *
 * atomic_operations microbenchmark.
  *
 * Microbenchmark for atomic operation throughput on global memory:
 *   - atomicAdd
 *   - atomicCAS (compare-and-swap)
 *   - atomicMin
 *
 * Strategy:
 *   All threads perform N atomic operations on a single global memory
 *   location.  This creates maximum contention and measures the hardware
 *   atomic unit throughput.  The achieved Giga-Operations/s is:
 *
 *       Gops = (num_threads * ops_per_thread) / elapsed_seconds / 1e9
 *
 * Usage:
 *   ./atomic_operations [ops_per_thread]
 *
 *   ops_per_thread  – optional number of atomic ops per thread
 *                     (default 1024).
 *
 * Environment variables (override argv):
 *   BENCH_PARAM_ops_per_thread
 *   BENCH_PARAM_num_blocks
 *   BENCH_PARAM_threads_per_block
 *   BENCH_PARAM_atomic_op   ("atomicAdd", "atomicCAS", "atomicMin")
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>


// ---------------------------------------------------------------------------
// Error-checking helper
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
// Tuning constants (defaults)
// ---------------------------------------------------------------------------

static const int DEFAULT_OPS_PER_THREAD  = 1024;
static const int DEFAULT_THREADS_PER_BLK = 256;
static const int DEFAULT_NUM_BLOCKS      = 1024;
static const int DEFAULT_NUM_WARMUP      = 5;

// ---------------------------------------------------------------------------
// Atomic kernels
// ---------------------------------------------------------------------------

__global__ void atomicAdd_kernel(unsigned int *counter, int ops_per_thread)
{
    for (int i = 0; i < ops_per_thread; ++i) {
        atomicAdd(counter, 1u);
    }
}

__global__ void atomicCAS_kernel(unsigned int *counter, int ops_per_thread)
{
    for (int i = 0; i < ops_per_thread; ++i) {
        unsigned int old = *counter;
        atomicCAS(counter, old, old + 1);
    }
}

__global__ void atomicMin_kernel(unsigned int *counter, int ops_per_thread)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < ops_per_thread; ++i) {
        atomicMin(counter, tid + (unsigned int)i);
    }
}

// ---------------------------------------------------------------------------
// Benchmark helper
// ---------------------------------------------------------------------------

typedef void (*KernelFunc)(unsigned int *, int);

/**
 * Run a single atomic benchmark: warmup + timed iterations.
 * Returns average kernel time in ms and Gops/s via out parameter.
 */
static void run_benchmark(const char *kernel_name,
                          KernelFunc kernel, unsigned int *d_counter,
                          int ops_per_thread, int num_blocks,
                          int threads_per_block, const char *atomic_op,
                          int num_warmup,
                          double *out_avg_ms, double *out_gops)
{
    /* Reset counter. */
    unsigned int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_counter, &zero, sizeof(unsigned int),
                        cudaMemcpyHostToDevice));

    /* Warm-up launches. */
    for (int w = 0; w < num_warmup; ++w) {
        kernel<<<dim3(num_blocks), dim3(threads_per_block), 0, 0>>>(d_counter, ops_per_thread);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Reset counter before timed run. */
    CUDA_CHECK(cudaMemcpy(d_counter, &zero, sizeof(unsigned int),
                        cudaMemcpyHostToDevice));

    /* Create CUDA timing events. */
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    /* Timed kernel launch. */
    CUDA_CHECK(cudaEventRecord(ev_start, 0));
        kernel<<<dim3(num_blocks), dim3(threads_per_block), 0, 0>>>(d_counter, ops_per_thread);
    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    long long num_threads = static_cast<long long>(num_blocks)
                            * threads_per_block;
    long long total_ops   = num_threads
                            * static_cast<long long>(ops_per_thread);
    double elapsed_s      = static_cast<double>(elapsed_ms) * 1e-3;
    double gops           = static_cast<double>(total_ops) / elapsed_s / 1e9;

    

    /* Human-readable output to stderr. */
    fprintf(stderr, "%-12s : %.3f ms total, %.3f ms/kernel, %.2f Gops/s\n",
            kernel_name, (double)elapsed_ms, (double)elapsed_ms, gops);

    /* JSON-lines output to stdout: kernel event */
    printf("{\"type\":\"kernel\",\"name\":\"%s\",\"time_ms\":%.6f,"
           "\"params\":{\"ops_per_thread\":%d,\"num_blocks\":%d,"
           "\"threads_per_block\":%d,\"atomic_op\":\"%s\"}}\n",
           kernel_name, (double)elapsed_ms, ops_per_thread, num_blocks,
           threads_per_block, atomic_op);

    *out_avg_ms = (double)elapsed_ms;
    *out_gops   = gops;

    /* Clean up events. */
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    /* Line-buffer stdout so JSON lines are flushed promptly. */
    setvbuf(stdout, NULL, _IOLBF, 0);

    const char* env_val;

    /* Read parameters: env vars take priority, then argv, then defaults. */
    int ops_per_thread = DEFAULT_OPS_PER_THREAD;
    env_val = getenv("BENCH_PARAM_ops_per_thread");
    if (env_val) {
        ops_per_thread = atoi(env_val);
    } else if (argc >= 2) {
        ops_per_thread = atoi(argv[1]);
    }
    if (ops_per_thread <= 0) {
        fprintf(stderr, "Invalid ops_per_thread\n");
        return EXIT_FAILURE;
    }

    int num_blocks = DEFAULT_NUM_BLOCKS;
    env_val = getenv("BENCH_PARAM_num_blocks");
    if (env_val) num_blocks = atoi(env_val);

    int threads_per_block = DEFAULT_THREADS_PER_BLK;
    env_val = getenv("BENCH_PARAM_threads_per_block");
    if (env_val) threads_per_block = atoi(env_val);

    const char* atomic_op = "atomicAdd";
    env_val = getenv("BENCH_PARAM_atomic_op");
    if (env_val) atomic_op = env_val;
    if (strcmp(atomic_op, "atomicAdd") != 0 &&
        strcmp(atomic_op, "atomicCAS") != 0 &&
        strcmp(atomic_op, "atomicMin") != 0) {
        fprintf(stderr, "Error: unsupported atomic_op '%s'\n", atomic_op);
        return EXIT_FAILURE;
    }

    int num_warmup = DEFAULT_NUM_WARMUP;


    /* Print device information to stderr. */
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    fprintf(stderr, "Device          : %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "CUs             : %d\n", prop.multiProcessorCount);
    fprintf(stderr, "Grid            : %d blocks × %d threads = %d total threads\n",
            num_blocks, threads_per_block,
            num_blocks * threads_per_block);
    fprintf(stderr, "Ops / thread    : %d\n", ops_per_thread);
    fprintf(stderr, "Atomic op       : %s\n", atomic_op);
    fprintf(stderr, "Warmup iters    : %d\n\n", num_warmup);

    /* Allocate a single unsigned int on the device as the shared counter. */
    unsigned int *d_counter = nullptr;
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(unsigned int)));

    double total_time_ms = 0.0;
    double max_gops = 0.0;

    bool run_add = (strcmp(atomic_op, "atomicAdd") == 0);
    bool run_cas = (strcmp(atomic_op, "atomicCAS") == 0);
    bool run_min = (strcmp(atomic_op, "atomicMin") == 0);

    if (run_add) {
        double avg_ms, gops;
        run_benchmark("atomicAdd_kernel",
                      (KernelFunc)atomicAdd_kernel, d_counter,
                      ops_per_thread, num_blocks, threads_per_block, atomic_op,
                      num_warmup, &avg_ms, &gops);
        total_time_ms += avg_ms;
        if (gops > max_gops) max_gops = gops;
    }

    if (run_cas) {
        double avg_ms, gops;
        run_benchmark("atomicCAS_kernel",
                      (KernelFunc)atomicCAS_kernel, d_counter,
                      ops_per_thread, num_blocks, threads_per_block, atomic_op,
                      num_warmup, &avg_ms, &gops);
        total_time_ms += avg_ms;
        if (gops > max_gops) max_gops = gops;
    }

    if (run_min) {
        double avg_ms, gops;
        run_benchmark("atomicMin_kernel",
                      (KernelFunc)atomicMin_kernel, d_counter,
                      ops_per_thread, num_blocks, threads_per_block, atomic_op,
                      num_warmup, &avg_ms, &gops);
        total_time_ms += avg_ms;
        if (gops > max_gops) max_gops = gops;
    }

    /* Summary */
    printf("{\"type\":\"summary\",\"total_time_ms\":%.6f,"
           "\"metrics\":[{\"name\":\"giga_ops_per_sec\",\"value\":%.2f}]}\n",
           total_time_ms, max_gops);

    /* Clean up. */
    CUDA_CHECK(cudaFree(d_counter));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
