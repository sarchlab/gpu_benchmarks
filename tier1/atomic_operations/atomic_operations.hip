/**
 * atomic_operations.hip
 *
 * HIP port of the atomic_operations microbenchmark.
 * Compiles for both AMD (ROCm hipcc) and NVIDIA (nvcc -x cu) targets.
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
 */

#include <cstdio>
#include <cstdlib>

#include <hip/hip_runtime.h>

// ---------------------------------------------------------------------------
// Error-checking helper
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
// Tuning constants
// ---------------------------------------------------------------------------

/** Default number of atomic operations per thread per kernel invocation. */
static const int DEFAULT_OPS_PER_THREAD = 1024;

/** Number of threads per block. */
static const int THREADS_PER_BLOCK = 256;

/** Number of thread blocks. */
static const int NUM_BLOCKS = 1024;

/** Number of warm-up iterations. */
static const int NUM_WARMUP = 5;

/** Number of timed iterations. */
static const int NUM_ITERATIONS = 5;

// ---------------------------------------------------------------------------
// Atomic kernels
// ---------------------------------------------------------------------------

/**
 * Each thread performs ops_per_thread atomicAdd operations on a single
 * global memory location.
 */
__global__ void atomicAdd_kernel(unsigned int *counter, int ops_per_thread)
{
    for (int i = 0; i < ops_per_thread; ++i) {
        atomicAdd(counter, 1u);
    }
}

/**
 * Each thread performs ops_per_thread atomicCAS operations on a single
 * global memory location.  Uses the standard CAS-loop pattern:
 * read current value, attempt to swap to value+1.
 */
__global__ void atomicCAS_kernel(unsigned int *counter, int ops_per_thread)
{
    for (int i = 0; i < ops_per_thread; ++i) {
        unsigned int old = *counter;
        atomicCAS(counter, old, old + 1);
    }
}

/**
 * Each thread performs ops_per_thread atomicMin operations on a single
 * global memory location.
 */
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

/**
 * Run a single atomic benchmark: warmup + timed iterations.
 *
 * @param name          Human-readable name of the operation.
 * @param csv_op        Short name for CSV output (e.g. "atomicAdd").
 * @param kernel        Function pointer to the __global__ kernel.
 * @param d_counter     Device pointer to the shared counter.
 * @param ops_per_thread Number of atomic ops each thread executes.
 */
typedef void (*KernelFunc)(unsigned int *, int);

static void run_benchmark(const char *name, const char *csv_op,
                          KernelFunc kernel, unsigned int *d_counter,
                          int ops_per_thread)
{
    /* Reset counter. */
    unsigned int zero = 0;
    HIP_CHECK(hipMemcpy(d_counter, &zero, sizeof(unsigned int),
                        hipMemcpyHostToDevice));

    /* Warm-up launches. */
    for (int w = 0; w < NUM_WARMUP; ++w) {
        hipLaunchKernelGGL(kernel, dim3(NUM_BLOCKS), dim3(THREADS_PER_BLOCK),
                           0, 0, d_counter, ops_per_thread);
    }
    HIP_CHECK(hipDeviceSynchronize());

    /* Reset counter before timed run. */
    HIP_CHECK(hipMemcpy(d_counter, &zero, sizeof(unsigned int),
                        hipMemcpyHostToDevice));

    /* Create HIP timing events. */
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    /* Timed benchmark loop. */
    HIP_CHECK(hipEventRecord(ev_start, 0));
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        hipLaunchKernelGGL(kernel, dim3(NUM_BLOCKS), dim3(THREADS_PER_BLOCK),
                           0, 0, d_counter, ops_per_thread);
    }
    HIP_CHECK(hipEventRecord(ev_stop, 0));
    HIP_CHECK(hipEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    /* Compute throughput.
     *   total_ops = iterations * num_threads * ops_per_thread
     */
    long long num_threads = static_cast<long long>(NUM_BLOCKS)
                            * THREADS_PER_BLOCK;
    long long total_ops   = static_cast<long long>(NUM_ITERATIONS)
                            * num_threads
                            * static_cast<long long>(ops_per_thread);
    double elapsed_s      = static_cast<double>(elapsed_ms) * 1e-3;
    double gops           = static_cast<double>(total_ops) / elapsed_s / 1e9;

    double avg_ms = elapsed_ms / NUM_ITERATIONS;

    /* Human-readable output to stderr. */
    fprintf(stderr, "%-12s : %.3f ms total, %.3f ms/kernel, %.2f Gops/s\n",
            name, (double)elapsed_ms, avg_ms, gops);

    /* CSV output to stdout: atomic_ops,<op>_global,<time_ms>,<Gops> */
    printf("atomic_ops,%s_global,%.3f,%.2f\n", csv_op, avg_ms, gops);

    /* Clean up events. */
    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    /* Parse optional argument. */
    int ops_per_thread = DEFAULT_OPS_PER_THREAD;
    if (argc >= 2) {
        ops_per_thread = atoi(argv[1]);
        if (ops_per_thread <= 0) {
            fprintf(stderr, "Invalid ops_per_thread: %s\n", argv[1]);
            return EXIT_FAILURE;
        }
    }

    /* Print device information to stderr. */
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));

    fprintf(stderr, "Device          : %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "CUs             : %d\n", prop.multiProcessorCount);
    fprintf(stderr, "Grid            : %d blocks × %d threads = %d total threads\n",
            NUM_BLOCKS, THREADS_PER_BLOCK,
            NUM_BLOCKS * THREADS_PER_BLOCK);
    fprintf(stderr, "Ops / thread    : %d\n", ops_per_thread);
    fprintf(stderr, "Warmup iters    : %d\n", NUM_WARMUP);
    fprintf(stderr, "Timed iters     : %d\n\n", NUM_ITERATIONS);

    /* Allocate a single unsigned int on the device as the shared counter. */
    unsigned int *d_counter = nullptr;
    HIP_CHECK(hipMalloc(&d_counter, sizeof(unsigned int)));

    /* Run each atomic operation benchmark. */
    run_benchmark("atomicAdd", "atomicAdd",
                  (KernelFunc)atomicAdd_kernel, d_counter, ops_per_thread);
    run_benchmark("atomicCAS", "atomicCAS",
                  (KernelFunc)atomicCAS_kernel, d_counter, ops_per_thread);
    run_benchmark("atomicMin", "atomicMin",
                  (KernelFunc)atomicMin_kernel, d_counter, ops_per_thread);

    /* Clean up. */
    HIP_CHECK(hipFree(d_counter));

    fprintf(stderr, "\nDone.\n");
    return EXIT_SUCCESS;
}
