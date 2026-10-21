// chai_hsto.cu — Histogram benchmark (Chai collaborative computing pattern)
//
// Computes a 256-bin histogram of N random bytes using shared memory
// privatization and global reduction.
//
// Native CUDA port of ../hip/chai_hsto.hip; compiles with nvcc.
//
// Usage:
//   ./chai_hsto [--size N]
//
//   --size N         Number of input bytes (default: 16777216 = 16M)
//
// Environment variables (override argv):
//   BENCH_PARAM_size, BENCH_PARAM_block_size
//
// Output (stdout): one {"type":"kernel",...} JSON line per timed pass
// carrying the GPU event time (time_ms) and host wall-clock time
// (host_time_ms) of the histogram kernel (the per-pass bin reset is
// untimed), then a summary whose total_time_ms is the SUM of all recorded
// GPU event times. GB/s is derived downstream from time_ms and params.
// Output (stderr): human-readable results

#include "../../../common_cuda/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

static int parseIntParam(int argc, char** argv, const char* name, int defaultVal) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) {
            int v = atoi(argv[i + 1]);
            if (v > 0) return v;
            break;
        }
    }
    return defaultVal;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define NUM_BINS 256
#define BLOCK_SIZE 256

// ---------------------------------------------------------------------------
// Kernel: histogram with shared memory privatization
//
// Each block builds a private histogram in shared memory, then atomically
// adds it to the global histogram.
// ---------------------------------------------------------------------------

__global__ void histogram_kernel(
    const unsigned char* __restrict__ data,
    unsigned int* __restrict__ histo,
    int N)
{
    // Shared memory histogram for this block
    __shared__ unsigned int s_histo[NUM_BINS];

    int tid = threadIdx.x;

    // Initialize shared histogram to zero
    if (tid < NUM_BINS) {
        s_histo[tid] = 0;
    }
    __syncthreads();

    // Each thread processes multiple elements (grid-stride loop)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        atomicAdd(&s_histo[data[i]], 1u);
    }
    __syncthreads();

    // Reduce shared histogram to global histogram
    if (tid < NUM_BINS) {
        if (s_histo[tid] > 0) {
            atomicAdd(&histo[tid], s_histo[tid]);
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference: histogram
// ---------------------------------------------------------------------------

static void histogram_cpu(const unsigned char* data, unsigned int* histo, int N) {
    memset(histo, 0, NUM_BINS * sizeof(unsigned int));
    for (int i = 0; i < N; ++i) {
        histo[data[i]]++;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults from params.json
    int size       = 16777216;
    int block_size = BLOCK_SIZE;

    // Command-line fallback
    size       = parseIntParam(argc, argv, "--size", size);
    block_size = parseIntParam(argc, argv, "--block_size", block_size);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_size");
    if (env_val) size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);

    int N     = size;

    size_t data_bytes  = (size_t)N * sizeof(unsigned char);
    size_t histo_bytes = NUM_BINS * sizeof(unsigned int);

    // Print device info
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Histogram  |  N: %d (%.2f MB)  |  Bins: %d\n\n",
            N, (double)data_bytes / (1024.0 * 1024.0), NUM_BINS);

    // Generate random input data (deterministic seed)
    unsigned char* h_data = (unsigned char*)malloc(data_bytes);
    unsigned int seed = 12345;
    for (int i = 0; i < N; ++i) {
        seed = seed * 1103515245u + 12345u;
        h_data[i] = (unsigned char)((seed >> 16) & 0xFF);
    }

    // Host histogram
    unsigned int* h_histo = (unsigned int*)malloc(histo_bytes);

    // Device allocation
    unsigned char* d_data;
    unsigned int* d_histo;
    CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_histo, histo_bytes));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice));

    // Kernel launch config
    int blockSize = block_size;
    int gridSize  = (N + blockSize - 1) / blockSize;
    // Cap grid size to avoid too many blocks
    if (gridSize > 1024) gridSize = 1024;

    char params[128];
    snprintf(params, sizeof(params),
             "\"size\":%d,\"block_size\":%d",
             size, block_size);

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t lap;

    // One histogram pass: the bins MUST be zeroed before every launch or
    // later passes would accumulate counts on top of earlier ones. The
    // memset stays outside the timed bracket — only the kernel is measured.
    auto run_pass = [&](bool timed) {
        CUDA_CHECK(cudaMemset(d_histo, 0, histo_bytes));
        /* Cold-cache calibration: evict the LLC (and hold the clock high) after
         * the per-pass H2D re-init so the pass's first kernel launch runs cold.
         * No-op when BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();
        if (timed) {
            lap = bench_time_op(&timer, [&]() {
                histogram_kernel<<<dim3(gridSize), dim3(blockSize)>>>(d_data, d_histo, N);
            });
        } else {
            histogram_kernel<<<dim3(gridSize), dim3(blockSize)>>>(d_data, d_histo, N);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    };

    double total_gpu_ms = 0.0;
    run_pass(true);
    bench_emit_iter("histogram_kernel", 0, &lap, params);
    total_gpu_ms += lap.gpu_ms;
    fprintf(stderr, "histogram_kernel: gpu = %.3f ms, host = %.3f ms\n",
            lap.gpu_ms, lap.host_ms);

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    bool verify_ok = true;
    // -------------------------------------------------------------------
    // Verification (compare GPU histogram against CPU reference)
    // -------------------------------------------------------------------
    {
        CUDA_CHECK(cudaMemcpy(h_histo, d_histo, histo_bytes, cudaMemcpyDeviceToHost));

        std::vector<unsigned int> cpu_histo(NUM_BINS);
        histogram_cpu(h_data, cpu_histo.data(), N);

        int errors = 0;
        for (int b = 0; b < NUM_BINS; ++b) {
            if (h_histo[b] != cpu_histo[b]) {
                if (errors < 10) {
                    fprintf(stderr, "Mismatch at bin %d: GPU=%u CPU=%u\n",
                            b, h_histo[b], cpu_histo[b]);
                }
                errors++;
            }
        }

        if (errors > 0) {
            fprintf(stderr, "FAIL: %d bin errors out of %d bins\n",
                    errors, NUM_BINS);
            verify_ok = false;
        } else
            fprintf(stderr, "PASS\n");
    }

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_histo));
    free(h_data);
    free(h_histo);

    return verify_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
