// tango_binomial_options.hip — Binomial option pricing (HIP)
//
// Cox-Ross-Rubinstein (CRR) binomial tree model for American put option
// pricing. Each threadblock processes one option, threads collaborate on
// backward induction through the tree using shared memory.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./tango_binomial_options [--options N] [--steps S] [--iterations I]
//
//   --options N      Number of options (default: 512)
//   --steps S        Steps per option (default: 1024)
//   --iterations I   Timed iterations (default: 5)
//
// Output (stdout): CSV row — tango_binomial_options,<N>x<steps>,<time_ms>,<opts/sec>
// Output (stderr): human-readable results

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <vector>

// ---------------------------------------------------------------------------
// Error-checking macro
// ---------------------------------------------------------------------------

#define HIP_CHECK(cmd)                                                         \
    do {                                                                       \
        hipError_t _e = (cmd);                                                 \
        if (_e != hipSuccess) {                                                \
            fprintf(stderr, "HIP error %s at %s:%d\n",                        \
                    hipGetErrorString(_e), __FILE__, __LINE__);                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

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
// Option parameters structure
// ---------------------------------------------------------------------------

struct OptionData {
    float S;      // stock price
    float K;      // strike price
    float T;      // time to expiration
    float r;      // risk-free rate
    float sigma;  // volatility
};

// ---------------------------------------------------------------------------
// Binomial option pricing kernel
//   One threadblock per option. Threads collaborate via shared memory to
//   perform backward induction through the binomial tree.
//   NUM_STEPS must be <= blockDim.x (threads per block).
// ---------------------------------------------------------------------------

extern __shared__ float shared_values[];

__global__ void binomial_kernel(
    const OptionData* __restrict__ options,
    float*            __restrict__ prices,
    int               numSteps)
{
    int optIdx = blockIdx.x;
    int tid    = threadIdx.x;

    OptionData opt = options[optIdx];

    // CRR parameters
    float dt = opt.T / (float)numSteps;
    float u  = expf(opt.sigma * sqrtf(dt));        // up factor
    float d  = 1.0f / u;                           // down factor
    float R  = expf(opt.r * dt);                   // risk-free growth
    float Rinv = 1.0f / R;
    float p  = (R - d) / (u - d);                  // risk-neutral probability
    float q  = 1.0f - p;

    // Step 1: Compute terminal payoffs (American put: max(K - S * u^j * d^(N-j), 0))
    // Each thread handles one leaf node
    if (tid <= numSteps) {
        float ST = opt.S * powf(u, (float)(2 * tid - numSteps));
        float payoff = fmaxf(opt.K - ST, 0.0f);
        shared_values[tid] = payoff;
    }
    __syncthreads();

    // Step 2: Backward induction
    for (int step = numSteps; step > 0; --step) {
        if (tid < step) {
            // Expected value discounted
            float cont = Rinv * (p * shared_values[tid + 1] + q * shared_values[tid]);
            // American option: can exercise early
            float ST = opt.S * powf(u, (float)(2 * tid - (step - 1)));
            float exercise = fmaxf(opt.K - ST, 0.0f);
            shared_values[tid] = fmaxf(cont, exercise);
        }
        __syncthreads();
    }

    // Thread 0 writes the option price
    if (tid == 0) {
        prices[optIdx] = shared_values[0];
    }
}

// ---------------------------------------------------------------------------
// CPU reference: binomial option pricing
// ---------------------------------------------------------------------------

static float binomial_cpu(OptionData opt, int numSteps) {
    float dt   = opt.T / (float)numSteps;
    float u    = expf(opt.sigma * sqrtf(dt));
    float d    = 1.0f / u;
    float R    = expf(opt.r * dt);
    float Rinv = 1.0f / R;
    float p    = (R - d) / (u - d);
    float q    = 1.0f - p;

    std::vector<float> vals((size_t)(numSteps + 1));

    // Terminal payoffs
    for (int j = 0; j <= numSteps; ++j) {
        float ST = opt.S * powf(u, (float)(2 * j - numSteps));
        vals[j] = fmaxf(opt.K - ST, 0.0f);
    }

    // Backward induction
    for (int step = numSteps; step > 0; --step) {
        for (int j = 0; j < step; ++j) {
            float cont = Rinv * (p * vals[j + 1] + q * vals[j]);
            float ST = opt.S * powf(u, (float)(2 * j - (step - 1)));
            float exercise = fmaxf(opt.K - ST, 0.0f);
            vals[j] = fmaxf(cont, exercise);
        }
    }

    return vals[0];
}

// ---------------------------------------------------------------------------
// Simple deterministic pseudo-random in range [lo, hi]
// ---------------------------------------------------------------------------

static float randRange(unsigned& seed, float lo, float hi) {
    seed = seed * 1103515245u + 12345u;
    float t = (float)(seed & 0x7fffffffu) / (float)0x7fffffffu;
    return lo + t * (hi - lo);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    int numOptions = parseIntParam(argc, argv, "--options", 512);
    int numSteps   = parseIntParam(argc, argv, "--steps",   1024);
    int iters      = parseIntParam(argc, argv, "--iterations", 5);

    // Clamp numSteps to a reasonable maximum (shared memory limit)
    if (numSteps > 1024) {
        fprintf(stderr, "Warning: clamping steps to 1024 (shared memory limit)\n");
        numSteps = 1024;
    }

    size_t optBytes   = (size_t)numOptions * sizeof(OptionData);
    size_t priceBytes = (size_t)numOptions * sizeof(float);

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Options: %d  |  Steps: %d  |  Iterations: 5 warmup + %d timed\n\n",
            numOptions, numSteps, iters);

    // Host allocation + initialization
    OptionData* h_options = (OptionData*)malloc(optBytes);
    float*      h_prices  = (float*)malloc(priceBytes);

    unsigned seed = 42u;
    for (int i = 0; i < numOptions; ++i) {
        h_options[i].S     = randRange(seed, 5.0f, 200.0f);
        h_options[i].K     = randRange(seed, 1.0f, 300.0f);
        h_options[i].T     = randRange(seed, 0.25f, 10.0f);
        h_options[i].r     = 0.02f;
        h_options[i].sigma = randRange(seed, 0.1f, 1.0f);
    }

    // Device allocation
    OptionData* d_options;
    float*      d_prices;
    HIP_CHECK(hipMalloc(&d_options, optBytes));
    HIP_CHECK(hipMalloc(&d_prices,  priceBytes));
    HIP_CHECK(hipMemcpy(d_options, h_options, optBytes, hipMemcpyHostToDevice));

    // Kernel launch config: one block per option, numSteps+1 threads per block
    int blockSize = numSteps + 1;
    int gridSize  = numOptions;
    size_t sharedBytes = (size_t)(numSteps + 1) * sizeof(float);

    // -------------------------------------------------------------------
    // Warmup (5 iterations)
    // -------------------------------------------------------------------
    for (int w = 0; w < 5; ++w) {
        hipLaunchKernelGGL(binomial_kernel, dim3(gridSize), dim3(blockSize),
                           sharedBytes, 0, d_options, d_prices, numSteps);
        HIP_CHECK(hipDeviceSynchronize());
    }

    // -------------------------------------------------------------------
    // Timed iterations
    // -------------------------------------------------------------------
    hipEvent_t evStart, evStop;
    HIP_CHECK(hipEventCreate(&evStart));
    HIP_CHECK(hipEventCreate(&evStop));

    std::vector<double> times((size_t)iters);
    for (int i = 0; i < iters; ++i) {
        HIP_CHECK(hipEventRecord(evStart, 0));
        hipLaunchKernelGGL(binomial_kernel, dim3(gridSize), dim3(blockSize),
                           sharedBytes, 0, d_options, d_prices, numSteps);
        HIP_CHECK(hipEventRecord(evStop, 0));
        HIP_CHECK(hipEventSynchronize(evStop));

        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, evStart, evStop));
        times[i] = (double)ms;
    }

    HIP_CHECK(hipEventDestroy(evStart));
    HIP_CHECK(hipEventDestroy(evStop));

    // Compute average time
    double sum = 0.0;
    for (int i = 0; i < iters; ++i) sum += times[i];
    double avg_ms = sum / iters;

    // options/sec = numOptions / time_s
    double opts_per_sec = (double)numOptions / (avg_ms * 1e-3);

    // CSV output to stdout
    printf("tango_binomial_options,%dx%d,%.4f,%.4f\n",
           numOptions, numSteps, avg_ms, opts_per_sec);

    // Human-readable output to stderr
    fprintf(stderr, "Average time: %.4f ms\n", avg_ms);
    fprintf(stderr, "Performance:  %.4f options/sec\n", opts_per_sec);

    // -------------------------------------------------------------------
    // Verification
    // -------------------------------------------------------------------
    HIP_CHECK(hipMemcpy(h_prices, d_prices, priceBytes, hipMemcpyDeviceToHost));

    int verifyN = (numOptions < 10) ? numOptions : 10;
    int errors = 0;
    for (int i = 0; i < verifyN; ++i) {
        float ref = binomial_cpu(h_options[i], numSteps);
        float tol = 1e-2f * fabsf(ref) + 1e-4f;
        if (fabsf(h_prices[i] - ref) > tol) {
            fprintf(stderr, "Mismatch at option %d: GPU=%.6f CPU=%.6f (diff=%.6f)\n",
                    i, h_prices[i], ref, fabsf(h_prices[i] - ref));
            errors++;
        }
    }

    if (errors > 0)
        fprintf(stderr, "FAIL: %d errors out of %d verified\n", errors, verifyN);
    else
        fprintf(stderr, "PASS\n");

    HIP_CHECK(hipFree(d_options));
    HIP_CHECK(hipFree(d_prices));
    free(h_options);
    free(h_prices);

    return 0;
}
