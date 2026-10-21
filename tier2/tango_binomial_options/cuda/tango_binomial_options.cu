// tango_binomial_options.cu — Binomial option pricing (CUDA)
//
// Cox-Ross-Rubinstein (CRR) binomial tree model for American put option
// pricing. Each threadblock processes one option, threads collaborate on
// backward induction through the tree using shared memory.
//
// Native CUDA port of ../hip/tango_binomial_options.hip; compiles with nvcc.
//
// Usage:
//   ./tango_binomial_options [--options N]
//
//   --options N      Number of options (default: 512)
//
// The binomial-tree resolution is fixed at the canonical 1024 steps (not a
// parameter) — it enters the work as steps^2, so sweeping it only multiplies
// runtime without adding a distinct workload.
//
// Environment variables (override argv):
//   BENCH_PARAM_options,
//
// Output (stdout): one {"type":"kernel",...} JSON line for the single timed
// launch carrying the GPU event time and host wall-clock time, then a summary
// whose total_time_ms equals that record's GPU event time.
// Options/sec is derived downstream from time_ms and params.
// Output (stderr): human-readable results

#include "../../../common_cuda/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "cpu_reference.h"

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
// Binomial option pricing kernel
//   One threadblock per option. Threads collaborate via shared memory to
//   perform backward induction through the binomial tree.
//   Uses stride loops so blockDim.x can be smaller than numSteps+1.
// ---------------------------------------------------------------------------

extern __shared__ float shared_values[];

__global__ void binomial_kernel(
    const OptionData* __restrict__ options,
    float*            __restrict__ prices,
    int               numSteps)
{
    int optIdx = blockIdx.x;
    int tid    = threadIdx.x;
    int bdim   = blockDim.x;

    OptionData opt = options[optIdx];

    // CRR parameters
    float dt = opt.T / (float)numSteps;
    float u  = expf(opt.sigma * sqrtf(dt));        // up factor
    float d  = 1.0f / u;                           // down factor
    float R  = expf(opt.r * dt);                   // risk-free growth
    float Rinv = 1.0f / R;
    float p  = (R - d) / (u - d);                  // risk-neutral probability
    float q  = 1.0f - p;

    int numNodes = numSteps + 1;

    // Double-buffer the tree level. A single in-place
    // buffer races: computing node j reads shared_values[j+1] while the thread
    // owning node j+1 overwrites it in the same sweep — only one barrier per
    // level, so on sm_70+ this corrupts even within a warp. `cur` is read-only
    // for the level, `nxt` is written, then they swap.
    float* cur = shared_values;
    float* nxt = shared_values + numNodes;

    // Step 1: Compute terminal payoffs (stride loop)
    for (int j = tid; j < numNodes; j += bdim) {
        float ST = opt.S * powf(u, (float)(2 * j - numSteps));
        float payoff = fmaxf(opt.K - ST, 0.0f);
        cur[j] = payoff;
    }
    __syncthreads();

    // Step 2: Backward induction
    for (int step = numSteps; step > 0; --step) {
        for (int j = tid; j < step; j += bdim) {
            // Expected value discounted
            float cont = Rinv * (p * cur[j + 1] + q * cur[j]);
            // American option: can exercise early
            float ST = opt.S * powf(u, (float)(2 * j - (step - 1)));
            float exercise = fmaxf(opt.K - ST, 0.0f);
            nxt[j] = fmaxf(cont, exercise);
        }
        __syncthreads();
        float* tmp = cur; cur = nxt; nxt = tmp;
    }

    // Thread 0 writes the option price (final result lives in `cur` after the
    // last swap).
    if (tid == 0) {
        prices[optIdx] = cur[0];
    }
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
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults from params.json
    int numOptions = 512;

    // Binomial-tree resolution is fixed at the canonical CRR value and is NOT a
    // swept parameter: it enters the work as steps^2 (per-node powf induction),
    // so sweeping it would multiply runtime into timeout territory while telling
    // us nothing the `options` scaling axis doesn't. The kernel emits it so the
    // tree depth stays visible in the record. 2*(numSteps+1)*4 B ≈ 8 KB shared,
    // well under the 48 KB default.
    const int numSteps = binomial_reference::num_steps;

    // Command-line fallback
    numOptions = parseIntParam(argc, argv, "--options", numOptions);

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_options");
    if (env_val) numOptions = atoi(env_val);

    size_t optBytes   = (size_t)numOptions * sizeof(OptionData);
    size_t priceBytes = (size_t)numOptions * sizeof(float);

    // Print device info
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Options: %d  |  Steps: %d\n\n",
            numOptions, numSteps);

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
    CUDA_CHECK(cudaMalloc(&d_options, optBytes));
    CUDA_CHECK(cudaMalloc(&d_prices,  priceBytes));
    CUDA_CHECK(cudaMemcpy(d_options, h_options, optBytes, cudaMemcpyHostToDevice));

    // Kernel launch config: one block per option
    // Clamp blockSize to 1024 (CUDA max), kernel uses stride loops
    int blockSize = (numSteps + 1 > 1024) ? 1024 : (numSteps + 1);
    int gridSize  = numOptions;
    size_t sharedBytes = (size_t)2 * (numSteps + 1) * sizeof(float);

    char params[128];
    snprintf(params, sizeof(params),
             "\"options\":%d,\"steps\":%d,\"verification\":\"full_cpu_v1\"",
             numOptions, numSteps);

    // Single timed launch (bench_run_once: optional cold-cache warmer, then
    // one bracketed launch). The kernel reads only the immutable option data
    // and fully overwrites d_prices, so every execution performs identical
    // work — no state reset needed.
    double total_gpu_ms =
        bench_run_once("binomial_kernel", params, [&]() {
            binomial_kernel<<<dim3(gridSize), dim3(blockSize), sharedBytes, 0>>>(
                d_options, d_prices, numSteps);
        });

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // -------------------------------------------------------------------
    // Verification
    // -------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(h_prices, d_prices, priceBytes, cudaMemcpyDeviceToHost));

    std::vector<float> reference;
    bool available = true;
    try {
        reference = binomial_reference::prices(h_options, numOptions);
    } catch (const std::exception& exc) {
        fprintf(stderr, "Verification unavailable: CPU reference failed (%s)\n", exc.what());
        available = false;
    }
    int errors = 0;
    if (available) {
        for (int i = 0; i < numOptions; ++i) {
            float ref = reference[i];
            float tol = 1e-2f * fabsf(ref) + 1e-4f;
            if (!std::isfinite(h_prices[i]) || !std::isfinite(ref) ||
                fabsf(h_prices[i] - ref) > tol) {
                if (errors < 10)
                    fprintf(stderr, "Mismatch at option %d: GPU=%.6f CPU=%.6f (diff=%.6f)\n",
                            i, h_prices[i], ref, fabsf(h_prices[i] - ref));
                errors++;
            }
        }
        fprintf(stderr, "Binomial verification: %s (%d prices, %d mismatches)\n",
                errors ? "FAIL" : "PASS", numOptions, errors);
    }

    CUDA_CHECK(cudaFree(d_options));
    CUDA_CHECK(cudaFree(d_prices));
    free(h_options);
    free(h_prices);

    return (!available || errors > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
