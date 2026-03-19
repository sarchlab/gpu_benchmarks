/*
 * tango_binomial_options.metal — Metal compute shader for binomial option pricing.
 *
 * Cox-Ross-Rubinstein binomial tree model for American put options.
 * Each threadgroup processes one option using threadgroup shared memory
 * for backward induction through the binomial tree.
 *
 * Buffer layout:
 *   buffer(0): options    — OptionData[N], input option parameters
 *   buffer(1): prices     — float[N], output option prices
 *   buffer(2): params     — uint2 { numOptions, numSteps }
 */

#include <metal_stdlib>
using namespace metal;

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
//   One threadgroup per option. Threads collaborate via threadgroup memory
//   to perform backward induction.
// ---------------------------------------------------------------------------

kernel void binomial_kernel(
    device const OptionData* options  [[ buffer(0) ]],
    device float*            prices   [[ buffer(1) ]],
    constant uint2&          params   [[ buffer(2) ]],
    threadgroup float*       shmem    [[ threadgroup(0) ]],
    uint                     groupId  [[ threadgroup_position_in_grid ]],
    uint                     tid      [[ thread_index_in_threadgroup ]])
{
    uint numSteps = params.y;

    OptionData opt = options[groupId];

    // CRR parameters
    float dt   = opt.T / float(numSteps);
    float u    = exp(opt.sigma * sqrt(dt));       // up factor
    float d    = 1.0f / u;                        // down factor
    float R    = exp(opt.r * dt);                 // risk-free growth
    float Rinv = 1.0f / R;
    float p    = (R - d) / (u - d);               // risk-neutral probability
    float q    = 1.0f - p;

    // Step 1: Compute terminal payoffs
    if (tid <= numSteps) {
        float ST = opt.S * pow(u, float(2 * int(tid) - int(numSteps)));
        float payoff = max(opt.K - ST, 0.0f);
        shmem[tid] = payoff;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Backward induction
    for (uint step = numSteps; step > 0; --step) {
        if (tid < step) {
            float cont = Rinv * (p * shmem[tid + 1] + q * shmem[tid]);
            float ST = opt.S * pow(u, float(2 * int(tid) - int(step - 1)));
            float exercise = max(opt.K - ST, 0.0f);
            shmem[tid] = max(cont, exercise);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes the option price
    if (tid == 0) {
        prices[groupId] = shmem[0];
    }
}
