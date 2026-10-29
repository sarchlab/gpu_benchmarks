#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#define main benchmark_main
#include "../tier1/shared_mem_bandwidth/hip/shared_mem_bandwidth.hip"
#undef main

static float reference(int elements, int iterations)
{
    std::vector<float> memory(elements, 1.0f);
    float accumulator = 0.0f;
    for (int it = 0; it < iterations; ++it) {
        for (float &value : memory) {
            accumulator = (accumulator + value) * 0.5f;
            value = accumulator;
        }
    }
    return accumulator;
}

int main()
{
    int cases = 0;
    for (int iterations : {64, 65536}) {
        // 160 also covers an accepted shape that does not divide the array.
        for (int threads : {32, 64, 160, 256, 1024}) {
            const int remainder = SMEM_FLOATS % threads;
            const float shorter = reference(SMEM_FLOATS / threads, iterations);
            const float longer = remainder ? reference(SMEM_FLOATS / threads + 1, iterations) : shorter;
            for (int blocks : {1, 2}) {
                for (bool conflict : {false, true}) {
                    const size_t count = static_cast<size_t>(blocks) * threads;
                    std::vector<float> output(count + 2, std::numeric_limits<float>::quiet_NaN());
                    const size_t bytes = output.size() * sizeof(float);
                    float *device = nullptr;
                    HIP_CHECK(hipMalloc(&device, bytes));
                    HIP_CHECK(hipMemcpy(device, output.data(), bytes, hipMemcpyHostToDevice));
                    if (conflict) smem_bw_conflict<<<blocks, threads>>>(device + 1, iterations);
                    else smem_bw_no_conflict<<<blocks, threads>>>(device + 1, iterations);
                    HIP_CHECK(hipGetLastError());
                    HIP_CHECK(hipDeviceSynchronize());
                    HIP_CHECK(hipMemcpy(output.data(), device, bytes, hipMemcpyDeviceToHost));
                    HIP_CHECK(hipFree(device));
                    if (!std::isnan(output.front()) || !std::isnan(output.back())) return 1;
                    for (size_t i = 0; i < count; ++i) {
                        const float actual = output[i + 1];
                        const float expected = (i % threads) < static_cast<size_t>(remainder) ? longer : shorter;
                        if (!std::isfinite(actual) || !std::isfinite(expected) ||
                            actual < 0.0f || actual > 1.0f || std::fabs(actual - expected) > 2e-6f) {
                            fprintf(stderr, "bad/unwritten output %zu: conflict=%d iters=%d threads=%d blocks=%d got=%.9g want=%.9g\n",
                                    i, conflict, iterations, threads, blocks, actual, expected);
                            return 1;
                        }
                    }
                    ++cases;
                }
            }
        }
    }
    printf("PASS: every finite per-thread output and both guards in %d GPU configurations\n", cases);
    return 0;
}
