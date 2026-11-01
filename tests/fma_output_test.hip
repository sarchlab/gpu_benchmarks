// GPU-only regression: exercise the production kernel with observable output.
// Compile through either FP32/FP64 Makefile's optional `test` target.
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#define main benchmark_main
#if defined(TEST_FP32)
#include "../tier1/fp32_throughput/hip/fp32_throughput.hip"
using Scalar = float;
#define FMA_KERNEL fp32_fma_kernel
static Scalar reference_fma(Scalar a, Scalar b, Scalar c) { return fmaf(a, b, c); }
static constexpr Scalar multiplier = 1.0000001f;
static constexpr Scalar addend = 0.0000001f;
static constexpr double tolerance = 5e-6;
#elif defined(TEST_FP64)
#include "../tier1/fp64_throughput/hip/fp64_throughput.hip"
using Scalar = double;
#define FMA_KERNEL fp64_fma_kernel
static Scalar reference_fma(Scalar a, Scalar b, Scalar c) { return fma(a, b, c); }
static constexpr Scalar multiplier = 1.0000000001;
static constexpr Scalar addend = 0.0000000001;
static constexpr double tolerance = 1e-12;
#else
#error Define TEST_FP32 or TEST_FP64
#endif
#undef main

static Scalar reference(int local_thread, int fmas)
{
    // Each of four chains receives fmas/4 operations. Allow rounding differences
    // in initialization/reduction; this test checks complete finite output.
    const Scalar base = Scalar(1) + Scalar(local_thread) * Scalar(0.001);
    Scalar accumulators[] = {base, base + Scalar(0.1), base + Scalar(0.2),
                             base + Scalar(0.3)};
    for (int step = 0; step < fmas / 4; ++step) {
        for (auto &a : accumulators) a = reference_fma(a, multiplier, addend);
    }
    return ((accumulators[0] + accumulators[1]) + accumulators[2]) + accumulators[3];
}

int main()
{
    int cases = 0;
    for (int fmas : {256, 1048576}) {
        for (int threads : {1, 32, 64, 256, 1024}) {
            std::vector<Scalar> expected(threads);
            for (int i = 0; i < threads; ++i) expected[i] = reference(i, fmas);
            for (int blocks : {1, 2}) {
                const size_t count = static_cast<size_t>(blocks) * threads;
                std::vector<Scalar> output(count + 2, std::numeric_limits<Scalar>::quiet_NaN());
                const size_t bytes = output.size() * sizeof(Scalar);
                Scalar *device = nullptr;
                HIP_CHECK(hipMalloc(&device, bytes));
                HIP_CHECK(hipMemcpy(device, output.data(), bytes, hipMemcpyHostToDevice));
                FMA_KERNEL<<<blocks, threads>>>(device + 1, fmas);
                HIP_CHECK(hipGetLastError());
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemcpy(output.data(), device, bytes, hipMemcpyDeviceToHost));
                HIP_CHECK(hipFree(device));
                if (!std::isnan(output.front()) || !std::isnan(output.back())) {
                    fprintf(stderr, "output boundary overwritten: fmas=%d blocks=%d threads=%d\n",
                            fmas, blocks, threads);
                    return 1;
                }
                for (size_t i = 0; i < count; ++i) {
                    const double want = expected[i % threads];
                    const double got = output[i + 1];
                    if (!std::isfinite(got) || !std::isfinite(want) ||
                        std::fabs(got - want) > tolerance * std::fabs(want)) {
                        fprintf(stderr, "bad/unwritten thread %zu: fmas=%d blocks=%d threads=%d got=%.17g want=%.17g\n",
                                i, fmas, blocks, threads, got, want);
                        return 1;
                    }
                }
                ++cases;
            }
        }
    }
    printf("PASS: every thread and both output guards in %d GPU configurations\n", cases);
    return 0;
}
