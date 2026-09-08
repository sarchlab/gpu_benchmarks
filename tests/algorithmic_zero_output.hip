// Exercise the actual production reset kernels, including exact positive zero
// and unchanged guards. Compile separately for each self-contained benchmark.
#define main benchmark_main
#if defined(TEST_GRAM_ZERO) && !defined(TEST_MG_ZERO)
#include "../tier2/polybench_gramschmidt/polybench_gramschmidt.hip"
#elif defined(TEST_MG_ZERO) && !defined(TEST_GRAM_ZERO)
#include "../tier2/npb_mg/npb_mg.hip"
#else
#error Select exactly one of TEST_GRAM_ZERO or TEST_MG_ZERO
#endif
#undef main

#include <cstdint>
#include <cstdio>
#include <vector>

static bool check_zero(int count, uint32_t poison) {
    std::vector<uint32_t> words(count + 2, poison);
    words.front() = 0x5a123456u;
    words.back() = 0xdeadbeefu;
    float* device;
    const size_t bytes = words.size() * sizeof(uint32_t);
    HIP_CHECK(hipMalloc(&device, bytes));
    HIP_CHECK(hipMemcpy(device, words.data(), bytes, hipMemcpyHostToDevice));
#if defined(TEST_GRAM_ZERO)
    gram_norm_zero<<<1, 1>>>(device + 1);
#else
    coarse_zero<<<(count + 255) / 256, 256>>>(device + 1, count);
#endif
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(words.data(), device, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(device));
    if (words.front() != 0x5a123456u || words.back() != 0xdeadbeefu) {
        fprintf(stderr, "Guard overwritten at count=%d\n", count);
        return false;
    }
    for (int i = 1; i <= count; ++i) {
        if (words[i] != 0u) {
            fprintf(stderr, "count=%d index=%d: expected +0, got 0x%08x\n",
                    count, i - 1, words[i]);
            return false;
        }
    }
    return true;
}

int main() {
#if defined(TEST_GRAM_ZERO)
    const int counts[] = {1};
#else
    // Includes the largest coarse grid in the declared N=512 sweep.
    const int counts[] = {1, 2, 17, 64, 255, 256, 257, 511, 512, 513,
                          32768, 16777216};
#endif
    const uint32_t poisons[] = {0x7fc12345u, 0x80000000u, 0x3f800000u, 0xff800000u};
    int passed = 0;
    for (int count : counts) {
        for (uint32_t poison : poisons) {
            if (!check_zero(count, poison)) return 1;
            ++passed;
        }
    }
    printf("PASS: %d reset output/guard cases\n", passed);
    return 0;
}
