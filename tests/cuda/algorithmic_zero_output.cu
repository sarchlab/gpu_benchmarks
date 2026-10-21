// Exercise the actual production reset kernels, including exact positive zero
// and unchanged guards. Compile separately for each self-contained benchmark.
#define main benchmark_main
#if defined(TEST_GRAM_ZERO) && !defined(TEST_MG_ZERO)
#include "../../tier2/polybench_gramschmidt/cuda/polybench_gramschmidt.cu"
#elif defined(TEST_MG_ZERO) && !defined(TEST_GRAM_ZERO)
#include "../../tier2/npb_mg/cuda/npb_mg.cu"
#else
#error Select exactly one of TEST_GRAM_ZERO or TEST_MG_ZERO
#endif
#undef main

#include <cstdint>
#include <cstdio>
#include <vector>

#if defined(TEST_GRAM_ZERO)
using Word=uint32_t; using Value=float;
#else
using Word=uint64_t; using Value=double;
#endif
static bool check_zero(int count, Word poison) {
    std::vector<Word> words(count + 2, poison);
    words.front() = 0x5a123456u;
    words.back() = 0xdeadbeefu;
    Value* device;
    const size_t bytes = words.size() * sizeof(Word);
    CUDA_CHECK(cudaMalloc(&device, bytes));
    CUDA_CHECK(cudaMemcpy(device, words.data(), bytes, cudaMemcpyHostToDevice));
#if defined(TEST_GRAM_ZERO)
    gram_norm_zero<<<1, 1>>>(device + 1);
#else
    coarse_zero<<<(count + 255) / 256, 256>>>(device + 1, count);
#endif
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(words.data(), device, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device));
    if (words.front() != 0x5a123456u || words.back() != 0xdeadbeefu) {
        fprintf(stderr, "Guard overwritten at count=%d\n", count);
        return false;
    }
    for (int i = 1; i <= count; ++i) {
        if (words[i] != 0u) {
            fprintf(stderr, "count=%d index=%d: expected +0, got 0x%llx\n",
                    count, i - 1, static_cast<unsigned long long>(words[i]));
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
                          32768, 17173512};
#endif
#if defined(TEST_GRAM_ZERO)
    const Word poisons[] = {0x7fc12345u, 0x80000000u, 0x3f800000u, 0xff800000u};
#else
    const Word poisons[] = {UINT64_C(0x7ff8123456789abc), UINT64_C(0x8000000000000000),
                            UINT64_C(0x3ff0000000000000), UINT64_C(0xfff0000000000000)};
#endif
    int passed = 0;
    for (int count : counts) {
        for (Word poison : poisons) {
            if (!check_zero(count, poison)) return 1;
            ++passed;
        }
    }
    printf("PASS: %d reset output/guard cases\n", passed);
    return 0;
}
