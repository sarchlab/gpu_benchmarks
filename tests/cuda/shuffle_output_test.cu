// GPU-only oracle for the production shuffle kernels, using closed-form lane
// origins rather than another implementation of the device loops.
#include <algorithm>
#include <cstdio>
#include <vector>

#define main benchmark_main
#include "../../tier1/warp_shuffle/cuda/warp_shuffle.cu"
#undef main

int main()
{
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    const int width = prop.warpSize;
    if (width != 32 && width != 64) return 1;
    std::vector<int> shapes = {width, 3 * width, 64, 128, 256, 512, 1024};
    std::sort(shapes.begin(), shapes.end());
    shapes.erase(std::unique(shapes.begin(), shapes.end()), shapes.end());
    const KernelFunc kernels[] = {shfl_xor_kernel, shfl_kernel,
                                  shfl_up_kernel, shfl_down_kernel};
    const char *names[] = {"shfl_xor", "shfl", "shfl_up", "shfl_down"};
    int cases = 0;
    for (int operations : {1, 3, 4, 6, 65536}) {
        for (int threads : shapes) {
            if (threads > prop.maxThreadsPerBlock || threads > prop.maxThreadsDim[0]) continue;
            for (int blocks : {1, 2}) {
                const int count = blocks * threads;
                for (int variant = 0; variant < 4; ++variant) {
                    std::vector<int> output(count + 2, -1);
                    const size_t bytes = output.size() * sizeof(int);
                    int *device = nullptr;
                    CUDA_CHECK(cudaMalloc(&device, bytes));
                    CUDA_CHECK(cudaMemcpy(device, output.data(), bytes, cudaMemcpyHostToDevice));
                    kernels[variant]<<<blocks, threads>>>(device + 1, operations);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());
                    CUDA_CHECK(cudaMemcpy(output.data(), device, bytes, cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaFree(device));
                    if (output.front() != -1 || output.back() != -1) {
                        fprintf(stderr, "output guard overwritten\n");
                        return 1;
                    }
                    for (int id = 0; id < count; ++id) {
                        const int lane = id % width;
                        int origin = id ^ (operations & 1); // XOR and indexed swap pairs.
                        if (variant == 2) origin = id - lane + std::max(0, lane - operations);
                        if (variant == 3) origin = id - lane + std::min(width - 1, lane + operations);
                        if (output[id + 1] != origin) {
                            fprintf(stderr, "%s: bad/unwritten thread %d: ops=%d width=%d "
                                    "threads=%d blocks=%d got=%d want=%d\n",
                                    names[variant], id, operations, width, threads, blocks,
                                    output[id + 1], origin);
                            return 1;
                        }
                    }
                    ++cases;
                }
            }
        }
    }
    printf("PASS: all lane origins and output guards in %d GPU cases (native width %d)\n", cases, width);
    return 0;
}
