// Independent pinned-operator checks of the actual production GPU kernel.
#define main benchmark_main
#include "../tier2/polybench_2dconv/hip/polybench_2dconv.hip"
#undef main

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

static bool check_convolution(int rows, int cols, int impulse) {
    const size_t cells = (size_t)rows * cols;
    const float guard = -12345.125f;
    std::vector<float> input(cells + 2, guard), output(cells + 2, guard);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            input[1 + (size_t)i * cols + j] = impulse >= 0 ? 0.0f :
                (float)((i * 17 + j * 13) % 101 - 50) / 8.0f;
            if (i > 0 && i < rows-1 && j > 0 && j < cols-1)
                output[1 + (size_t)i * cols + j] = std::numeric_limits<float>::quiet_NaN();
        }
    }
    if (impulse >= 0) input[1 + impulse] = 1.0f;  // nine independent3x3 basis inputs
    float *d_input, *d_output;
    const size_t bytes = input.size() * sizeof(float);
    HIP_CHECK(hipMalloc(&d_input, bytes));
    HIP_CHECK(hipMalloc(&d_output, bytes));
    HIP_CHECK(hipMemcpy(d_input, input.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_output, output.data(), bytes, hipMemcpyHostToDevice));
    convolution2D_kernel<<<dim3((cols + 15)/16, (rows + 15)/16), dim3(16,16)>>>(
        d_input + 1, d_output + 1, rows, cols);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(output.data(), d_output, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    if (output.front() != guard || output.back() != guard) {
        fprintf(stderr, "Allocation guard overwritten\n");
        return false;
    }
    // This row/column mapping is independently transcribed from the pinned
    // CPU conv2D expression. It also matches the pinned GPU operator.
    const double weights[3][3] = {{0.2, 0.5, -0.8}, {-0.3, 0.6, -0.9}, {0.4, 0.7, 0.1}};
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const float actual = output[1 + (size_t)i * cols + j];
            if (i == 0 || i == rows-1 || j == 0 || j == cols-1) {
                if (actual != guard) { fprintf(stderr, "Border overwritten at %d,%d\n", i,j); return false; }
                continue;
            }
            double expected = 0.0;
            for (int di = -1; di <= 1; ++di)
                for (int dj = -1; dj <= 1; ++dj)
                    expected += weights[di+1][dj+1] * input[1 + (size_t)(i+di) * cols + j+dj];
            // Each unit impulse yields exactly one representable float
            // coefficient: no reduction-rounding tolerance is needed there.
            if (!std::isfinite(actual) || !std::isfinite(expected) ||
                (impulse >= 0 && actual != (float)expected) ||
                std::fabs((double)actual - expected) > 1e-5 * std::max(1.0, std::fabs(expected))) {
                fprintf(stderr, "%dx%d impulse=%d at %d,%d: got %.9g expected %.12g\n",
                        rows,cols,impulse,i,j,actual,expected);
                return false;
            }
        }
    }
    printf("PASS: %dx%d impulse=%d\n", rows,cols,impulse);
    return true;
}

int main(int argc, char** argv) {
    if (argc == 3 && strcmp(argv[1], "--size") == 0) {
        int n = atoi(argv[2]);
        if (n < 3 || n > 16384) return 2;
        return check_convolution(n,n,-1) ? 0 : 1;
    }
    if (argc == 3 && strcmp(argv[1], "--impulse") == 0) {
        int i = atoi(argv[2]);
        if (i < 0 || i > 8) return 2;
        return check_convolution(3,3,i) ? 0 : 1;
    }
    if (argc != 1) return 2;
    for (int i = 0; i < 9; ++i) if (!check_convolution(3,3,i)) return 1;
    const int shapes[][2] = {{3,3},{3,17},{17,3},{15,16},{16,15},
                            {17,31},{31,17},{33,65},{64,64},{128,256}};
    for (const auto& shape : shapes)
        if (!check_convolution(shape[0],shape[1],-1)) return 1;
    return 0;
}
