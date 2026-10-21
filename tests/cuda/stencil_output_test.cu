// Independent full-state checks of the actual production timestep sequence.
#define main benchmark_main
#include "../../tier2/parboil_stencil/cuda/parboil_stencil.cu"
#undef main

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <limits>
#include <vector>

static bool check_stencil(int nx, int ny, int nz, int steps, int impulse = -1) {
    const size_t cells = (size_t)nx * ny * nz;
    const float guard = -12345.125f;
    std::vector<float> input(cells + 2, guard), second(cells + 2, guard);
    const auto index = [=](int x, int y, int z) { return ((size_t)z * ny + y) * nx + x; };
    const auto border = [=](int x, int y, int z) {
        return x == 0 || x == nx-1 || y == 0 || y == ny-1 || z == 0 || z == nz-1;
    };
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                size_t i = index(x,y,z) + 1;
                input[i] = impulse >= 0 ? 0.0f : (float)((x*17 + y*13 + z*7) % 67 - 33) / 16.0f;
            }
    // Center and six independent face-neighbor unit impulses in a3x3x3 grid.
    const int basis[] = {13,22,4,16,10,14,12};
    if (impulse >= 0) input[1 + basis[impulse]] = 1.0f;
    second = input;
    for (int z = 1; z < nz-1; ++z)
        for (int y = 1; y < ny-1; ++y)
            for (int x = 1; x < nx-1; ++x)
                second[1 + index(x,y,z)] = std::numeric_limits<float>::quiet_NaN();

    float *a, *b;
    size_t bytes = input.size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMemcpy(a, input.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b, second.data(), bytes, cudaMemcpyHostToDevice));
    bench_lap_t elapsed;
    float* latest = run_stencil_pass(a+1, b+1, nx, ny, nz, steps, &elapsed);
    if (latest != (steps % 2 ? b+1 : a+1)) {
        CUDA_CHECK(cudaFree(a));
        CUDA_CHECK(cudaFree(b));
        fprintf(stderr, "Wrong final buffer\n");
        return false;
    }
    std::vector<float> output(cells+2), previous(cells+2);
    CUDA_CHECK(cudaMemcpy(output.data(), latest-1, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(previous.data(), latest == a+1 ? b : a, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    if (!std::isfinite(elapsed.gpu_ms) || elapsed.gpu_ms < 0 ||
        !std::isfinite(elapsed.host_ms) || elapsed.host_ms < 0) return false;
    if (output.front() != guard || output.back() != guard ||
        previous.front() != guard || previous.back() != guard) {
        fprintf(stderr, "Allocation guard overwritten\n"); return false;
    }

    // Pinned CPU operator, computed independently in double. Preserve its
    // float coefficient definitions but do not reuse production constants.
    const double center = (float)(1.0f/6.0f), neighbor = (float)(1.0f/6.0f/6.0f);
    std::vector<double> ref(input.begin()+1, input.end()-1), next = ref;
    for (int t = 0; t < steps; ++t) {
        for (int z = 1; z < nz-1; ++z)
            for (int y = 1; y < ny-1; ++y)
                for (int x = 1; x < nx-1; ++x) {
                    double sum = ref[index(x+1,y,z)] + ref[index(x-1,y,z)]
                               + ref[index(x,y+1,z)] + ref[index(x,y-1,z)]
                               + ref[index(x,y,z+1)] + ref[index(x,y,z-1)];
                    next[index(x,y,z)] = neighbor * sum - center * ref[index(x,y,z)];
                }
        ref.swap(next);
    }
    double max_error = 0.0;
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                size_t i = index(x,y,z);
                float actual = output[1+i];
                if (border(x,y,z)) {
                    if (actual != input[1+i] || previous[1+i] != input[1+i]) {
                        fprintf(stderr, "Boundary changed at %d,%d,%d\n",x,y,z); return false;
                    }
                } else if (!std::isfinite(actual) || !std::isfinite(ref[i]) ||
                           (impulse >= 0 && actual != (float)ref[i]) ||
                           std::fabs((double)actual-ref[i]) > std::max(1e-7, 2e-5*std::fabs(ref[i]))) {
                    fprintf(stderr, "%dx%dx%d steps%d impulse%d at %d,%d,%d: got %.9g expected %.12g\n",
                            nx,ny,nz,steps,impulse,x,y,z,actual,ref[i]);
                    return false;
                }
                max_error = std::max(max_error, std::fabs((double)actual-ref[i]));
            }
    printf("PASS: %dx%dx%d steps%d impulse%d max_abs_error=%.9g\n", nx,ny,nz,steps,impulse,max_error);
    return true;
}

static int bounded_int(const char* text, int lo, int hi) {
    char* end;
    errno = 0;
    long value = strtol(text, &end, 10);
    return errno || end == text || *end || value < lo || value > hi ? -1 : (int)value;
}

int main(int argc, char** argv) {
    if (argc == 4 && strcmp(argv[1], "--case") == 0) {
        int n = bounded_int(argv[2],3,512), steps = bounded_int(argv[3],1,100);
        if (n < 0 || steps < 0) return 2;
        return check_stencil(n,n,n,steps) ? 0 : 1;
    }
    if (argc == 3 && strcmp(argv[1], "--impulse") == 0) {
        int i = bounded_int(argv[2],0,6);
        return i < 0 ? 2 : (check_stencil(3,3,3,1,i) ? 0 : 1);
    }
    if (argc != 1) return 2;
    for (int i = 0; i < 7; ++i) if (!check_stencil(3,3,3,1,i)) return 1;
    const int shapes[][3] = {{3,3,3},{3,5,7},{17,19,5},{31,16,7},{33,65,9},{64,64,64}};
    for (const auto& shape : shapes)
        for (int steps : {1,2,5,10,100})
            if (!check_stencil(shape[0],shape[1],shape[2],steps)) return 1;
    return 0;
}
