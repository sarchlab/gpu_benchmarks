// parboil_stencil.hip — Parboil 3D Jacobi Stencil benchmark (HIP, self-contained)
//
// 7-point 3D stencil over a 64×64×64 grid using ping-pong buffers.
// Each kernel invocation performs one time-step; multiple time-steps per iteration.
//
// Stencil:
//   out[i] = c0*in[i] + c1*(in[i-1] + in[i+1] + in[i-nx] + in[i+nx]
//                           + in[i-nx*ny] + in[i+nx*ny])
//   c0 = 0.6,  c1 = (1-c0)/6 ≈ 0.0667
//
// Boundary cells (1-cell border) remain fixed.
//
// Compiles with hipcc (AMD ROCm) or nvcc (CUDA).
//
// Usage:
//   ./parboil_stencil [--grid_dim N] [--num_timesteps T] [--iterations I]
//
//   --grid_dim N        Grid dimension (N×N×N)           (default: 64)
//   --num_timesteps T   Stencil time-steps per launch    (default: 10)
//   --iterations I      Timed benchmark iterations       (default: 5)
//
// Output (stdout): CSV row — kernel_name,problem_size,iterations,avg_ms,min_ms,max_ms,stddev_ms
// Output (stderr): GFLOPS and GB/s

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <vector>

// ---------------------------------------------------------------------------
// Inlined bench_common_hip.h utilities
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

struct BenchResult {
    const char* kernel_name;
    const char* problem_size;
    int         iterations;
    double      avg_ms;
    double      min_ms;
    double      max_ms;
    double      stddev_ms;
};

static int parseIterations(int argc, char** argv) {
    int iters = 5;
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], "--iterations") == 0) {
            int v = atoi(argv[i + 1]);
            if (v >= 1) iters = v;
            break;
        }
    }
    return iters;
}

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

static void printCSVHeader() {
    printf("kernel_name,problem_size,iterations,avg_ms,min_ms,max_ms,stddev_ms\n");
}

static void printCSVRow(const BenchResult& r) {
    printf("%s,%s,%d,%.4f,%.4f,%.4f,%.4f\n",
           r.kernel_name, r.problem_size, r.iterations,
           r.avg_ms, r.min_ms, r.max_ms, r.stddev_ms);
}

struct BenchmarkTimer {
    hipEvent_t start, stop;
    BenchmarkTimer() {
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
    }
    ~BenchmarkTimer() {
        (void)hipEventDestroy(start);
        (void)hipEventDestroy(stop);
    }
    void record_start(hipStream_t stream = 0) {
        HIP_CHECK(hipEventRecord(start, stream));
    }
    void record_stop(hipStream_t stream = 0) {
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));
    }
    float elapsed_ms() const {
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

template <typename Func>
static BenchResult runBenchmark(const char* kernel_name,
                                const char* problem_size,
                                int         iterations,
                                Func        func) {
    BenchmarkTimer timer;

    // Warmup
    func();
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<double> times(iterations);
    for (int i = 0; i < iterations; ++i) {
        timer.record_start();
        func();
        timer.record_stop();
        times[i] = static_cast<double>(timer.elapsed_ms());
    }

    double sum = 0.0, mn = DBL_MAX, mx = 0.0;
    for (int i = 0; i < iterations; ++i) {
        sum += times[i];
        if (times[i] < mn) mn = times[i];
        if (times[i] > mx) mx = times[i];
    }
    double avg = sum / iterations;

    double variance = 0.0;
    for (int i = 0; i < iterations; ++i) {
        double d = times[i] - avg;
        variance += d * d;
    }
    double stddev = (iterations > 1) ? sqrt(variance / (iterations - 1)) : 0.0;

    BenchResult r;
    r.kernel_name  = kernel_name;
    r.problem_size = problem_size;
    r.iterations   = iterations;
    r.avg_ms       = avg;
    r.min_ms       = mn;
    r.max_ms       = mx;
    r.stddev_ms    = stddev;
    return r;
}

// ---------------------------------------------------------------------------
// 7-point 3D stencil kernel
// ---------------------------------------------------------------------------

__global__ void stencil3d(const float* __restrict__ in,
                          float*       __restrict__ out,
                          int nx, int ny, int nz,
                          float c0, float c1) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z;

    // Only compute interior points (skip 1-cell boundary)
    if (ix >= 1 && ix < nx - 1 &&
        iy >= 1 && iy < ny - 1 &&
        iz >= 1 && iz < nz - 1) {
        int idx = iz * ny * nx + iy * nx + ix;
        out[idx] = c0 * in[idx]
                 + c1 * (in[idx - 1]       // x-1
                       + in[idx + 1]       // x+1
                       + in[idx - nx]      // y-1
                       + in[idx + nx]      // y+1
                       + in[idx - ny*nx]   // z-1
                       + in[idx + ny*nx]); // z+1
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    int grid_dim      = parseIntParam(argc, argv, "--grid_dim",      64);
    int num_timesteps = parseIntParam(argc, argv, "--num_timesteps", 10);
    int iters         = parseIterations(argc, argv);

    int nx = grid_dim;
    int ny = grid_dim;
    int nz = grid_dim;
    int total = nx * ny * nz;

    float c0 = 0.6f;
    float c1 = (1.0f - c0) / 6.0f;

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Grid: %d×%d×%d  |  Timesteps/launch: %d  |  Iterations: %d\n\n",
            nx, ny, nz, num_timesteps, iters);

    // Host allocation and initialization
    float* h_data = (float*)malloc(total * sizeof(float));
    srand(42);
    for (int i = 0; i < total; ++i) {
        h_data[i] = (float)(rand() % 1000) / 1000.0f;
    }

    // Device allocations (ping-pong buffers)
    float *d_A, *d_B;
    HIP_CHECK(hipMalloc(&d_A, total * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, total * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_A, h_data, total * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_data, total * sizeof(float), hipMemcpyHostToDevice));

    // Kernel launch configuration
    dim3 block(16, 16, 1);
    dim3 grid((nx + 15) / 16, (ny + 15) / 16, nz);

    char problemSize[64];
    snprintf(problemSize, sizeof(problemSize), "%dx%dx%d", nx, ny, nz);

    printCSVHeader();

    BenchResult r = runBenchmark(
        "stencil3d", problemSize, iters, [&]() {
            float* src = d_A;
            float* dst = d_B;
            for (int t = 0; t < num_timesteps; ++t) {
                stencil3d<<<grid, block>>>(src, dst, nx, ny, nz, c0, c1);
                // Ping-pong
                float* tmp = src; src = dst; dst = tmp;
            }
            HIP_CHECK(hipDeviceSynchronize());
        });

    printCSVRow(r);

    // Performance metrics
    long long interior_cells = (long long)(nx - 2) * (ny - 2) * (nz - 2);
    // 8 flops per interior cell per timestep (6 adds + 1 center mul + 1 neighbor mul... 
    // more precisely: 6 neighbor reads contribute 6 multiplies by c1, 6 adds, 
    // plus 1 multiply by c0 = 8 multiply-adds total, ~13 flops)
    // Using the spec: 8 * interior * timesteps
    double gflops = 8.0 * (double)interior_cells * num_timesteps
                    / (r.avg_ms * 1e-3) / 1e9;
    // 7 reads + 1 write per interior cell per timestep
    double gb_s = 8.0 * sizeof(float) * (double)interior_cells * num_timesteps
                  / (r.avg_ms * 1e-3) / 1e9;

    fprintf(stderr, "GFLOPS: %.2f  |  GB/s: %.2f  (avg %.4f ms, %d timesteps/iter)\n",
            gflops, gb_s, r.avg_ms, num_timesteps);

    // CSV summary line
    printf("parboil_stencil,%dx%dx%d,%.4f\n", nx, ny, nz, gflops);

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    free(h_data);

    return 0;
}
