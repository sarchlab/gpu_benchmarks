// parboil_cutcp.hip — Parboil CUTCP benchmark: Coulombic potential with cutoff (HIP)
//
// Direct summation of electrostatic charge interactions on a 3D grid.
// Each thread computes the potential at one grid point by iterating over
// all atoms within a cutoff radius.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Environment variables:
//   BENCH_PARAM_num_atoms, BENCH_PARAM_block_size,
//   BENCH_PARAM_grid_spacing, BENCH_PARAM_cutoff_radius
//   BENCH_PARAM_num_warmup   (warmup launches, default 3)
//   BENCH_PARAM_num_iters    (timed launches, each recorded, default 7)
//
// Output (stdout): one {"type":"kernel",...} JSON line per timed iteration
// carrying the GPU event time and host wall-clock time, then a summary whose
// total_time_ms is the SUM of all iterations' GPU event times. Throughput
// (interactions/sec) is derived downstream from time_ms and params.
// Output (stderr): human-readable results

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define GRID_DIM_SIDE 64

// ---------------------------------------------------------------------------
// Kernel: compute Coulombic potential at each grid point
// ---------------------------------------------------------------------------

__global__ void cutcp_kernel(
    const float4* __restrict__ atoms,
    float*        __restrict__ potential,
    int           num_atoms,
    int           grid_side,
    float         grid_spacing,
    float         cutoff,
    float         cutoff2)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_points = grid_side * grid_side * grid_side;
    if (gid >= total_points) return;

    int gz = gid / (grid_side * grid_side);
    int gy = (gid / grid_side) % grid_side;
    int gx = gid % grid_side;

    float px = gx * grid_spacing;
    float py = gy * grid_spacing;
    float pz = gz * grid_spacing;

    float pot = 0.0f;

    for (int i = 0; i < num_atoms; ++i) {
        float4 atom = atoms[i];
        float dx = px - atom.x;
        float dy = py - atom.y;
        float dz = pz - atom.z;
        float r2 = dx * dx + dy * dy + dz * dz;

        if (r2 < cutoff2 && r2 > 1e-12f) {
            float r = sqrtf(r2);
            pot += atom.w / r;
        }
    }

    potential[gid] = pot;
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

static float cpu_potential(
    const float* atoms_x, const float* atoms_y,
    const float* atoms_z, const float* atoms_q,
    int num_atoms, float px, float py, float pz,
    float cutoff2)
{
    float pot = 0.0f;
    for (int i = 0; i < num_atoms; ++i) {
        float dx = px - atoms_x[i];
        float dy = py - atoms_y[i];
        float dz = pz - atoms_z[i];
        float r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < cutoff2 && r2 > 1e-12f) {
            pot += atoms_q[i] / sqrtf(r2);
        }
    }
    return pot;
}

// ---------------------------------------------------------------------------
// Simple PRNG
// ---------------------------------------------------------------------------

static inline unsigned int lcg_rand(unsigned int* state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static inline float rand_float(unsigned int* state, float lo, float hi) {
    return lo + (float)(lcg_rand(state) & 0xFFFF) / 65535.0f * (hi - lo);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    int num_atoms    = 10000;
    int block_size   = 128;
    float grid_spacing = 0.5f;
    float cutoff     = 12.0f;

    const char* env_val;
    env_val = getenv("BENCH_PARAM_num_atoms");
    if (env_val) num_atoms = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_grid_spacing");
    if (env_val) grid_spacing = (float)atof(env_val);
    env_val = getenv("BENCH_PARAM_cutoff_radius");
    if (env_val) cutoff = (float)atof(env_val);
    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

    float cutoff2 = cutoff * cutoff;
    int grid_side = GRID_DIM_SIDE;
    int total_grid_points = grid_side * grid_side * grid_side;
    float domain_size = grid_side * grid_spacing;

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "CUTCP  |  Atoms: %d  |  Grid: %d^3 = %d  |  Cutoff: %.1f  |  "
            "Spacing: %.2f  |  Iterations: %d warmup + %d timed\n\n",
            num_atoms, grid_side, total_grid_points, cutoff, grid_spacing,
            num_warmup, num_iters);

    // Generate synthetic atom data
    std::vector<float> h_atoms_x(num_atoms), h_atoms_y(num_atoms);
    std::vector<float> h_atoms_z(num_atoms), h_atoms_q(num_atoms);
    std::vector<float4> h_atoms(num_atoms);

    unsigned int seed = 42;
    for (int i = 0; i < num_atoms; ++i) {
        float x = rand_float(&seed, 0.0f, domain_size);
        float y = rand_float(&seed, 0.0f, domain_size);
        float z = rand_float(&seed, 0.0f, domain_size);
        float q = rand_float(&seed, -1.0f, 1.0f);
        h_atoms_x[i] = x;
        h_atoms_y[i] = y;
        h_atoms_z[i] = z;
        h_atoms_q[i] = q;
        h_atoms[i] = make_float4(x, y, z, q);
    }

    size_t atom_bytes = (size_t)num_atoms * sizeof(float4);
    size_t grid_bytes = (size_t)total_grid_points * sizeof(float);

    float4* d_atoms;
    float*  d_potential;

    HIP_CHECK(hipMalloc(&d_atoms,     atom_bytes));
    HIP_CHECK(hipMalloc(&d_potential,  grid_bytes));
    HIP_CHECK(hipMemcpy(d_atoms, h_atoms.data(), atom_bytes, hipMemcpyHostToDevice));

    int gridSize = (total_grid_points + block_size - 1) / block_size;

    char params[192];
    snprintf(params, sizeof(params),
             "\"num_atoms\":%d,\"block_size\":%d,"
             "\"grid_spacing\":%.2f,\"cutoff_radius\":%.1f",
             num_atoms, block_size, grid_spacing, cutoff);

    // Warmup + timed iterations, each recorded individually. The kernel
    // rewrites d_potential as a pure function of the constant atom data, so
    // repeated launches do identical work — no per-iteration reset needed.
    double total_gpu_ms =
        bench_run_loop("cutcp_kernel", num_warmup, num_iters, params, [&]() {
            cutcp_kernel<<<dim3(gridSize), dim3(block_size)>>>(
                d_atoms, d_potential, num_atoms, grid_side, grid_spacing,
                cutoff, cutoff2);
        });

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d iterations\n",
            total_gpu_ms, num_iters);

    // Verification
    {
        std::vector<float> h_potential(total_grid_points);
        HIP_CHECK(hipMemcpy(h_potential.data(), d_potential, grid_bytes,
                            hipMemcpyDeviceToHost));

        int verify_n = 64;
        int errors = 0;
        unsigned int vseed = 123;
        for (int v = 0; v < verify_n; ++v) {
            int idx = lcg_rand(&vseed) % total_grid_points;
            int gz = idx / (grid_side * grid_side);
            int gy = (idx / grid_side) % grid_side;
            int gx = idx % grid_side;
            float px = gx * grid_spacing;
            float py = gy * grid_spacing;
            float pz = gz * grid_spacing;

            float cpu_pot = cpu_potential(
                h_atoms_x.data(), h_atoms_y.data(),
                h_atoms_z.data(), h_atoms_q.data(),
                num_atoms, px, py, pz, cutoff2);

            float gpu_pot = h_potential[idx];
            float diff = fabsf(gpu_pot - cpu_pot);
            float tol = 1e-3f * (fabsf(cpu_pot) + 1e-6f);
            if (diff > tol) {
                if (errors < 10) {
                    fprintf(stderr, "Mismatch at grid[%d] (%d,%d,%d): "
                            "GPU=%.6f CPU=%.6f diff=%.6f\n",
                            idx, gx, gy, gz, gpu_pot, cpu_pot, diff);
                }
                errors++;
            }
        }

        if (errors > 0)
            fprintf(stderr, "FAIL: %d errors in %d verified points\n", errors, verify_n);
        else
            fprintf(stderr, "PASS\n");
    }

    HIP_CHECK(hipFree(d_atoms));
    HIP_CHECK(hipFree(d_potential));

    return 0;
}
