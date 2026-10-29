// cuda_nbody.hip — CUDA N-body gravitational simulation benchmark (HIP)
//
// All-pairs N-body with tile-based shared memory optimization.
// Each thread computes the force on one body from all others.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./cuda_nbody [--bodies N] [--timesteps T]
//
//   --bodies N       Number of bodies (default: 16384)
//   --timesteps T    Simulation timesteps per run (default: 10)
//
// Environment variables (override argv):
//   BENCH_PARAM_bodies, BENCH_PARAM_timesteps
//
// Output (stdout): per timed pass, one {"type":"kernel",...} JSON line for
// nbody_kernel carrying the accumulated GPU event time of all timesteps in
// the pass plus the host wall-clock time, then a summary whose total_time_ms
// is the SUM of all recorded GPU event times. GFLOPS is derived downstream
// from time_ms and params.
// Output (stderr): human-readable results

#include "../../../common_hip/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <climits>
#include <exception>

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define TILE_SIZE 256
#define SOFTENING 1e-5f
#define DT        0.01f

// ---------------------------------------------------------------------------
// Kernel: tile-based N-body force computation + velocity/position update
//   Uses shared memory to cache tiles of positions for coalesced access.
// ---------------------------------------------------------------------------

// Positions are double-buffered: the kernel reads every body's position from
// posIn (its own and all tiles) and writes only its own updated position to
// posOut. Reading and writing the same array in one launch would race, because
// a block tiling posIn[idx] has no ordering guarantee against another block
// writing posOut[i] — upstream (GPU Gems 3) double-buffers for exactly this.
// Velocity is per-thread (thread i is the sole reader/writer of vel[i]), so it
// stays single-buffered.
__global__ void nbody_kernel(const float4* __restrict__ posIn,
                              float4* __restrict__ posOut,
                              float4* __restrict__ vel,
                              int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float4 myPos;
    if (i < N) myPos = posIn[i];

    float3 acc = {0.0f, 0.0f, 0.0f};

    __shared__ float4 shPos[TILE_SIZE];

    for (int tile = 0; tile < N; tile += TILE_SIZE) {
        int idx = tile + threadIdx.x;
        if (idx < N)
            shPos[threadIdx.x] = posIn[idx];
        else
            shPos[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        __syncthreads();

        if (i < N) {
            for (int j = 0; j < TILE_SIZE; ++j) {
                float dx = shPos[j].x - myPos.x;
                float dy = shPos[j].y - myPos.y;
                float dz = shPos[j].z - myPos.z;

                float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                float invDist  = rsqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;
                float mass_j   = shPos[j].w;

                acc.x += dx * invDist3 * mass_j;
                acc.y += dy * invDist3 * mass_j;
                acc.z += dz * invDist3 * mass_j;
            }
        }

        __syncthreads();
    }

    if (i < N) {
        float4 myVel = vel[i];
        myVel.x += acc.x * DT;
        myVel.y += acc.y * DT;
        myVel.z += acc.z * DT;

        myPos.x += myVel.x * DT;
        myPos.y += myVel.y * DT;
        myPos.z += myVel.z * DT;

        posOut[i] = myPos;
        vel[i]    = myVel;
    }
}

// ---------------------------------------------------------------------------
// CPU reference N-body for verification
// ---------------------------------------------------------------------------

static void nbody_cpu(float4* pos, float4* vel, int N) {
    std::vector<float3> acc(N, {0.0f, 0.0f, 0.0f});

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float dx = pos[j].x - pos[i].x;
            float dy = pos[j].y - pos[i].y;
            float dz = pos[j].z - pos[i].z;

            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist  = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;
            float mass_j   = pos[j].w;

            acc[i].x += dx * invDist3 * mass_j;
            acc[i].y += dy * invDist3 * mass_j;
            acc[i].z += dz * invDist3 * mass_j;
        }
    }

    for (int i = 0; i < N; ++i) {
        vel[i].x += acc[i].x * DT;
        vel[i].y += acc[i].y * DT;
        vel[i].z += acc[i].z * DT;

        pos[i].x += vel[i].x * DT;
        pos[i].y += vel[i].y * DT;
        pos[i].z += vel[i].z * DT;
    }
}

// Position and velocity use the same relative-plus-floor rule independently.
// The fourth component is immutable (mass for position, padding for velocity).
static bool nbody_vector_matches(const float4& got, const float4& ref) {
    const float tol = 1e-2f * (fabsf(ref.x) + fabsf(ref.y) + fabsf(ref.z)) + 1e-5f;
    return std::isfinite(got.x) && std::isfinite(got.y) &&
           std::isfinite(got.z) && std::isfinite(got.w) &&
           std::isfinite(ref.x) && std::isfinite(ref.y) &&
           std::isfinite(ref.z) && std::isfinite(ref.w) && std::isfinite(tol) &&
           fabsf(got.x - ref.x) <= tol && fabsf(got.y - ref.y) <= tol &&
           fabsf(got.z - ref.z) <= tol && got.w == ref.w;
}

// ---------------------------------------------------------------------------
// Simple deterministic PRNG for initialization
// ---------------------------------------------------------------------------

static float rand_float(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (float)(state & 0xFFFFFF) / (float)0xFFFFFF * 2.0f - 1.0f;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Default values from params.json
    int N          = 16384;
    int timesteps  = 10;

    // Command-line args as fallback
    N          = parseIntParam(argc, argv, "--bodies", N);
    timesteps  = parseIntParam(argc, argv, "--timesteps", timesteps);

    // Env vars override
    const char* env_val;
    env_val = getenv("BENCH_PARAM_bodies");
    if (env_val) N = atoi(env_val);
    env_val = getenv("BENCH_PARAM_timesteps");
    if (env_val) timesteps = atoi(env_val);

    if (N <= 0 || timesteps <= 0 || N > INT_MAX - (TILE_SIZE - 1)) {
        fprintf(stderr, "Invalid bodies/timesteps or padded body index range\n");
        return EXIT_FAILURE;
    }

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Bodies: %d  |  Timesteps: %d\n\n",
            N, timesteps);

    size_t bytes = (size_t)N * sizeof(float4);

    // Host allocation + initialization
    std::vector<float4> h_pos(N), h_vel(N);
    std::vector<float4> h_pos_orig(N), h_vel_orig(N);
    uint32_t rng = 42;

    for (int i = 0; i < N; ++i) {
        h_pos[i].x = rand_float(rng);
        h_pos[i].y = rand_float(rng);
        h_pos[i].z = rand_float(rng);
        h_pos[i].w = 1.0f;  // mass
        h_vel[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        h_pos_orig[i] = h_pos[i];
        h_vel_orig[i] = h_vel[i];
    }

    // Device allocation. Two position buffers (ping-pong): each step reads one
    // and writes the other, so no launch reads a buffer another block is writing.
    float4 *d_pos, *d_pos2, *d_vel;
    HIP_CHECK(hipMalloc(&d_pos,  bytes));
    HIP_CHECK(hipMalloc(&d_pos2, bytes));
    HIP_CHECK(hipMalloc(&d_vel,  bytes));

    int blockSize = TILE_SIZE;
    int gridSize  = (N + blockSize - 1) / blockSize;

    char params[160];
    snprintf(params, sizeof(params),
             "\"bodies\":%d,\"timesteps\":%d,\"verification\":\"full_cpu_v1\"",
             N, timesteps);

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_nbody;

    float4* final_positions = nullptr;

    // One full simulation pass. The kernel integrates positions/velocities in
    // place, so particle state is reset from the pristine host copies at the
    // start of every pass (untimed) — each pass simulates the same trajectory.
    // Only the kernel launches are inside the timed brackets.
    auto run_nbody_pass = [&]() {
        bench_lap_reset(&acc_nbody);

        HIP_CHECK(hipMemcpy(d_pos, h_pos_orig.data(), bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vel, h_vel_orig.data(), bytes, hipMemcpyHostToDevice));

        /* Cold-cache calibration: evict the LLC (and hold the clock high) after
         * the per-pass H2D re-init so the pass's first kernel launch runs cold.
         * No-op when BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        float4* posIn  = d_pos;
        float4* posOut = d_pos2;
        for (int t = 0; t < timesteps; ++t) {
            bench_lap_add(&acc_nbody, bench_time_op(&timer, [&]() {
                nbody_kernel<<<dim3(gridSize), dim3(blockSize)>>>(posIn, posOut, d_vel, N);
            }));
            float4* tmp = posIn; posIn = posOut; posOut = tmp;
        }
        final_positions = posIn;
    };

    double total_gpu_ms = 0.0;
    run_nbody_pass();
    bench_emit_iter("nbody_kernel", 0, &acc_nbody, params);
    total_gpu_ms += acc_nbody.gpu_ms;
    fprintf(stderr, "nbody_kernel gpu = %.3f ms, host = %.3f ms\n",
            acc_nbody.gpu_ms, acc_nbody.host_ms);

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // Verify the actual measured final state, including every integration step.
    // Reuse the host buffers: measured state replaces initialization copies;
    // the pristine copies become the complete CPU trajectory.
    HIP_CHECK(hipMemcpy(h_pos.data(), final_positions, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_vel.data(), d_vel, bytes, hipMemcpyDeviceToHost));
    bool verify_ok = true;
    try {
        for (int t = 0; t < timesteps; ++t)
            nbody_cpu(h_pos_orig.data(), h_vel_orig.data(), N);

        int errors = 0;
        for (int i = 0; i < N; ++i) {
            const bool position_ok = nbody_vector_matches(h_pos[i], h_pos_orig[i]);
            const bool velocity_ok = nbody_vector_matches(h_vel[i], h_vel_orig[i]);
            if (!position_ok || !velocity_ok) {
                if (errors < 10) {
                    fprintf(stderr, "Mismatch at body %d: position=%s velocity=%s\n",
                            i, position_ok ? "ok" : "bad", velocity_ok ? "ok" : "bad");
                    if (!position_ok)
                        fprintf(stderr, "  position GPU=(%.9g,%.9g,%.9g,%.9g) CPU=(%.9g,%.9g,%.9g,%.9g)\n",
                                h_pos[i].x, h_pos[i].y, h_pos[i].z, h_pos[i].w,
                                h_pos_orig[i].x, h_pos_orig[i].y, h_pos_orig[i].z, h_pos_orig[i].w);
                    if (!velocity_ok)
                        fprintf(stderr, "  velocity GPU=(%.9g,%.9g,%.9g,%.9g) CPU=(%.9g,%.9g,%.9g,%.9g)\n",
                                h_vel[i].x, h_vel[i].y, h_vel[i].z, h_vel[i].w,
                                h_vel_orig[i].x, h_vel_orig[i].y, h_vel_orig[i].z, h_vel_orig[i].w);
                }
                ++errors;
            }
        }
        verify_ok = errors == 0;
        fprintf(stderr, "%s (verified %d bodies, %d timesteps, positions and velocities, %d mismatches)\n",
                verify_ok ? "PASS" : "FAIL", N, timesteps, errors);
    } catch (const std::exception& error) {
        fprintf(stderr, "FAIL: complete CPU reference unavailable: %s\n", error.what());
        verify_ok = false;
    }

    HIP_CHECK(hipFree(d_pos));
    HIP_CHECK(hipFree(d_pos2));
    HIP_CHECK(hipFree(d_vel));

    return verify_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
