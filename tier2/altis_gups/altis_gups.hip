// altis_gups.hip — GUPS (Giga Updates Per Second) random memory access benchmark (HIP)
//
// Measures random memory access throughput by performing random read-modify-write
// operations on a large table of 64-bit entries. Each thread uses an xorshift
// PRNG to generate random indices and XOR-updates the table.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Environment variables:
//   BENCH_PARAM_table_size, BENCH_PARAM_block_size, BENCH_PARAM_num_updates,
//   BENCH_PARAM_num_threads
//   BENCH_PARAM_num_warmup   (warmup passes, default 3)
//   BENCH_PARAM_num_iters    (timed passes, each recorded, default 7)
//
// Output (stdout): one {"type":"kernel",...} JSON line per timed pass carrying
// the GPU event time and host wall-clock time, then a summary whose
// total_time_ms is the SUM of all passes' GPU event times. GUPS
// (total_updates / time) is derived downstream from time_ms and params.
// Output (stderr): human-readable results

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

// ---------------------------------------------------------------------------
// GUPS kernel: each thread performs num_updates random XOR updates on the table
// ---------------------------------------------------------------------------

__global__ void gups_kernel(unsigned long long* __restrict__ table,
                            int table_size,
                            int num_updates,
                            unsigned long long seed_base) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Per-thread xorshift64 state seeded uniquely
    unsigned long long state = seed_base + (unsigned long long)tid * 6364136223846793005ULL + 1;
    if (state == 0) state = 1;

    for (int u = 0; u < num_updates; ++u) {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;

        int idx = (int)(state % (unsigned long long)table_size);
        atomicXor(&table[idx], state);
    }
}

// ---------------------------------------------------------------------------
// CPU reference: perform random XOR updates
// ---------------------------------------------------------------------------

static void gups_cpu_ref(unsigned long long* table, int table_size,
                         int total_threads, int num_updates,
                         unsigned long long seed_base) {
    for (int tid = 0; tid < total_threads; ++tid) {
        unsigned long long state = seed_base + (unsigned long long)tid * 6364136223846793005ULL + 1;
        if (state == 0) state = 1;

        for (int u = 0; u < num_updates; ++u) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;

            int idx = (int)(state % (unsigned long long)table_size);
            table[idx] ^= state;
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Defaults from params.json
    int table_size   = 1048576;
    int block_size   = 256;
    int num_updates  = 128;
    int num_threads  = 65536;

    // Environment variable overrides
    const char* env_val;
    env_val = getenv("BENCH_PARAM_table_size");
    if (env_val) table_size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");
    if (env_val) block_size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_num_updates");
    if (env_val) num_updates = atoi(env_val);
    env_val = getenv("BENCH_PARAM_num_threads");
    if (env_val) num_threads = atoi(env_val);
    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

    int gridSize   = (num_threads + block_size - 1) / block_size;
    int actual_threads = gridSize * block_size;
    long long total_updates = (long long)actual_threads * num_updates;
    unsigned long long seed_base = 42ULL;

    // Print device info
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "GUPS benchmark  |  Table size: %d (%.2f MB)  |  "
            "Updates/thread: %d  |  Threads: %d  |  Total updates: %lld  |  "
            "Iterations: %d warmup + %d timed\n\n",
            table_size, (double)table_size * 8.0 / 1e6,
            num_updates, actual_threads, total_updates,
            num_warmup, num_iters);

    // -----------------------------------------------------------------------
    // Allocate table on device
    // -----------------------------------------------------------------------
    size_t table_bytes = (size_t)table_size * sizeof(unsigned long long);

    unsigned long long* d_table;
    HIP_CHECK(hipMalloc(&d_table, table_bytes));

    // Initialize table: each entry = index
    std::vector<unsigned long long> h_table(table_size);
    for (int i = 0; i < table_size; ++i) {
        h_table[i] = (unsigned long long)i;
    }

    auto reset_table = [&]() {
        HIP_CHECK(hipMemcpy(d_table, h_table.data(), table_bytes, hipMemcpyHostToDevice));
    };

    char params[192];
    snprintf(params, sizeof(params),
             "\"table_size\":%d,\"block_size\":%d,\"num_updates\":%d,"
             "\"num_threads\":%d",
             table_size, block_size, num_updates, num_threads);

    // One measurement pass. The kernel XOR-accumulates into the table, so the
    // pristine table is re-uploaded before every pass (untimed, as the old
    // code did) to keep each pass's work identical. The timed bracket
    // contains only the kernel launch.
    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t lap;

    auto run_pass = [&]() {
        reset_table();
        lap = bench_time_op(&timer, [&]() {
            gups_kernel<<<dim3(gridSize), dim3(block_size)>>>(
                d_table, table_size, num_updates, seed_base);
        });
    };

    for (int w = 0; w < num_warmup; ++w) {
        run_pass();
    }

    double total_gpu_ms = 0.0;
    for (int i = 0; i < num_iters; ++i) {
        run_pass();
        bench_emit_iter("gups_kernel", i, &lap, params);
        total_gpu_ms += lap.gpu_ms;
        fprintf(stderr, "gups_kernel iter %d: gpu = %.3f ms, host = %.3f ms\n",
                i, lap.gpu_ms, lap.host_ms);
    }

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d iterations\n",
            total_gpu_ms, num_iters);

    // -----------------------------------------------------------------------
    // Verification
    // -----------------------------------------------------------------------
    {
        reset_table();
        gups_kernel<<<dim3(gridSize), dim3(block_size)>>>(
            d_table, table_size, num_updates, seed_base);
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<unsigned long long> gpu_table(table_size);
        HIP_CHECK(hipMemcpy(gpu_table.data(), d_table, table_bytes, hipMemcpyDeviceToHost));

        std::vector<unsigned long long> cpu_table(h_table.begin(), h_table.end());
        gups_cpu_ref(cpu_table.data(), table_size, actual_threads, num_updates, seed_base);

        int verify_n = (table_size < 1024) ? table_size : 1024;
        int errors = 0;
        for (int i = 0; i < verify_n; ++i) {
            if (gpu_table[i] != cpu_table[i]) {
                if (errors < 10) {
                    fprintf(stderr, "Mismatch at index %d: GPU=%llu CPU=%llu\n",
                            i, gpu_table[i], cpu_table[i]);
                }
                errors++;
            }
        }

        if (errors > 0)
            fprintf(stderr, "FAIL: %d mismatches in first %d entries\n", errors, verify_n);
        else
            fprintf(stderr, "PASS\n");
    }

    HIP_CHECK(hipFree(d_table));
    return 0;
}
