// Altis RandomAccess: exactly 4*N LFSR XOR updates to a power-of-two table.
// Pinned source: utcs-scea/altis at 042e2926d1fe9a5ea5b3e2d659b75bf925b152cc,
// src/cuda/level1/gups/gups.cu. The square-table jump and both atomic policies
// follow that source. Initial table upload is untimed; one selected update
// kernel is measured once, then every actual table entry is checked on CPU.
// BENCH_PARAM_table_size, block_size, num_threads, atomic_mode (cas|xor).
// The old num_updates override is incompatible with the fixed 4*N workload.

/* Adapted from Altis; RandomAccess was contributed by the DARPA HPCS program.
BSD 2-Clause License

Copyright (c) 2021, Bodun Hu, et al. at SCEA in UT Austin CS Dept
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "../../../common_cuda/bench_timing.h"
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

#define CUDA_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

// Host arithmetic and independent serial reference.
using gups_word_t = unsigned long long;
union GupsWord {
    gups_word_t value;
    uint2 halves;
};
static_assert(sizeof(GupsWord) == 8, "GUPS requires 64-bit table entries");

static gups_word_t gups_step(gups_word_t state) {
    return (state << 1) ^ ((state >> 63) ? 0x7ULL : 0ULL);
}

static void gups_square_table(gups_word_t* squares) {
    gups_word_t state = 1;
    for (int bit = 0; bit < 64; ++bit) {
        squares[bit] = state;
        state = gups_step(gups_step(state));
    }
}

struct GupsGeometry {
    int blocks;
    gups_word_t actual_threads;
    gups_word_t total_updates;
    size_t table_bytes;
};

static bool gups_geometry(int table_size, int block_size, int threads, GupsGeometry* out) {
    if (table_size <= 0 || block_size <= 0 || threads <= 0 ||
        (table_size & (table_size - 1)) != 0) return false;
    const long long blocks = (static_cast<long long>(threads) + block_size - 1) / block_size;
    out->blocks = static_cast<int>(blocks);
    out->actual_threads = static_cast<gups_word_t>(blocks) * block_size;
    out->total_updates = 4ULL * table_size;
    out->table_bytes = static_cast<size_t>(table_size) * sizeof(GupsWord);
    // Positive int inputs imply actual_threads <2^32 and updates <=2^32,
    // so even the last thread's partition product fits unsigned 64 bits.
    return true;
}

static void gups_cpu_reference(GupsWord* table, int table_size) {
    gups_word_t state = 1;
    for (gups_word_t i = 0; i < 4ULL * table_size; ++i) {
        state = gups_step(state);
        table[state & (static_cast<gups_word_t>(table_size) - 1)].value ^= state;
    }
}

static size_t gups_verify(const GupsWord* measured, const GupsWord* expected, int table_size) {
    size_t errors = 0;
    for (int i = 0; i < table_size; ++i) {
        if (measured[i].value != expected[i].value) {
            if (errors < 10)
                fprintf(stderr, "Mismatch at entry %d: GPU=%llu CPU=%llu\n",
                        i, measured[i].value, expected[i].value);
            ++errors;
        }
    }
    return errors;
}

// Device stream jump and update kernels.
static __constant__ gups_word_t gups_squares[64];

// Pinned d_starts: square the LFSR polynomial state, then multiply by x for
// each set bit of the requested offset. This is part of the timed GPU work.
static __device__ gups_word_t gups_starts(gups_word_t offset) {
    if (offset == 0) return 1;
    int bit = 63 - __clzll(offset);
    gups_word_t state = 2;
    while (bit > 0) {
        gups_word_t squared = 0;
        for (int j = 0; j < 64; ++j)
            if ((state >> j) & 1) squared ^= gups_squares[j];
        state = squared;
        --bit;
        if ((offset >> bit) & 1)
            state = (state << 1) ^ ((state >> 63) ? 0x7ULL : 0ULL);
    }
    return state;
}

template<bool UseCAS>
static __device__ void gups_updates(GupsWord* table, int table_size) {
    const gups_word_t threads = static_cast<gups_word_t>(gridDim.x) * blockDim.x;
    const gups_word_t thread = static_cast<gups_word_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const gups_word_t updates = 4ULL * table_size;
    const gups_word_t begin = thread * updates / threads;
    const gups_word_t end = (thread + 1) * updates / threads;
    GupsWord random;
    random.value = gups_starts(begin);
    for (gups_word_t i = begin; i < end; ++i) {
        random.value = (random.value << 1) ^ ((random.value >> 63) ? 0x7ULL : 0ULL);
        GupsWord* target = &table[random.value & (static_cast<gups_word_t>(table_size) - 1)];
        if (UseCAS) {
            // Retrying 64-bit CAS from the pinned implementation.
            gups_word_t old = target->value;
            gups_word_t assumed;
            do {
                assumed = old;
                old = atomicCAS(&target->value, assumed, assumed ^ random.value);
            } while (assumed != old);
        } else {
            // The pinned XOR variant uses two 32-bit atomics, not one 64-bit
            // atomic. XOR commutes, so the final halves combine exactly.
            atomicXor(&target->halves.x, random.halves.x);
            atomicXor(&target->halves.y, random.halves.y);
        }
    }
}

__global__ void gups_cas_kernel(GupsWord* table, int table_size) {
    gups_updates<true>(table, table_size);
}

__global__ void gups_xor_kernel(GupsWord* table, int table_size) {
    gups_updates<false>(table, table_size);
}

int main(int argc, char** argv) try {
    (void)argc; (void)argv;
    setvbuf(stdout, NULL, _IONBF, 0);
    int table_size = 1048576;
    int block_size = 256;
    int num_threads = 65536;
    const char* mode = "xor";
    if (const char* value = getenv("BENCH_PARAM_table_size")) table_size = atoi(value);
    if (const char* value = getenv("BENCH_PARAM_block_size")) block_size = atoi(value);
    if (const char* value = getenv("BENCH_PARAM_num_threads")) num_threads = atoi(value);
    if (const char* value = getenv("BENCH_PARAM_atomic_mode")) mode = value;
    if (getenv("BENCH_PARAM_num_updates")) {
        fprintf(stderr, "Unsupported num_updates override: pinned GUPS performs exactly 4*table_size updates\n");
        return EXIT_FAILURE;
    }
    const bool use_cas = strcmp(mode, "cas") == 0;
    if (!use_cas && strcmp(mode, "xor") != 0) {
        fprintf(stderr, "Unsupported atomic_mode (expected cas or xor)\n");
        return EXIT_FAILURE;
    }
    mode = use_cas ? "cas" : "xor";
    GupsGeometry geometry;
    if (!gups_geometry(table_size, block_size, num_threads, &geometry)) {
        fprintf(stderr, "GUPS requires a positive power-of-two table and positive launch dimensions\n");
        return EXIT_FAILURE;
    }
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    if (block_size > prop.maxThreadsPerBlock || block_size > prop.maxThreadsDim[0] ||
        geometry.blocks > prop.maxGridSize[0]) {
        fprintf(stderr, "GUPS launch geometry exceeds device limits\n");
        return EXIT_FAILURE;
    }
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Table: %d entries | Threads: %d requested, %llu actual | Mode: %s | Logical updates: %llu\n",
            table_size, num_threads, geometry.actual_threads, mode, geometry.total_updates);

    std::vector<GupsWord> expected(table_size), measured(table_size);
    for (int i = 0; i < table_size; ++i) expected[i].value = static_cast<gups_word_t>(i);
    gups_word_t squares[64];
    gups_square_table(squares);
    CUDA_CHECK(cudaMemcpyToSymbol(gups_squares, squares, sizeof(squares)));
    GupsWord* d_table;
    CUDA_CHECK(cudaMalloc(&d_table, geometry.table_bytes));
    CUDA_CHECK(cudaMemcpy(d_table, expected.data(), geometry.table_bytes, cudaMemcpyHostToDevice));

    char params[512];
    snprintf(params, sizeof(params),
             "\"table_size\":%d,\"block_size\":%d,\"num_threads\":%d,"
             "\"num_blocks\":%d,\"actual_threads\":%llu,\"total_updates\":%llu,"
             "\"atomic_mode\":\"%s\",\"atomic_policy\":\"%s\","
             "\"stream_seed\":1,\"lfsr_polynomial\":7,"
             "\"algorithm\":\"altis_lfsr4n_v1\",\"verification\":\"full_table_cpu_v1\"",
             table_size, block_size, num_threads, geometry.blocks, geometry.actual_threads,
             geometry.total_updates, mode, use_cas ? "cas64_retry" : "xor32_pair");
    const double total_gpu_ms = bench_run_once(use_cas ? "gups_cas_kernel" : "gups_xor_kernel", params, [&]() {
        if (use_cas)
            gups_cas_kernel<<<dim3(geometry.blocks), dim3(block_size)>>>(d_table, table_size);
        else
            gups_xor_kernel<<<dim3(geometry.blocks), dim3(block_size)>>>(d_table, table_size);
    });
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // Download the actual measured table; no reset or second GPU update run.
    CUDA_CHECK(cudaMemcpy(measured.data(), d_table, geometry.table_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_table));
    gups_cpu_reference(expected.data(), table_size);
    const size_t errors = gups_verify(measured.data(), expected.data(), table_size);
    fprintf(stderr, "%s (verified %d table entries, %llu logical updates, %zu mismatches)\n",
            errors == 0 ? "PASS" : "FAIL", table_size, geometry.total_updates, errors);
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
} catch (const std::exception& error) {
    fprintf(stderr, "FAIL: GUPS host storage unavailable: %s\n", error.what());
    return EXIT_FAILURE;
}
