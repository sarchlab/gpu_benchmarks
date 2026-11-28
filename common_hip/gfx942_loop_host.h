/**
 * gfx942_loop_host.h
 *
 * The host side that the hand-written gfx942 loop benchmarks (instruction_cost
 * and loop_wait) have in common, so that it exists once: parameter reading,
 * the CPU reference of the VALU instructions, and the single timed launch of
 * one kernel of an embedded code object followed by the comparison of every
 * thread's stored result. The kernel side is gfx942_loop_kernels.h.
 *
 * A benchmark's main() reads its own parameters, picks the kernel, computes
 * the 64-bit value every thread must store, and calls lk_run().
 */

#ifndef GFX942_LOOP_HOST_H
#define GFX942_LOOP_HOST_H

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "bench_timing.h"
#include "gfx942_loop_kernels.h"

#define HIP_CHECK(call) BENCH_TIMING_CHECK(call)

/* The processor the kernels are assembled for (-mcpu in the Makefiles). */
static const char LK_KERNEL_ARCH[] = "gfx942";

static const int LK_DEFAULT_ITERATIONS = 65536;
/** One wavefront: the per-instruction cost of a wavefront running alone. */
static const int LK_DEFAULT_THREADS_PER_BLOCK = 64;
static const int LK_DEFAULT_NUM_BLOCKS = 1;

// ---------------------------------------------------------------------------
// The VALU instructions (single-sourced with the assembly)
// ---------------------------------------------------------------------------

#define LK_ENUMERATOR(instruction) LK_##instruction,
#define LK_NAME(instruction) #instruction,

enum LkValu { LK_FOR_EACH_VALU(LK_ENUMERATOR) LK_VALU_COUNT };

static const char *const LK_VALU_NAMES[LK_VALU_COUNT] = {LK_FOR_EACH_VALU(LK_NAME)};

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

static float lk_f32_from_bits(uint32_t bits)
{
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32_t lk_bits_from_f32(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static double lk_f64_from_bits(uint64_t bits)
{
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint64_t lk_bits_from_f64(double value)
{
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static uint64_t lk_pair(uint32_t lo, uint32_t hi)
{
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

/**
 * The 64-bit X of an instruction after `applications` applications of
 * X = op(X, ...) from its initial operands in gfx942_loop_kernels.h. GPU and
 * CPU both round to nearest and fuse the FMAs, so the comparison is bit-exact.
 */
static uint64_t lk_reference(LkValu instruction, uint64_t applications)
{
    const float f32_m = lk_f32_from_bits(LK_F32_M);
    const float f32_a = lk_f32_from_bits(LK_F32_A);
    switch (instruction) {
    case LK_v_fma_f32: {
        float x = lk_f32_from_bits(LK_F32_X0);
        for (uint64_t i = 0; i < applications; ++i) x = fmaf(x, f32_m, f32_a);
        return lk_bits_from_f32(x);
    }
    case LK_v_mul_f32: {
        float x = lk_f32_from_bits(LK_F32_X0);
        for (uint64_t i = 0; i < applications; ++i) x = x * f32_m;
        return lk_bits_from_f32(x);
    }
    case LK_v_add_f32: {
        float x = lk_f32_from_bits(LK_F32_X0);
        for (uint64_t i = 0; i < applications; ++i) x = x + f32_a;
        return lk_bits_from_f32(x);
    }
    case LK_v_mul_lo_u32: {
        uint32_t x = LK_U32_X0;
        for (uint64_t i = 0; i < applications; ++i) x *= LK_U32_M;
        return x;
    }
    case LK_v_add_u32: {
        uint32_t x = LK_U32_X0;
        for (uint64_t i = 0; i < applications; ++i) x += LK_U32_A;
        return x;
    }
    case LK_v_fma_f64: {
        const double m = lk_f64_from_bits(lk_pair(LK_F64_M_LO, LK_F64_M_HI));
        const double a = lk_f64_from_bits(lk_pair(LK_F64_A_LO, LK_F64_A_HI));
        double x = lk_f64_from_bits(lk_pair(LK_F64_X0_LO, LK_F64_X0_HI));
        for (uint64_t i = 0; i < applications; ++i) x = fma(x, m, a);
        return lk_bits_from_f64(x);
    }
    case LK_v_pk_fma_f32: {
        float lo = lk_f32_from_bits(LK_F32_X0);
        float hi = lk_f32_from_bits(LK_F32_X0_HI);
        for (uint64_t i = 0; i < applications; ++i) {
            lo = fmaf(lo, f32_m, f32_a);
            hi = fmaf(hi, f32_m, f32_a);
        }
        return lk_pair(lk_bits_from_f32(lo), lk_bits_from_f32(hi));
    }
    case LK_v_mov_b64:
    case LK_VALU_COUNT:
        break;
    }
    return lk_pair(LK_B64_LO, LK_B64_HI); /* a move leaves X unchanged */
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

static void lk_fail(const char *message)
{
    printf("{\"type\":\"error\",\"severity\":\"fatal\",\"message\":\"%s\"}\n", message);
    fprintf(stderr, "FAIL: %s\n", message);
    exit(EXIT_FAILURE);
}

/**
 * An absent variable takes the fallback. A present one must be, in full, a
 * decimal integer that fits an int: "oops", "4096.5" and "" are fatal rather
 * than read as some other number.
 */
static int lk_int_param(const char *name, int fallback)
{
    const char *value = getenv(name);
    if (!value) return fallback;

    errno = 0;
    char *end = nullptr;
    const long parsed = strtol(value, &end, 10);
    if (isspace(static_cast<unsigned char>(value[0])) || end == value || *end != '\0' ||
        errno == ERANGE || parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
        fprintf(stderr, "%s=\"%s\"\n", name, value);
        char message[96];
        snprintf(message, sizeof(message), "%s must be an integer", name);
        lk_fail(message);
    }
    return static_cast<int>(parsed);
}

static const char *lk_string_param(const char *name, const char *fallback)
{
    const char *value = getenv(name);
    return value ? value : fallback;
}

/** The index of `value` among `count` names, or `count` when it is not one. */
static int lk_index_of(const char *value, const char *const *names, int count)
{
    int index = 0;
    while (index < count && strcmp(value, names[index]) != 0) ++index;
    return index;
}

/** The launch every loop benchmark takes from its environment. */
struct LkLaunch {
    int iterations;
    int num_blocks;
    int threads_per_block;
};

static LkLaunch lk_launch_params()
{
    LkLaunch launch;
    launch.iterations = lk_int_param("BENCH_PARAM_iterations", LK_DEFAULT_ITERATIONS);
    launch.num_blocks = lk_int_param("BENCH_PARAM_num_blocks", LK_DEFAULT_NUM_BLOCKS);
    launch.threads_per_block =
        lk_int_param("BENCH_PARAM_threads_per_block", LK_DEFAULT_THREADS_PER_BLOCK);

    /* The loop tests at the bottom, so a kernel runs at least one iteration. */
    if (launch.iterations < 1) lk_fail("iterations must be at least 1");

    /* 1024 is the workgroup size the kernels declare in their metadata. */
    if (launch.num_blocks <= 0 || launch.threads_per_block <= 0 ||
        launch.threads_per_block > 1024 ||
        static_cast<size_t>(launch.num_blocks) >
            std::numeric_limits<size_t>::max() / sizeof(uint64_t) /
                static_cast<size_t>(launch.threads_per_block)) {
        lk_fail("Invalid or overflowing launch dimensions");
    }
    return launch;
}

// ---------------------------------------------------------------------------
// The timed launch
// ---------------------------------------------------------------------------

/**
 * Whether a device can run the kernels. `arch_name` is the architecture HIP
 * reports: the processor name, optionally followed by features, as in
 * "gfx942:sramecc+:xnack-". Only the processor name is compared.
 */
static bool lk_is_kernel_arch(const char *arch_name)
{
    const size_t length = strlen(LK_KERNEL_ARCH);
    return strncmp(arch_name, LK_KERNEL_ARCH, length) == 0 &&
           (arch_name[length] == '\0' || arch_name[length] == ':');
}

/**
 * Launch `kernel_name` of `code_object` and time it EXACTLY ONCE, emit its
 * {"type":"kernel",...} record and the summary, and require every thread to
 * have stored `expected`; a mismatch is a fatal error. `workload_params` is
 * the JSON fragment of the benchmark's own parameters, placed between
 * "iterations" and the launch dimensions in the record.
 */
static void lk_run(const unsigned char *code_object, const char *kernel_name,
                   const char *workload_params, const LkLaunch &launch, uint64_t expected)
{
    const size_t total_threads =
        static_cast<size_t>(launch.num_blocks) * static_cast<size_t>(launch.threads_per_block);

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    if (!lk_is_kernel_arch(prop.gcnArchName)) {
        char message[384];
        snprintf(message, sizeof(message),
                 "The kernels are %s assembly; this device reports architecture %s",
                 LK_KERNEL_ARCH, prop.gcnArchName);
        lk_fail(message);
    }
    fprintf(stderr, "Device          : %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Kernel          : %s\n", kernel_name);
    fprintf(stderr, "Grid            : %d blocks × %d threads = %zu total threads\n",
            launch.num_blocks, launch.threads_per_block, total_threads);
    fprintf(stderr, "Iterations      : %d\n\n", launch.iterations);

    hipModule_t module;
    HIP_CHECK(hipModuleLoadData(&module, code_object));
    hipFunction_t kernel;
    HIP_CHECK(hipModuleGetFunction(&kernel, module, kernel_name));

    /* A sentinel no kernel stores, so a thread that stored nothing is caught. */
    uint64_t *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, total_threads * sizeof(uint64_t)));
    HIP_CHECK(hipMemset(d_out, 0xff, total_threads * sizeof(uint64_t)));

    char params[384];
    snprintf(params, sizeof(params),
             "\"iterations\":%d,%s,\"num_blocks\":%d,\"threads_per_block\":%d,"
             "\"verification\":\"full_cpu_v1\"",
             launch.iterations, workload_params, launch.num_blocks, launch.threads_per_block);

    int kernel_iterations = launch.iterations;
    void *kernel_args[] = {&d_out, &kernel_iterations};
    const double total_gpu_ms = bench_run_once(kernel_name, params, [&]() {
        HIP_CHECK(hipModuleLaunchKernel(kernel, launch.num_blocks, 1, 1,
                                        launch.threads_per_block, 1, 1, 0, 0, kernel_args,
                                        nullptr));
    });

    std::vector<uint64_t> out(total_threads);
    HIP_CHECK(hipMemcpy(out.data(), d_out, total_threads * sizeof(uint64_t),
                        hipMemcpyDeviceToHost));
    for (size_t i = 0; i < total_threads; ++i) {
        if (out[i] != expected) {
            fprintf(stderr, "thread %zu stored 0x%016llx, expected 0x%016llx\n", i,
                    static_cast<unsigned long long>(out[i]),
                    static_cast<unsigned long long>(expected));
            lk_fail("Stored result differs from the CPU reference");
        }
    }

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time  : %.3f ms\n", total_gpu_ms);

    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipModuleUnload(module));

    fprintf(stderr, "\nDone.\n");
}

#endif /* GFX942_LOOP_HOST_H */
