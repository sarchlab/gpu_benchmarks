/**
 * instruction_cost.hip
 *
 * Host program of the instruction_cost microbenchmark (AMD gfx942 only).
 *
 * The timed kernels are hand-written assembly (instruction_cost_kernels.S and
 * instruction_cost_other_kernels.S), assembled into two code objects that the
 * Makefile embeds in this executable. The loop body of a kernel is exactly
 * `ops_per_iter` instructions followed by three loop-control instructions, so
 * the time of a launch is a count of known instructions rather than of
 * whatever a compiler chose to emit. Sweeping `ops_per_iter` separates the
 * cost of one instruction (slope) from the cost of the loop control
 * (intercept).
 *
 * Environment variables:
 *   BENCH_PARAM_iterations         loop iterations per thread (>= 1)
 *   BENCH_PARAM_instruction        what the loop body is made of
 *   BENCH_PARAM_dependency         chained | independent
 *   BENCH_PARAM_ops_per_iter       instructions in the loop body;
 *                                  0 runs the loop control alone
 *   BENCH_PARAM_num_blocks
 *   BENCH_PARAM_threads_per_block
 *   BENCH_PARAM_vector_add_warmup  (default true; see bench_timing.h)
 *
 * Measurement: the selected kernel is launched and timed EXACTLY ONCE per
 * process. After the timed bracket every thread's stored result is compared
 * with a CPU reference; a mismatch is a fatal error. Output: one
 * {"type":"kernel",...} JSON line named after the launched kernel, then a
 * summary.
 */

#include "../../../common_hip/gfx942_loop_host.h"
#include "instruction_cost_kernels.h"

/* The assembled code objects, embedded by the Makefile (ld -r -b binary). */
extern "C" const unsigned char _binary_instruction_cost_kernels_hsaco_start[];
extern "C" const unsigned char _binary_instruction_cost_other_kernels_hsaco_start[];

// ---------------------------------------------------------------------------
// The loop bodies (single-sourced with the assembly)
// ---------------------------------------------------------------------------

#define IC_SCALAR_KIND(body) KIND_##body,

enum BodyKind {
    ONE_VALU, /* copies of one VALU instruction */
    TWO_VALU, /* two VALU instructions, alternating */
    IC_FOR_EACH_SCALAR(IC_SCALAR_KIND) /* copies of that scalar or branch instruction */
};

/* A value of the `instruction` parameter. `first` and `second` are the VALU
 * instructions of the first two kinds. */
struct Body {
    const char *name;
    BodyKind kind;
    LkValu first;
    LkValu second;
};

#define IC_VALU_BODY(instruction) \
    {#instruction, ONE_VALU, LK_##instruction, LK_##instruction},
#define IC_PAIR_BODY(body, first, second) {#body, TWO_VALU, LK_##first, LK_##second},
#define IC_SCALAR_BODY(body) {#body, KIND_##body, LK_VALU_COUNT, LK_VALU_COUNT},

static const Body BODIES[] = {LK_FOR_EACH_VALU(IC_VALU_BODY) IC_FOR_EACH_PAIR(IC_PAIR_BODY)
                                  IC_FOR_EACH_SCALAR(IC_SCALAR_BODY)};
static const int BODY_COUNT = sizeof(BODIES) / sizeof(BODIES[0]);

#define IC_OPS_PER_ITER_VALUE(a, b, k) , k

/* Loop-body sizes that have a kernel; 0 is ic_empty. */
static const int OPS_PER_ITER_VALUES[] = {
    0 IC_FOR_EACH_OPS_PER_ITER(IC_OPS_PER_ITER_VALUE, , )};

static const int DEFAULT_OPS_PER_ITER = 4;

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

/**
 * The 64-bit value a kernel of `ops_per_iter` > 0 instructions stores per
 * thread after `iterations` iterations.
 */
static uint64_t expected_result(const Body &body, bool chained, int ops_per_iter,
                                int iterations)
{
    const uint64_t executed =
        static_cast<uint64_t>(ops_per_iter) * static_cast<uint64_t>(iterations);
    /* A pair body runs its second instruction ops_per_iter / 2 times an
     * iteration, each time right after the first. */
    const uint64_t seconds =
        static_cast<uint64_t>(ops_per_iter / 2) * static_cast<uint64_t>(iterations);
    switch (body.kind) {
    case ONE_VALU:
        /* Chained, X saw every instruction the loop executed. Independent,
         * the stored register holds one application whatever the count. */
        return lk_reference(body.first, chained ? executed : 1);
    case TWO_VALU:
        /* Independent, the stored register is copy 0: one application of the
         * first instruction to the pair's operands. */
        if (body.first == LK_v_mul_lo_u32) {
            if (!chained) return lk_reference(LK_v_mul_lo_u32, 1);
            /* One copy is a multiply alone. */
            if (ops_per_iter == 1) return lk_reference(LK_v_mul_lo_u32, executed);
            /* X = X * M then X = X + A, over and over. */
            uint32_t x = LK_U32_X0;
            for (uint64_t i = 0; i < seconds; ++i) x = x * LK_U32_M + LK_U32_A;
            return x;
        }
        /* v_mov_b64 with v_fma_f64: X starts as the FMA's, and a move leaves
         * it alone, so only the FMAs count. */
        return lk_reference(LK_v_fma_f64, chained ? seconds : 0);
    case KIND_s_add_u32:
        /* The 32-bit counter the body incremented, zero-extended. */
        return static_cast<uint32_t>(executed);
    case KIND_s_cbranch_scc0:
    case KIND_s_branch:
        break;
    }
    return lk_reference(LK_v_mov_b64, 0); /* a branch leaves X as in ic_empty */
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main()
{
    /* Line-buffer stdout so JSON lines are flushed promptly. */
    setvbuf(stdout, NULL, _IOLBF, 0);

    const char *instruction_name = lk_string_param("BENCH_PARAM_instruction", BODIES[0].name);
    const char *dependency = lk_string_param("BENCH_PARAM_dependency", "chained");
    const int ops_per_iter = lk_int_param("BENCH_PARAM_ops_per_iter", DEFAULT_OPS_PER_ITER);

    int index = 0;
    while (index < BODY_COUNT && strcmp(instruction_name, BODIES[index].name) != 0) ++index;
    if (index == BODY_COUNT) lk_fail("Unknown instruction");
    const Body &body = BODIES[index];

    const bool chained = strcmp(dependency, "chained") == 0;
    if (!chained && strcmp(dependency, "independent") != 0) lk_fail("Unknown dependency");

    bool has_kernel = false;
    for (int k : OPS_PER_ITER_VALUES) has_kernel = has_kernel || k == ops_per_iter;
    if (!has_kernel) lk_fail("No kernel for this ops_per_iter");

    const LkLaunch launch = lk_launch_params();

    /* Zero copies is the same kernel for every instruction and dependency. A
     * scalar or branch body has one kernel for both dependencies. The VALU
     * bodies of one instruction and ic_empty are in the first code object. */
    char kernel_name[64];
    const unsigned char *code_object = _binary_instruction_cost_kernels_hsaco_start;
    uint64_t expected = lk_reference(LK_v_mov_b64, 0); /* ic_empty stores the untouched X */
    if (ops_per_iter == 0) {
        snprintf(kernel_name, sizeof(kernel_name), "ic_empty");
    } else {
        if (body.kind == ONE_VALU || body.kind == TWO_VALU) {
            snprintf(kernel_name, sizeof(kernel_name), "ic_%s_%s_k%d", body.name, dependency,
                     ops_per_iter);
        } else {
            snprintf(kernel_name, sizeof(kernel_name), "ic_%s_k%d", body.name, ops_per_iter);
        }
        if (body.kind != ONE_VALU) {
            code_object = _binary_instruction_cost_other_kernels_hsaco_start;
        }
        expected = expected_result(body, chained, ops_per_iter, launch.iterations);
    }

    char workload_params[160];
    snprintf(workload_params, sizeof(workload_params),
             "\"instruction\":\"%s\",\"dependency\":\"%s\",\"ops_per_iter\":%d", body.name,
             dependency, ops_per_iter);

    lk_run(code_object, kernel_name, workload_params, launch, expected);
    return EXIT_SUCCESS;
}
