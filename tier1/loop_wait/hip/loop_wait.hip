/**
 * loop_wait.hip
 *
 * Host program of the loop_wait microbenchmark (AMD gfx942 only).
 *
 * The timed kernels are hand-written assembly (loop_wait_kernels.S), assembled
 * into a code object that the Makefile embeds in this executable. Every kernel
 * runs `ops_per_iter` copies of one VALU instruction and the three
 * loop-control instructions per iteration; `layout` says how the loop is laid
 * out. Comparing a layout with the instruction_cost loop of the same
 * instruction and `ops_per_iter` shows what that one change costs per
 * iteration.
 *
 * Environment variables:
 *   BENCH_PARAM_iterations         loop iterations per thread (>= 1)
 *   BENCH_PARAM_layout             how the loop is laid out
 *   BENCH_PARAM_instruction        VALU instruction in the loop body
 *   BENCH_PARAM_ops_per_iter       copies of the instruction per iteration
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
#include "loop_wait_kernels.h"

/* The assembled code object, embedded by the Makefile (ld -r -b binary). */
extern "C" const unsigned char _binary_loop_wait_kernels_hsaco_start[];

// ---------------------------------------------------------------------------
// The kernel set (single-sourced with the assembly)
// ---------------------------------------------------------------------------

#define LW_LAYOUT_NAME(layout, a) #layout,
#define LW_OPS_PER_ITER_VALUE(a, b, k) k,

static const char *const LAYOUT_NAMES[] = {LW_FOR_EACH_LAYOUT(LW_LAYOUT_NAME, )};
static const int LAYOUT_COUNT = sizeof(LAYOUT_NAMES) / sizeof(LAYOUT_NAMES[0]);

/* Loop-body sizes that have a kernel. */
static const int OPS_PER_ITER_VALUES[] = {LW_FOR_EACH_OPS_PER_ITER(LW_OPS_PER_ITER_VALUE, , )};

static const int DEFAULT_OPS_PER_ITER = 4;

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main()
{
    /* Line-buffer stdout so JSON lines are flushed promptly. */
    setvbuf(stdout, NULL, _IOLBF, 0);

    const char *layout = lk_string_param("BENCH_PARAM_layout", LAYOUT_NAMES[0]);
    const char *instruction_name =
        lk_string_param("BENCH_PARAM_instruction", LK_VALU_NAMES[0]);
    const int ops_per_iter = lk_int_param("BENCH_PARAM_ops_per_iter", DEFAULT_OPS_PER_ITER);

    if (lk_index_of(layout, LAYOUT_NAMES, LAYOUT_COUNT) == LAYOUT_COUNT) {
        lk_fail("Unknown layout");
    }

    const int instruction = lk_index_of(instruction_name, LK_VALU_NAMES, LK_VALU_COUNT);
    if (instruction == LK_VALU_COUNT) lk_fail("Unknown instruction");

    bool has_kernel = false;
    for (int k : OPS_PER_ITER_VALUES) has_kernel = has_kernel || k == ops_per_iter;
    if (!has_kernel) lk_fail("No kernel for this ops_per_iter");

    const LkLaunch launch = lk_launch_params();

    char kernel_name[64];
    snprintf(kernel_name, sizeof(kernel_name), "lw_%s_%s_k%d", layout, instruction_name,
             ops_per_iter);

    /* A layout moves instructions around but every copy is X = op(X, ...), so
     * X saw every VALU instruction the loop executed, whatever the layout. */
    const uint64_t expected = lk_reference(
        static_cast<LkValu>(instruction),
        static_cast<uint64_t>(ops_per_iter) * static_cast<uint64_t>(launch.iterations));

    char workload_params[160];
    snprintf(workload_params, sizeof(workload_params),
             "\"layout\":\"%s\",\"instruction\":\"%s\",\"ops_per_iter\":%d", layout,
             instruction_name, ops_per_iter);

    lk_run(_binary_loop_wait_kernels_hsaco_start, kernel_name, workload_params, launch,
           expected);
    return EXIT_SUCCESS;
}
