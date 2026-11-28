/**
 * instruction_cost_kernels.h
 *
 * The kernel set of the instruction_cost benchmark, shared by the hand-written
 * kernels (instruction_cost_kernels.S and instruction_cost_other_kernels.S,
 * which the assembler runs through the C preprocessor) and the host program
 * (instruction_cost.hip). It holds preprocessor definitions only, so both
 * languages can include it and neither repeats a list.
 *
 * A loop body is one of three kinds. The eight VALU instructions, their
 * operands and the kernel scaffolding come from gfx942_loop_kernels.h, which
 * loop_wait shares.
 */

#ifndef INSTRUCTION_COST_KERNELS_H
#define INSTRUCTION_COST_KERNELS_H

#include "../../../common_hip/gfx942_loop_kernels.h"

/* Bodies of one VALU instruction: LK_FOR_EACH_VALU(X), X(instruction) for
 * each. Their kernels and ic_empty are in instruction_cost_kernels.S. */

/* Bodies that alternate two VALU instructions, starting with the first:
 * X(body, first, second) for each. */
#define IC_FOR_EACH_PAIR(X)                              \
    X(v_mul_lo_u32_v_add_u32, v_mul_lo_u32, v_add_u32)   \
    X(v_mov_b64_v_fma_f64, v_mov_b64, v_fma_f64)

/* Bodies of one scalar or branch instruction: X(body) for each. They have one
 * kernel per ops_per_iter, whatever the dependency. */
#define IC_FOR_EACH_SCALAR(X) \
    X(s_add_u32)              \
    X(s_cbranch_scc0)         \
    X(s_branch)

/* The loop-body sizes that have a kernel per body: X(a, b, ops_per_iter) for
 * each, with the first two arguments passed through. Zero copies is the
 * shared ic_empty kernel. */
#define IC_FOR_EACH_OPS_PER_ITER(X, a, b) \
    X(a, b, 1)                            \
    X(a, b, 2)                            \
    X(a, b, 4)                            \
    X(a, b, 8)                            \
    X(a, b, 16)

#endif /* INSTRUCTION_COST_KERNELS_H */
