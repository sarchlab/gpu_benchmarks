/**
 * loop_wait_kernels.h
 *
 * The kernel set of the loop_wait benchmark, shared by the hand-written
 * kernels (loop_wait_kernels.S, which the assembler runs through the C
 * preprocessor) and the host program (loop_wait.hip). It holds preprocessor
 * definitions only, so both languages can include it and neither repeats a
 * list.
 *
 * The eight VALU instructions, their operands and the kernel scaffolding come
 * from gfx942_loop_kernels.h, which instruction_cost shares.
 */

#ifndef LOOP_WAIT_KERNELS_H
#define LOOP_WAIT_KERNELS_H

#include "../../../common_hip/gfx942_loop_kernels.h"

/* The loop layouts: X(layout, a) for each, with the second argument passed
 * through. loop_wait_kernels.S has one lw_loop_<layout> macro per layout. */
#define LW_FOR_EACH_LAYOUT(X, a) \
    X(head_offset_8, a)          \
    X(head_offset_20, a)         \
    X(head_offset_28, a)         \
    X(head_offset_52, a)         \
    X(add_at_head, a)            \
    X(valu_before_branch, a)     \
    X(compiled, a)               \
    X(sgpr_at_head, a)           \
    X(sgpr_in_middle, a)

/* The numbers of VALU instructions per iteration that have a kernel per
 * layout and instruction: X(a, b, ops_per_iter) for each, with the first two
 * arguments passed through. Every layout needs at least two. */
#define LW_FOR_EACH_OPS_PER_ITER(X, a, b) \
    X(a, b, 2)                            \
    X(a, b, 4)                            \
    X(a, b, 8)                            \
    X(a, b, 16)

#endif /* LOOP_WAIT_KERNELS_H */
