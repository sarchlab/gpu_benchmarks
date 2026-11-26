/**
 * instruction_cost_kernels.h
 *
 * The kernel set and operand constants of the instruction_cost benchmark,
 * shared by the hand-written kernels (instruction_cost_kernels.S, which the
 * assembler runs through the C preprocessor) and the host program
 * (instruction_cost.hip). It holds preprocessor definitions only, so both
 * languages can include it and neither repeats a list or a constant.
 */

#ifndef INSTRUCTION_COST_KERNELS_H
#define INSTRUCTION_COST_KERNELS_H

/* The measured instructions: X(instruction) for each. */
#define IC_FOR_EACH_INSTRUCTION(X) \
    X(v_fma_f32)                   \
    X(v_mul_f32)                   \
    X(v_add_f32)                   \
    X(v_mul_lo_u32)                \
    X(v_add_u32)                   \
    X(v_fma_f64)                   \
    X(v_pk_fma_f32)                \
    X(v_mov_b64)

/* The loop-body sizes that have a kernel per instruction and dependency:
 * X(instruction, dependency, ops_per_iter) for each, with the first two
 * arguments passed through. Zero copies is the shared ic_empty kernel. */
#define IC_FOR_EACH_OPS_PER_ITER(X, instruction, dependency) \
    X(instruction, dependency, 1)                            \
    X(instruction, dependency, 2)                            \
    X(instruction, dependency, 4)                            \
    X(instruction, dependency, 8)                            \
    X(instruction, dependency, 16)

/* Initial operand bit patterns. Every thread starts from the same values. */
#define IC_F32_X0    0x3f800000 /* 1.0f: accumulator */
#define IC_F32_X0_HI 0x3f000000 /* 0.5f: accumulator, upper v_pk_fma_f32 lane */
#define IC_F32_M     0x3f800001 /* 1.0000001f: multiplier */
#define IC_F32_A     0x33d6bf95 /* 1.0e-7f: addend */

#define IC_U32_X0 1          /* accumulator */
#define IC_U32_M  1664525    /* multiplier (odd) */
#define IC_U32_A  1013904223 /* addend */

#define IC_F64_X0_LO 0x00000000 /* 1.0: accumulator */
#define IC_F64_X0_HI 0x3ff00000
#define IC_F64_M_LO  0x1ad7f29b /* 1.0000001: multiplier */
#define IC_F64_M_HI  0x3ff00000
#define IC_F64_A_LO  0x9abcaf48 /* 1.0e-7: addend */
#define IC_F64_A_HI  0x3e7ad7f2

#define IC_B64_LO 0x89abcdef /* v_mov_b64 and ic_empty: an arbitrary pattern */
#define IC_B64_HI 0x01234567

#endif /* INSTRUCTION_COST_KERNELS_H */
