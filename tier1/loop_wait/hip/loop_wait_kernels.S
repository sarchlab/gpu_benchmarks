/**
 * loop_wait_kernels.S
 *
 * Hand-written gfx942 (MI300) kernels for the loop_wait benchmark. Nothing
 * here is compiler-generated: the timed loop of every kernel is exactly the
 * instruction sequence written below.
 *
 * The reference loop is the chained instruction_cost loop, whose head is on a
 * 64-byte boundary:
 *
 *     loop:
 *         <ops_per_iter copies of X = op(X, ...)>      V x K
 *         s_add_i32      s3, s3, -1                    S
 *         s_cmp_lg_u32   s3, 0                         S
 *         s_cbranch_scc1 loop                          B
 *
 * Each layout below changes one thing about it and keeps the rest: the same
 * K VALU instructions and the same three loop-control instructions per
 * iteration. One kernel exists per (layout, instruction, ops_per_iter):
 *
 *     lw_<layout>_<instruction>_k<ops_per_iter>(uint64_t *out, int iterations)
 *
 * A kernel runs its loop `iterations` times (at least 1; the loop tests at
 * the bottom), then stores X per thread to out[global thread index]. X is the
 * same in every layout, so the host compares it with one CPU reference. The
 * instructions, their operands, the kernel entry and exit and the metadata
 * entry are in gfx942_loop_kernels.h. Assemble with the C preprocessor
 * (clang -x assembler-with-cpp); see the Makefile.
 */

#include "loop_wait_kernels.h"

	.amdhsa_code_object_version 6

/* Scalar registers a kernel may use: s0..s5 and s[6:7], the scalar operand of
   the sgpr layouts. */
#define LW_SGPRS 8

/* `count` copies of X = op(X, ...). */
.macro lw_copies instruction, count
	.rept \count
	lk_op_\instruction	2
	.endr
.endm

/* ---- A scalar-register operand ------------------------------------------ */

/* Each instruction has lw_sgpr_init_<instruction>, which loads s6 (or s[6:7])
   with the bits of one of its vector operands, and lw_sgpr_op_<instruction>,
   the copy X = op(X, ...) that reads the scalar register in place of that
   operand. The operand is M for a multiply or an FMA and A for an add. A
   two-operand instruction takes a scalar register as its first source only,
   so there X moves to the second. */

.macro lw_sgpr_init_v_fma_f32
	s_mov_b32	s6, LK_F32_M
.endm
.macro lw_sgpr_op_v_fma_f32
	v_fma_f32	v2, v2, s6, v6
.endm

.macro lw_sgpr_init_v_mul_f32
	s_mov_b32	s6, LK_F32_M
.endm
.macro lw_sgpr_op_v_mul_f32
	v_mul_f32	v2, s6, v2
.endm

.macro lw_sgpr_init_v_add_f32
	s_mov_b32	s6, LK_F32_A
.endm
.macro lw_sgpr_op_v_add_f32
	v_add_f32	v2, s6, v2
.endm

.macro lw_sgpr_init_v_mul_lo_u32
	s_mov_b32	s6, LK_U32_M
.endm
.macro lw_sgpr_op_v_mul_lo_u32
	v_mul_lo_u32	v2, v2, s6
.endm

.macro lw_sgpr_init_v_add_u32
	s_mov_b32	s6, LK_U32_A
.endm
.macro lw_sgpr_op_v_add_u32
	v_add_u32	v2, s6, v2
.endm

.macro lw_sgpr_init_v_fma_f64
	s_mov_b32	s6, LK_F64_M_LO
	s_mov_b32	s7, LK_F64_M_HI
.endm
.macro lw_sgpr_op_v_fma_f64
	v_fma_f64	v[2:3], v[2:3], s[6:7], v[6:7]
.endm

.macro lw_sgpr_init_v_pk_fma_f32
	s_mov_b32	s6, LK_F32_M
	s_mov_b32	s7, LK_F32_M
.endm
.macro lw_sgpr_op_v_pk_fma_f32
	v_pk_fma_f32	v[2:3], v[2:3], s[6:7], v[6:7]
.endm

/* A move's only source is X, which a move never changes: s[6:7] holds the
   initial X. Unlike the other instructions' scalar-operand copies, this one
   therefore does not read what the previous copy wrote, so the sgpr layouts
   of v_mov_b64 also remove one read-after-write dependency per iteration. */
.macro lw_sgpr_init_v_mov_b64
	s_mov_b32	s6, LK_B64_LO
	s_mov_b32	s7, LK_B64_HI
.endm
.macro lw_sgpr_op_v_mov_b64
	v_mov_b64	v[2:3], s[6:7]
.endm

/* ---- The layouts -------------------------------------------------------- */

/* lw_loop_<layout> name, instruction, k is everything from the operands to
   the loop branch. */

/* The reference loop with its head `bytes` into a 64-byte line. The padding
   is s_nop before the loop, executed once. */
.macro lw_loop_head_offset name, instruction, k, bytes
	.p2alignl	6, 0xbf800000			// s_nop up to a 64-byte line
	.rept \bytes / 4
	s_nop	0
	.endr
.Lloop_\name:
	lw_copies	\instruction, \k
	s_add_i32	s3, s3, -1
	s_cmp_lg_u32	s3, 0
	s_cbranch_scc1	.Lloop_\name
.endm
.macro lw_loop_head_offset_8 name, instruction, k
	lw_loop_head_offset	\name, \instruction, \k, 8
.endm
.macro lw_loop_head_offset_20 name, instruction, k
	lw_loop_head_offset	\name, \instruction, \k, 20
.endm
.macro lw_loop_head_offset_28 name, instruction, k
	lw_loop_head_offset	\name, \instruction, \k, 28
.endm
.macro lw_loop_head_offset_52 name, instruction, k
	lw_loop_head_offset	\name, \instruction, \k, 52
.endm

/* S, V x K, S, B: the counter decrement at the loop head. */
.macro lw_loop_add_at_head name, instruction, k
	.p2alignl	6, 0xbf800000
.Lloop_\name:
	s_add_i32	s3, s3, -1
	lw_copies	\instruction, \k
	s_cmp_lg_u32	s3, 0
	s_cbranch_scc1	.Lloop_\name
.endm

/* V x (K-1), S, S, V, B: one VALU instruction between the compare and the
   branch. A VALU instruction leaves SCC alone. */
.macro lw_loop_valu_before_branch name, instruction, k
	.p2alignl	6, 0xbf800000
.Lloop_\name:
	lw_copies	\instruction, (\k - 1)
	s_add_i32	s3, s3, -1
	s_cmp_lg_u32	s3, 0
	lk_op_\instruction	2
	s_cbranch_scc1	.Lloop_\name
.endm

/* V, S, V x (K-2), S, V, B: both of the above, as compilers lay a loop out.
   The decrement follows the first VALU instruction. */
.macro lw_loop_compiled name, instruction, k
	.p2alignl	6, 0xbf800000
.Lloop_\name:
	lk_op_\instruction	2
	s_add_i32	s3, s3, -1
	lw_copies	\instruction, (\k - 2)
	s_cmp_lg_u32	s3, 0
	lk_op_\instruction	2
	s_cbranch_scc1	.Lloop_\name
.endm

/* The reference loop, its first VALU instruction (the branch target) reading
   a scalar register. */
.macro lw_loop_sgpr_at_head name, instruction, k
	lw_sgpr_init_\instruction
	.p2alignl	6, 0xbf800000
.Lloop_\name:
	lw_sgpr_op_\instruction
	lw_copies	\instruction, (\k - 1)
	s_add_i32	s3, s3, -1
	s_cmp_lg_u32	s3, 0
	s_cbranch_scc1	.Lloop_\name
.endm

/* The reference loop, VALU instruction K/2 (counting from 0) reading a scalar
   register. */
.macro lw_loop_sgpr_in_middle name, instruction, k
	lw_sgpr_init_\instruction
	.p2alignl	6, 0xbf800000
.Lloop_\name:
	lw_copies	\instruction, (\k / 2)
	lw_sgpr_op_\instruction
	lw_copies	\instruction, (\k - \k / 2 - 1)
	s_add_i32	s3, s3, -1
	s_cmp_lg_u32	s3, 0
	s_cbranch_scc1	.Lloop_\name
.endm

/* ---- The kernels -------------------------------------------------------- */

.macro lw_named_kernel name, layout, instruction, k
	lk_kernel_begin	\name
	lk_init_\instruction
	s_waitcnt	lgkmcnt(0)
	lw_loop_\layout	\name, \instruction, \k
	lk_kernel_end	\name, 2, LW_SGPRS
.endm
.macro lw_kernel layout, instruction, k
	lw_named_kernel	lw_\layout\()_\instruction\()_k\k, \layout, \instruction, \k
.endm

#define LW_IRP_LAYOUT(layout, a) , layout
#define LW_IRP_VALU(instruction) , instruction
#define LW_IRP_OPS_PER_ITER(a, b, k) , k

	.irp layout LW_FOR_EACH_LAYOUT(LW_IRP_LAYOUT, )
	.irp instruction LK_FOR_EACH_VALU(LW_IRP_VALU)
	.irp k LW_FOR_EACH_OPS_PER_ITER(LW_IRP_OPS_PER_ITER, , )
	lw_kernel	\layout, \instruction, \k
	.endr
	.endr
	.endr

	lk_text_end

/* ---- Code object metadata ----------------------------------------------- */

#define LW_METADATA_KERNEL(layout, instruction, k) \
    LK_METADATA(lw_##layout##_##instruction##_k##k, LW_SGPRS),
#define LW_METADATA_LAYOUT(layout, instruction) \
    LW_FOR_EACH_OPS_PER_ITER(LW_METADATA_KERNEL, layout, instruction)
#define LW_METADATA_VALU(instruction) \
    LW_FOR_EACH_LAYOUT(LW_METADATA_LAYOUT, instruction)

	.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels: [ LK_FOR_EACH_VALU(LW_METADATA_VALU) ]
...
	.end_amdgpu_metadata

	.section	".note.GNU-stack","",@progbits
