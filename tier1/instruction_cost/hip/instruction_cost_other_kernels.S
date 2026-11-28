/**
 * instruction_cost_other_kernels.S
 *
 * Hand-written gfx942 (MI300) kernels for the instruction_cost benchmark: the
 * loop bodies that are not copies of one VALU instruction. They are a code
 * object of their own so that instruction_cost_kernels.S keeps assembling to
 * the bytes GPU Benchmarks v1.2 measured.
 *
 * Every kernel is the loop of instruction_cost_kernels.S, on a 64-byte
 * boundary, with another body:
 *
 *     loop:
 *         <ops_per_iter instructions>
 *         s_add_i32      s3, s3, -1
 *         s_cmp_lg_u32   s3, 0
 *         s_cbranch_scc1 loop
 *
 * Two VALU instructions, alternating and starting with the first:
 *
 *     ic_<first>_<second>_<dependency>_k<ops_per_iter>
 *
 *     chained:      every copy writes X, v[2:3].
 *     independent:  copy j writes v[8+2j : 9+2j]; X is never written.
 *
 * One scalar or branch instruction, repeated:
 *
 *     ic_<instruction>_k<ops_per_iter>
 *
 *     s_add_u32        s_add_u32 s6, s6, 1; s6 starts at 0
 *     s_cbranch_scc0   a conditional branch that is never taken: SCC is 1
 *                      throughout the body
 *     s_branch         an unconditional branch
 *
 * Both branches target the instruction that follows them.
 *
 * The stored result is X (chained), v[8:9] (independent), s6 (s_add_u32) or
 * the untouched X of ic_empty (the branches), which the host compares with a
 * CPU reference. Assemble with the C preprocessor
 * (clang -x assembler-with-cpp); see the Makefile.
 */

#include "instruction_cost_kernels.h"

	.amdhsa_code_object_version 6

/* Scalar registers a kernel may use: s0..s5 and s6, the s_add_u32 counter. */
#define IC_SGPRS 7

	.altmacro

/* ---- Two VALU instructions ---------------------------------------------- */

/* The operands of a pair. v_mul_lo_u32 and v_add_u32 share theirs; a move
   takes whatever X the FMA starts from. */
.macro ic_init_v_mul_lo_u32_v_add_u32
	lk_init_v_mul_lo_u32
.endm
.macro ic_init_v_mov_b64_v_fma_f64
	lk_init_v_fma_f64
.endm

/* Copies j..k-1, two per step: even copies are `first`, odd copies `second`.
   Chained, a copy writes X; independent, copy j writes v[8+2j : 9+2j]. */
.macro ic_pair_copies first, second, dependency, k, j=0
	.if \j < \k
	.ifc \dependency, chained
	lk_op_\first	2
	.else
	lk_op_\first	%(8 + 2 * \j)
	.endif
	.endif
	.if \j + 1 < \k
	.ifc \dependency, chained
	lk_op_\second	2
	.else
	lk_op_\second	%(10 + 2 * \j)
	.endif
	ic_pair_copies	\first, \second, \dependency, \k, %(\j + 2)
	.endif
.endm

.macro ic_pair_kernel name, body, first, second, dependency, k
	lk_kernel_begin	\name
	ic_init_\body
	s_waitcnt	lgkmcnt(0)
	.p2alignl	6, 0xbf800000			// s_nop up to a 64-byte line
.Lloop_\name:
	ic_pair_copies	\first, \second, \dependency, \k
	s_add_i32	s3, s3, -1
	s_cmp_lg_u32	s3, 0
	s_cbranch_scc1	.Lloop_\name
	.ifc \dependency, chained
	lk_kernel_end	\name, 2, IC_SGPRS
	.else
	lk_kernel_end	\name, 8, IC_SGPRS
	.endif
.endm

/* ---- One scalar or branch instruction ----------------------------------- */

/* Each body has ic_setup_<body>, run once before the loop, ic_op_<body>, one
   copy, and ic_result_<body>, run once after the loop to put what the kernel
   stores into v[2:3]. */

.macro ic_setup_s_add_u32
	s_mov_b32	s6, 0
.endm
.macro ic_op_s_add_u32
	s_add_u32	s6, s6, 1
.endm
.macro ic_result_s_add_u32
	v_mov_b32	v2, s6
	v_mov_b32	v3, 0
.endm

/* SCC is 1 on entry to the loop, and the loop branch is taken only when it is
   1, so the body always sees 1 and never branches. */
.macro ic_setup_s_cbranch_scc0
	s_cmp_eq_u32	s3, s3
.endm
.macro ic_op_s_cbranch_scc0
	s_cbranch_scc0	0f
0:
.endm
.macro ic_result_s_cbranch_scc0
.endm

.macro ic_setup_s_branch
.endm
.macro ic_op_s_branch
	s_branch	0f
0:
.endm
.macro ic_result_s_branch
.endm

/* The kernel borrows the v_mov_b64 operands, as ic_empty does, so that it has
   a defined X to store. */
.macro ic_scalar_kernel name, body, k
	lk_kernel_begin	\name
	lk_init_v_mov_b64
	s_waitcnt	lgkmcnt(0)
	ic_setup_\body
	.p2alignl	6, 0xbf800000			// s_nop up to a 64-byte line
.Lloop_\name:
	.rept \k
	ic_op_\body
	.endr
	s_add_i32	s3, s3, -1
	s_cmp_lg_u32	s3, 0
	s_cbranch_scc1	.Lloop_\name
	ic_result_\body
	lk_kernel_end	\name, 2, IC_SGPRS
.endm

/* ---- The kernels -------------------------------------------------------- */

#define IC_IRP_OPS_PER_ITER(a, b, k) , k

/* The kernels of every pair in a list of `body, first, second,`. */
.macro ic_pair_kernels body, first, second, rest:vararg
	.irp count IC_FOR_EACH_OPS_PER_ITER(IC_IRP_OPS_PER_ITER, , )
	ic_pair_kernel	ic_\body\()_chained_k\count, \body, \first, \second, chained, \count
	ic_pair_kernel	ic_\body\()_independent_k\count, \body, \first, \second, independent, \count
	.endr
	.ifnb \rest
	ic_pair_kernels	\rest
	.endif
.endm

#define IC_PAIR_ARGUMENTS(body, first, second) body, first, second,

	ic_pair_kernels	IC_FOR_EACH_PAIR(IC_PAIR_ARGUMENTS)

#define IC_IRP_SCALAR(body) , body

	.irp body IC_FOR_EACH_SCALAR(IC_IRP_SCALAR)
	.irp count IC_FOR_EACH_OPS_PER_ITER(IC_IRP_OPS_PER_ITER, , )
	ic_scalar_kernel	ic_\body\()_k\count, \body, \count
	.endr
	.endr

	lk_text_end

/* ---- Code object metadata ----------------------------------------------- */

#define IC_METADATA_PAIR_KERNEL(body, dependency, k) \
    LK_METADATA(ic_##body##_##dependency##_k##k, IC_SGPRS),
#define IC_METADATA_PAIR(body, first, second)                          \
    IC_FOR_EACH_OPS_PER_ITER(IC_METADATA_PAIR_KERNEL, body, chained)   \
    IC_FOR_EACH_OPS_PER_ITER(IC_METADATA_PAIR_KERNEL, body, independent)

#define IC_METADATA_SCALAR_KERNEL(body, unused, k) \
    LK_METADATA(ic_##body##_k##k, IC_SGPRS),
#define IC_METADATA_SCALAR(body) \
    IC_FOR_EACH_OPS_PER_ITER(IC_METADATA_SCALAR_KERNEL, body, )

	.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels: [ IC_FOR_EACH_PAIR(IC_METADATA_PAIR) IC_FOR_EACH_SCALAR(IC_METADATA_SCALAR) ]
...
	.end_amdgpu_metadata

	.section	".note.GNU-stack","",@progbits
