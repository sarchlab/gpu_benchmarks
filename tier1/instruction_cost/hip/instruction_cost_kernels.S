/**
 * instruction_cost_kernels.S
 *
 * Hand-written gfx942 (MI300) kernels for the instruction_cost benchmark: the
 * loop bodies made of one VALU instruction, and the loop control alone.
 * Nothing here is compiler-generated: the timed loop of every kernel is
 * exactly the instruction sequence written below.
 *
 * These are the 81 kernels of GPU Benchmarks v1.2, and with the same ROCm
 * release this file assembles to the code object v1.2 measured, byte for
 * byte. The bodies added since are in instruction_cost_other_kernels.S, a
 * code object of their own, so that it stays so.
 *
 * One kernel exists per (instruction, dependency, ops_per_iter):
 *
 *     ic_<instruction>_<dependency>_k<ops_per_iter>(uint64_t *out, int iterations)
 *
 * plus ic_empty, whose loop holds the loop control alone. A kernel runs
 *
 *     loop:
 *         <ops_per_iter copies of the instruction>
 *         s_add_i32      s3, s3, -1
 *         s_cmp_lg_u32   s3, 0
 *         s_cbranch_scc1 loop
 *
 * `iterations` times (it must be at least 1; the loop tests at the bottom),
 * then stores one 64-bit result per thread to out[global thread index].
 *
 * Registers (per lane; 32-bit instructions use the low register of a pair):
 *     v[2:3]   X, the accumulator
 *     v[4:5]   M, a multiplier         loop-invariant
 *     v[6:7]   A, an addend            loop-invariant
 *     v[8:9], v[10:11], ...            destinations of the independent variant
 *
 * chained:      every copy is X = op(X, ...). Each instruction reads the
 *               register the previous one wrote, across iterations too.
 * independent:  copy j is v[8+2j : 9+2j] = op(X, ...). X is never written, so
 *               no instruction reads a register written inside the loop and no
 *               two copies share a destination.
 *
 * The stored result is X (chained, ic_empty) or v[8:9] (independent), which
 * the host compares with a CPU reference. The instructions, their operands,
 * the kernel entry and exit and the metadata entry are in
 * gfx942_loop_kernels.h. Assemble with the C preprocessor
 * (clang -x assembler-with-cpp); see the Makefile.
 */

#include "instruction_cost_kernels.h"

	.amdhsa_code_object_version 6

/* Scalar registers a kernel may use: s0..s5, the widest use here. */
#define IC_SGPRS 6

	.altmacro

/* Copies j..k-1 of the independent variant: copy j writes v[8+2j : 9+2j]. */
.macro ic_independent_copies instruction, k, j=0
	.if \j < \k
	lk_op_\instruction	%(8 + 2 * \j)
	ic_independent_copies	\instruction, \k, %(\j + 1)
	.endif
.endm

.macro ic_kernel name, instruction, dependency, k
	lk_kernel_begin	\name
	lk_init_\instruction
	s_waitcnt	lgkmcnt(0)
	.p2alignl	6, 0xbf800000			// s_nop up to a 64-byte line
.Lloop_\name:
	.ifc \dependency, chained
	.rept \k
	lk_op_\instruction	2
	.endr
	.else
	ic_independent_copies	\instruction, \k
	.endif
	s_add_i32	s3, s3, -1
	s_cmp_lg_u32	s3, 0
	s_cbranch_scc1	.Lloop_\name
	.ifc \dependency, chained
	lk_kernel_end	\name, 2, IC_SGPRS
	.else
	lk_kernel_end	\name, 8, IC_SGPRS
	.endif
.endm

/* The loop control alone. It borrows the v_mov_b64 operands so that it has a
   defined X to store. */
	ic_kernel	ic_empty, v_mov_b64, chained, 0

#define IC_IRP_INSTRUCTION(instruction) , instruction
#define IC_IRP_OPS_PER_ITER(a, b, k) , k

	.irp instruction LK_FOR_EACH_VALU(IC_IRP_INSTRUCTION)
	.irp k IC_FOR_EACH_OPS_PER_ITER(IC_IRP_OPS_PER_ITER, , )
	ic_kernel	ic_\instruction\()_chained_k\k, \instruction, chained, \k
	ic_kernel	ic_\instruction\()_independent_k\k, \instruction, independent, \k
	.endr
	.endr

	lk_text_end

/* ---- Code object metadata ----------------------------------------------- */

#define IC_METADATA_KERNEL(instruction, dependency, k) \
    LK_METADATA(ic_##instruction##_##dependency##_k##k, IC_SGPRS),
#define IC_METADATA_INSTRUCTION(instruction)                                  \
    IC_FOR_EACH_OPS_PER_ITER(IC_METADATA_KERNEL, instruction, chained)        \
    IC_FOR_EACH_OPS_PER_ITER(IC_METADATA_KERNEL, instruction, independent)

	.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels: [ LK_FOR_EACH_VALU(IC_METADATA_INSTRUCTION) LK_METADATA(ic_empty, IC_SGPRS) ]
...
	.end_amdgpu_metadata

	.section	".note.GNU-stack","",@progbits
