/**
 * instruction_cost_kernels.S
 *
 * Hand-written gfx942 (MI300) kernels for the instruction_cost benchmark.
 * Nothing here is compiler-generated: the timed loop of every kernel is
 * exactly the instruction sequence written below.
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
 * the host compares with a CPU reference. Assemble with the C preprocessor
 * (clang -x assembler-with-cpp); see the Makefile.
 */

#include "instruction_cost_kernels.h"

	.amdhsa_code_object_version 6

/* ---- Operands ----------------------------------------------------------- */

/* X, M, A for a 32-bit instruction. v3 and v9 are the upper halves of the two
   registers a result can be stored from, so they are cleared. */
.macro ic_init_32 x, m, a
	v_mov_b32	v2, \x
	v_mov_b32	v3, 0
	v_mov_b32	v4, \m
	v_mov_b32	v6, \a
	v_mov_b32	v9, 0
.endm

/* X, M, A for an instruction that works on a register pair. */
.macro ic_init_64 x_lo, x_hi, m_lo, m_hi, a_lo, a_hi
	v_mov_b32	v2, \x_lo
	v_mov_b32	v3, \x_hi
	v_mov_b32	v4, \m_lo
	v_mov_b32	v5, \m_hi
	v_mov_b32	v6, \a_lo
	v_mov_b32	v7, \a_hi
.endm

/* ---- The measured instructions ------------------------------------------ */

/* Each instruction has ic_init_<instruction>, which loads its operands, and
   ic_op_<instruction> d, one copy writing v<d> (or v[d:d+1]). d is 2 for the
   chained variant, making the copy X = op(X, ...). */

.macro ic_init_v_fma_f32
	ic_init_32	IC_F32_X0, IC_F32_M, IC_F32_A
.endm
.macro ic_op_v_fma_f32 d
	v_fma_f32	v\d, v2, v4, v6
.endm

.macro ic_init_v_mul_f32
	ic_init_32	IC_F32_X0, IC_F32_M, IC_F32_A
.endm
.macro ic_op_v_mul_f32 d
	v_mul_f32	v\d, v2, v4
.endm

.macro ic_init_v_add_f32
	ic_init_32	IC_F32_X0, IC_F32_M, IC_F32_A
.endm
.macro ic_op_v_add_f32 d
	v_add_f32	v\d, v2, v6
.endm

.macro ic_init_v_mul_lo_u32
	ic_init_32	IC_U32_X0, IC_U32_M, IC_U32_A
.endm
.macro ic_op_v_mul_lo_u32 d
	v_mul_lo_u32	v\d, v2, v4
.endm

.macro ic_init_v_add_u32
	ic_init_32	IC_U32_X0, IC_U32_M, IC_U32_A
.endm
.macro ic_op_v_add_u32 d
	v_add_u32	v\d, v2, v6
.endm

.macro ic_init_v_fma_f64
	ic_init_64	IC_F64_X0_LO, IC_F64_X0_HI, IC_F64_M_LO, IC_F64_M_HI, IC_F64_A_LO, IC_F64_A_HI
.endm
.macro ic_op_v_fma_f64 d
	v_fma_f64	v[\d:\d+1], v[2:3], v[4:5], v[6:7]
.endm

/* Two f32 lanes per register pair: the low registers hold one FMA's operands,
   the high registers the other's. M and A are the same in both lanes. */
.macro ic_init_v_pk_fma_f32
	ic_init_64	IC_F32_X0, IC_F32_X0_HI, IC_F32_M, IC_F32_M, IC_F32_A, IC_F32_A
.endm
.macro ic_op_v_pk_fma_f32 d
	v_pk_fma_f32	v[\d:\d+1], v[2:3], v[4:5], v[6:7]
.endm

/* A move has one source. Chained, it copies X onto itself: the value never
   changes, but each move still reads what the previous one wrote. */
.macro ic_init_v_mov_b64
	ic_init_64	IC_B64_LO, IC_B64_HI, 0, 0, 0, 0
.endm
.macro ic_op_v_mov_b64 d
	v_mov_b64	v[\d:\d+1], v[2:3]
.endm

/* ---- Kernel ------------------------------------------------------------- */

	.altmacro

/* Copies j..k-1 of the independent variant: copy j writes v[8+2j : 9+2j]. */
.macro ic_independent_copies instruction, k, j=0
	.if \j < \k
	ic_op_\instruction	%(8 + 2 * \j)
	ic_independent_copies	\instruction, \k, %(\j + 1)
	.endif
.endm

/* Kernel entry state (the descriptor below requests it): s[0:1] = kernarg
   segment, s2 = workgroup id x, v0 = thread id within the workgroup. The
   kernarg segment is { out, iterations, hidden arguments }, laid out as hipcc
   lays it out for `(uint64_t *out, int iterations)`. */
.macro ic_kernel name, instruction, dependency, k
	.text
	.protected	\name
	.globl	\name
	.p2align	8
	.type	\name,@function
\name:
	s_load_dword	s3, s[0:1], 0x8			// s3 = iterations
	ic_init_\instruction
	s_waitcnt	lgkmcnt(0)
	.p2alignl	6, 0xbf800000			// s_nop up to a 64-byte line
.Lloop_\name:
	.ifc \dependency, chained
	.rept \k
	ic_op_\instruction	2
	.endr
	.else
	ic_independent_copies	\instruction, \k
	.endif
	s_add_i32	s3, s3, -1
	s_cmp_lg_u32	s3, 0
	s_cbranch_scc1	.Lloop_\name

	/* out[workgroup id x * workgroup size x + v0] = result */
	s_load_dword	s3, s[0:1], 0x1c		// hidden_group_size_x (16 bits)
	s_load_dwordx2	s[4:5], s[0:1], 0x0		// out
	v_lshlrev_b32	v0, 3, v0
	s_waitcnt	lgkmcnt(0)
	s_and_b32	s0, s3, 0xffff
	s_mul_hi_u32	s1, s0, s2
	s_mul_i32	s0, s0, s2
	s_lshl_b64	s[0:1], s[0:1], 3
	s_add_u32	s0, s4, s0
	s_addc_u32	s1, s5, s1
	.ifc \dependency, chained
	global_store_dwordx2	v0, v[2:3], s[0:1]
	.else
	global_store_dwordx2	v0, v[8:9], s[0:1]
	.endif
	s_endpgm
.Lend_\name:
	.size	\name, .Lend_\name-\name

	/* Kernel descriptor: the values hipcc emits for an equivalent kernel,
	   with the register counts of the widest kernel here (v0..v39, s0..s5). */
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel \name
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 272
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 40
		.amdhsa_next_free_sgpr 6
		.amdhsa_accum_offset 40
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
	.end_amdhsa_kernel
.endm

/* The loop control alone. It borrows the v_mov_b64 operands so that it has a
   defined X to store. */
	ic_kernel	ic_empty, v_mov_b64, chained, 0

#define IC_IRP_INSTRUCTION(instruction) , instruction
#define IC_IRP_OPS_PER_ITER(instruction, dependency, k) , k

	.irp instruction IC_FOR_EACH_INSTRUCTION(IC_IRP_INSTRUCTION)
	.irp k IC_FOR_EACH_OPS_PER_ITER(IC_IRP_OPS_PER_ITER, , )
	ic_kernel	ic_\instruction\()_chained_k\k, \instruction, chained, \k
	ic_kernel	ic_\instruction\()_independent_k\k, \instruction, independent, \k
	.endr
	.endr

	/* The instruction prefetcher reads past the last kernel: pad with s_nop,
	   as the compiler does. */
	.text
	.p2alignl	6, 0xbf800000
	.fill	256, 4, 0xbf800000

/* ---- Code object metadata ----------------------------------------------- */

/* The HIP runtime reads each kernel's argument layout from this note. The
   layout is the one hipcc emits for `(uint64_t *out, int iterations)`: two
   explicit arguments, then the hidden launch-geometry arguments. Every kernel
   has the same entry, written in YAML flow style so that the preprocessor can
   stamp it out from the kernel list. */
#define IC_METADATA_ARGS                                                  \
    [ {.offset: 0, .size: 8, .value_kind: global_buffer, .address_space: global}, \
      {.offset: 8, .size: 4, .value_kind: by_value},                      \
      {.offset: 16, .size: 4, .value_kind: hidden_block_count_x},         \
      {.offset: 20, .size: 4, .value_kind: hidden_block_count_y},         \
      {.offset: 24, .size: 4, .value_kind: hidden_block_count_z},         \
      {.offset: 28, .size: 2, .value_kind: hidden_group_size_x},          \
      {.offset: 30, .size: 2, .value_kind: hidden_group_size_y},          \
      {.offset: 32, .size: 2, .value_kind: hidden_group_size_z},          \
      {.offset: 34, .size: 2, .value_kind: hidden_remainder_x},           \
      {.offset: 36, .size: 2, .value_kind: hidden_remainder_y},           \
      {.offset: 38, .size: 2, .value_kind: hidden_remainder_z},           \
      {.offset: 56, .size: 8, .value_kind: hidden_global_offset_x},       \
      {.offset: 64, .size: 8, .value_kind: hidden_global_offset_y},       \
      {.offset: 72, .size: 8, .value_kind: hidden_global_offset_z},       \
      {.offset: 80, .size: 2, .value_kind: hidden_grid_dims} ]

#define IC_METADATA(KERNEL)                                               \
    { .name: KERNEL, .symbol: KERNEL.kd, .args: IC_METADATA_ARGS,         \
      .kernarg_segment_size: 272, .kernarg_segment_align: 8,              \
      .group_segment_fixed_size: 0, .private_segment_fixed_size: 0,       \
      .sgpr_count: 6, .sgpr_spill_count: 0, .vgpr_count: 40,              \
      .vgpr_spill_count: 0, .agpr_count: 0, .wavefront_size: 64,          \
      .max_flat_workgroup_size: 1024, .uniform_work_group_size: 1,        \
      .uses_dynamic_stack: false }

#define IC_METADATA_KERNEL(instruction, dependency, k) \
    IC_METADATA(ic_##instruction##_##dependency##_k##k),
#define IC_METADATA_INSTRUCTION(instruction)                                  \
    IC_FOR_EACH_OPS_PER_ITER(IC_METADATA_KERNEL, instruction, chained)        \
    IC_FOR_EACH_OPS_PER_ITER(IC_METADATA_KERNEL, instruction, independent)

	.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels: [ IC_FOR_EACH_INSTRUCTION(IC_METADATA_INSTRUCTION) IC_METADATA(ic_empty) ]
...
	.end_amdgpu_metadata

	.section	".note.GNU-stack","",@progbits
