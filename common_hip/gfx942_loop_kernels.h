/**
 * gfx942_loop_kernels.h
 *
 * What the hand-written gfx942 (MI300) loop kernels of instruction_cost and
 * loop_wait have in common, so that it exists once: the measured VALU
 * instructions and their operands, the kernel entry, the result store and
 * kernel descriptor, and the code-object metadata entry.
 *
 * Both languages include it. The preprocessor definitions serve the kernel
 * sources (*.S, which the assembler runs through the C preprocessor) and the
 * host programs; the assembler macros exist only when assembling.
 *
 * Every kernel has the signature `(uint64_t *out, int iterations)`, counts its
 * loop down in s3, and stores one 64-bit result per thread:
 *
 *         lk_kernel_begin  <name>          // symbol; s3 = iterations (requested)
 *         lk_init_<instruction>            // operands
 *         s_waitcnt        lgkmcnt(0)
 *         .p2alignl        6, 0xbf800000   // s_nop up to a 64-byte line
 *     loop:
 *         ...                              // the timed loop, written by the benchmark
 *         s_add_i32        s3, s3, -1
 *         s_cmp_lg_u32     s3, 0
 *         s_cbranch_scc1   loop
 *         lk_kernel_end    <name>, <result>, <sgprs>
 *
 * Registers (per lane; 32-bit instructions use the low register of a pair):
 *     v[2:3]   X, the accumulator
 *     v[4:5]   M, a multiplier         loop-invariant
 *     v[6:7]   A, an addend            loop-invariant
 */

#ifndef GFX942_LOOP_KERNELS_H
#define GFX942_LOOP_KERNELS_H

/* The measured VALU instructions: X(instruction) for each. */
#define LK_FOR_EACH_VALU(X) \
    X(v_fma_f32)            \
    X(v_mul_f32)            \
    X(v_add_f32)            \
    X(v_mul_lo_u32)         \
    X(v_add_u32)            \
    X(v_fma_f64)            \
    X(v_pk_fma_f32)         \
    X(v_mov_b64)

/* Initial operand bit patterns. Every thread starts from the same values. */
#define LK_F32_X0    0x3f800000 /* 1.0f: accumulator */
#define LK_F32_X0_HI 0x3f000000 /* 0.5f: accumulator, upper v_pk_fma_f32 lane */
#define LK_F32_M     0x3f800001 /* 1.0000001f: multiplier */
#define LK_F32_A     0x33d6bf95 /* 1.0e-7f: addend */

#define LK_U32_X0 1          /* accumulator */
#define LK_U32_M  1664525    /* multiplier (odd) */
#define LK_U32_A  1013904223 /* addend */

#define LK_F64_X0_LO 0x00000000 /* 1.0: accumulator */
#define LK_F64_X0_HI 0x3ff00000
#define LK_F64_M_LO  0x1ad7f29b /* 1.0000001: multiplier */
#define LK_F64_M_HI  0x3ff00000
#define LK_F64_A_LO  0x9abcaf48 /* 1.0e-7: addend */
#define LK_F64_A_HI  0x3e7ad7f2

#define LK_B64_LO 0x89abcdef /* v_mov_b64: an arbitrary pattern */
#define LK_B64_HI 0x01234567

/* ---- Code object metadata ----------------------------------------------- */

/* The HIP runtime reads each kernel's argument layout from the metadata note.
   The layout is the one hipcc emits for `(uint64_t *out, int iterations)`: two
   explicit arguments, then the hidden launch-geometry arguments. Every kernel
   has the same entry, written in YAML flow style so that the preprocessor can
   stamp it out from a kernel list. */
#define LK_METADATA_ARGS                                                  \
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

/* The metadata entry of one kernel. SGPRS is the number of scalar registers
   the kernel may use, as given to lk_kernel_end. */
#define LK_METADATA(KERNEL, SGPRS)                                        \
    { .name: KERNEL, .symbol: KERNEL.kd, .args: LK_METADATA_ARGS,         \
      .kernarg_segment_size: 272, .kernarg_segment_align: 8,              \
      .group_segment_fixed_size: 0, .private_segment_fixed_size: 0,       \
      .sgpr_count: SGPRS, .sgpr_spill_count: 0, .vgpr_count: 40,          \
      .vgpr_spill_count: 0, .agpr_count: 0, .wavefront_size: 64,          \
      .max_flat_workgroup_size: 1024, .uniform_work_group_size: 1,        \
      .uses_dynamic_stack: false }

#ifdef __ASSEMBLER__

/* ---- Operands ----------------------------------------------------------- */

/* X, M, A for a 32-bit instruction. v3 and v9 are the upper halves of the two
   registers a result can be stored from, so they are cleared. */
.macro lk_init_32 x, m, a
	v_mov_b32	v2, \x
	v_mov_b32	v3, 0
	v_mov_b32	v4, \m
	v_mov_b32	v6, \a
	v_mov_b32	v9, 0
.endm

/* X, M, A for an instruction that works on a register pair. */
.macro lk_init_64 x_lo, x_hi, m_lo, m_hi, a_lo, a_hi
	v_mov_b32	v2, \x_lo
	v_mov_b32	v3, \x_hi
	v_mov_b32	v4, \m_lo
	v_mov_b32	v5, \m_hi
	v_mov_b32	v6, \a_lo
	v_mov_b32	v7, \a_hi
.endm

/* ---- The measured VALU instructions ------------------------------------- */

/* Each instruction has lk_init_<instruction>, which loads its operands, and
   lk_op_<instruction> d, one copy writing v<d> (or v[d:d+1]). With d = 2 the
   copy is X = op(X, ...). */

.macro lk_init_v_fma_f32
	lk_init_32	LK_F32_X0, LK_F32_M, LK_F32_A
.endm
.macro lk_op_v_fma_f32 d
	v_fma_f32	v\d, v2, v4, v6
.endm

.macro lk_init_v_mul_f32
	lk_init_32	LK_F32_X0, LK_F32_M, LK_F32_A
.endm
.macro lk_op_v_mul_f32 d
	v_mul_f32	v\d, v2, v4
.endm

.macro lk_init_v_add_f32
	lk_init_32	LK_F32_X0, LK_F32_M, LK_F32_A
.endm
.macro lk_op_v_add_f32 d
	v_add_f32	v\d, v2, v6
.endm

.macro lk_init_v_mul_lo_u32
	lk_init_32	LK_U32_X0, LK_U32_M, LK_U32_A
.endm
.macro lk_op_v_mul_lo_u32 d
	v_mul_lo_u32	v\d, v2, v4
.endm

.macro lk_init_v_add_u32
	lk_init_32	LK_U32_X0, LK_U32_M, LK_U32_A
.endm
.macro lk_op_v_add_u32 d
	v_add_u32	v\d, v2, v6
.endm

.macro lk_init_v_fma_f64
	lk_init_64	LK_F64_X0_LO, LK_F64_X0_HI, LK_F64_M_LO, LK_F64_M_HI, LK_F64_A_LO, LK_F64_A_HI
.endm
.macro lk_op_v_fma_f64 d
	v_fma_f64	v[\d:\d+1], v[2:3], v[4:5], v[6:7]
.endm

/* Two f32 lanes per register pair: the low registers hold one FMA's operands,
   the high registers the other's. M and A are the same in both lanes. */
.macro lk_init_v_pk_fma_f32
	lk_init_64	LK_F32_X0, LK_F32_X0_HI, LK_F32_M, LK_F32_M, LK_F32_A, LK_F32_A
.endm
.macro lk_op_v_pk_fma_f32 d
	v_pk_fma_f32	v[\d:\d+1], v[2:3], v[4:5], v[6:7]
.endm

/* A move has one source. With d = 2 it copies X onto itself: the value never
   changes, but each move still reads what the previous one wrote. */
.macro lk_init_v_mov_b64
	lk_init_64	LK_B64_LO, LK_B64_HI, 0, 0, 0, 0
.endm
.macro lk_op_v_mov_b64 d
	v_mov_b64	v[\d:\d+1], v[2:3]
.endm

/* ---- Kernel entry and exit ---------------------------------------------- */

/* Kernel entry state (the descriptor in lk_kernel_end requests it): s[0:1] =
   kernarg segment, s2 = workgroup id x, v0 = thread id within the workgroup.
   The kernarg segment is { out, iterations, hidden arguments }, laid out as
   hipcc lays it out for `(uint64_t *out, int iterations)`. The load of s3 is
   requested here; an s_waitcnt must precede the loop. */
.macro lk_kernel_begin name
	.text
	.protected	\name
	.globl	\name
	.p2align	8
	.type	\name,@function
\name:
	s_load_dword	s3, s[0:1], 0x8			// s3 = iterations
.endm

/* After the loop: store v[result:result+1] to out[global thread index], end
   the kernel and emit its descriptor. `sgprs` is the number of scalar
   registers the kernel may use; the code here uses s0..s5. */
.macro lk_kernel_end name, result, sgprs
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
	global_store_dwordx2	v0, v[\result:\result+1], s[0:1]
	s_endpgm
.Lend_\name:
	.size	\name, .Lend_\name-\name

	/* Kernel descriptor: the values hipcc emits for an equivalent kernel,
	   with room for v0..v39. */
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
		.amdhsa_next_free_sgpr \sgprs
		.amdhsa_accum_offset 40
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
	.end_amdhsa_kernel
.endm

/* After the last kernel: the instruction prefetcher reads past it, so pad
   with s_nop, as the compiler does. */
.macro lk_text_end
	.text
	.p2alignl	6, 0xbf800000
	.fill	256, 4, 0xbf800000
.endm

#endif /* __ASSEMBLER__ */

#endif /* GFX942_LOOP_KERNELS_H */
