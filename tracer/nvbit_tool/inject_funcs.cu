/* Author1: Mahmoud Khairy, abdallm@purdue.com - 2019 */
/* Author2: Jason Shen, shen203@purdue.edu - 2019 */

#include <stdint.h>
#include <stdio.h>

#include <cstdarg>

#include "utils/utils.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the inst_trace_t structure */
#include "common.h"

/* Instrumentation function that we want to inject, please note the use of
 *  extern "C" __device__ __noinline__
 *    To prevent "dead"-code elimination by the compiler.
 *
 * Arg alignment: NVBit 1.7.6 marshals call args into consecutive 32-bit
 * register units, and a 64-bit arg placed at an ODD unit offset arrives
 * corrupted on the device (dereferencing such a pointer faults with
 * IllegalAddress; observed empirically on H100/CUDA 12.9). Every 64-bit
 * parameter in this signature must therefore sit at an even 32-bit offset
 * counted from the first parameter — check the running offset whenever
 * adding/reordering parameters. The upstream signature satisfies this by
 * accident; psample_list is placed BEFORE the 32-bit sample_mode for the
 * same reason. The three upstream counter/flag pointers are consolidated
 * behind pctrl (tool_ctrl_t in common.h).
 */
extern "C" __device__ __noinline__ void
instrument_inst(int pred, int opcode_id, int32_t vpc, bool is_mem,
                uint64_t addr, int32_t width, int32_t desReg, int32_t srcReg1,
                int32_t srcReg2, int32_t srcReg3, int32_t srcReg4,
                int32_t srcReg5, int32_t srcNum, uint64_t immediate,
                uint64_t pchannel_dev, uint64_t pctrl, uint64_t psample_list,
                uint32_t sample_mode, uint32_t line_num,
                uint32_t instr_idx) {
  tool_ctrl_t *ctrl = (tool_ctrl_t *)pctrl;

  /* gpubench P1b live sampling: blocks outside the sample plan return
   * before any ballot/shuffle/channel work. The predicate is uniform
   * across the block, so the whole warp exits together. sample_mode is a
   * compile-time constant of the injected call (0 = trace everything). */
  if (sample_mode != 0) {
    const int4 scta = get_ctaid();
    int corner_k = PTRACE_SAMPLE_CORNER_K(sample_mode);
    if (corner_k == 0)
      corner_k = 2;
    bool sampled = (sample_mode & PTRACE_SAMPLE_CORNER) && scta.x < corner_k &&
                   scta.y < corner_k && scta.z < corner_k;
    if (!sampled && (sample_mode & PTRACE_SAMPLE_LIST)) {
      const ptrace_sample_list_t *lst =
          (const ptrace_sample_list_t *)psample_list;
      for (int i = 0; i < lst->n; i++) {
        if (lst->x[i] == scta.x && lst->y[i] == scta.y &&
            lst->z[i] == scta.z) {
          sampled = true;
          break;
        }
      }
    }
    if (!sampled) {
      if (sample_mode & PTRACE_SAMPLE_COUNT_SKIPPED) {
        const int skip_mask = __ballot_sync(__activemask(), 1);
        if (get_laneid() == __ffs(skip_mask) - 1) {
          atomicAdd(&ctrl->total, 1);
        }
      }
      return;
    }
  }

  const int active_mask = __ballot_sync(__activemask(), 1);
  const int predicate_mask = __ballot_sync(__activemask(), pred);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  if (ctrl->stop) {
    if (first_laneid == laneid) {
      atomicAdd(&ctrl->total, 1);
      return;
    }
  }

  inst_trace_t ma;

  if (is_mem) {
    /* collect memory address information */
    for (int i = 0; i < 32; i++) {
      ma.addrs[i] = __shfl_sync(active_mask, addr, i);
    }
    ma.width = width;
    ma.is_mem = true;
  } else {
    ma.is_mem = false;
  }

  int4 cta = get_ctaid();
  int uniqe_threadId = threadIdx.z * blockDim.y * blockDim.x +
                       threadIdx.y * blockDim.x + threadIdx.x;
  ma.line_num = line_num;
  ma.instr_idx = instr_idx;
  ma.warpid_tb = uniqe_threadId / 32;

  ma.cta_id_x = cta.x;
  ma.cta_id_y = cta.y;
  ma.cta_id_z = cta.z;
  ma.warpid_sm = get_warpid();
  ma.opcode_id = opcode_id;
  ma.vpc = vpc;
  ma.GPRDst = desReg;
  ma.GPRSrcs[0] = srcReg1;
  ma.GPRSrcs[1] = srcReg2;
  ma.GPRSrcs[2] = srcReg3;
  ma.GPRSrcs[3] = srcReg4;
  ma.GPRSrcs[4] = srcReg5;
  ma.numSrcs = srcNum;
  ma.imm = immediate;
  ma.active_mask = active_mask;
  ma.predicate_mask = predicate_mask;
  ma.sm_id = get_smid();

  /* first active lane pushes information on the channel */
  if (first_laneid == laneid) {
    ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
    channel_dev->push(&ma, sizeof(inst_trace_t));
    atomicAdd(&ctrl->total, 1);
    atomicAdd(&ctrl->reported, 1);
  }
}
