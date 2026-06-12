package ptrace

// predict.go computes the dynamic-count fingerprint of a trace — the
// counts-only validation gate for generated traces. The expensive part of
// tracing is recording addresses, not executing the kernel; a counts-only
// profile of the *target* size (CUPTI counters or an NVBit counting pass) is
// orders of magnitude cheaper than address tracing. Comparing those measured
// counters against the counts a generated trace predicts catches both regime
// hazards: a different binary and a different code path (remainder loop,
// different unroll factor) shift the instruction counts immediately.
//
// The measured side is produced externally (e.g. ncu: smsp__inst_executed,
// smsp__thread_inst_executed, l1tex sector counters) and supplied as the same
// JSON shape; zero/omitted fields are treated as "not measured" and skipped.

import (
	"fmt"
	"math/bits"
)

// KernelCounts is the dynamic-count fingerprint of one kernel invocation.
type KernelCounts struct {
	TraceFile    string `json:"trace_file,omitempty"`
	Blocks       int64  `json:"blocks,omitempty"`
	Warps        int64  `json:"warps,omitempty"`          // non-empty warps
	WarpInsts    int64  `json:"warp_insts,omitempty"`     // warp-level dynamic instructions
	ThreadInsts  int64  `json:"thread_insts,omitempty"`   // sum of active lanes per instruction
	MemWarpInsts int64  `json:"mem_warp_insts,omitempty"` // warp-level memory instructions
	Sectors      int64  `json:"sectors,omitempty"`        // 32B memory transactions
}

// CountsReport is the per-kernel fingerprint list plus the run total.
type CountsReport struct {
	Kernels []KernelCounts `json:"kernels,omitempty"`
	Total   KernelCounts   `json:"total"`
}

// PredictCounts computes the fingerprint of a decoded invocation. It works on
// real and generated traces alike — generation gates compare a generated
// trace's prediction to a counts-only profile of a real run.
func PredictCounts(blocks []Block, isa map[int]StaticInst) KernelCounts {
	var c KernelCounts
	c.Blocks = int64(len(blocks))
	for _, blk := range blocks {
		for _, warp := range blk.Warps {
			if len(warp) == 0 {
				continue
			}
			c.Warps++
			for _, d := range warp {
				c.WarpInsts++
				c.ThreadInsts += int64(bits.OnesCount32(d.Mask))
				if !isa[d.Pc].IsMem {
					continue
				}
				c.MemWarpInsts++
				if addrs, err := payloadToAddrs(d.Mode, d.Payload, d.Mask); err == nil {
					c.Sectors += distinctSectors(addrs, statSectorBits)
				}
			}
		}
	}
	return c
}

func distinctSectors(addrs []uint64, sectorBits uint) int64 {
	s := map[uint64]bool{}
	for _, a := range addrs {
		s[a>>sectorBits] = true
	}
	return int64(len(s))
}

// AddCounts accumulates b into a (TraceFile is left as a's).
func AddCounts(a, b KernelCounts) KernelCounts {
	a.Blocks += b.Blocks
	a.Warps += b.Warps
	a.WarpInsts += b.WarpInsts
	a.ThreadInsts += b.ThreadInsts
	a.MemWarpInsts += b.MemWarpInsts
	a.Sectors += b.Sectors
	return a
}

// CompareCounts checks a predicted fingerprint against a measured one within a
// relative tolerance, returning one message per violated field. A zero
// measured field means "not measured" and is skipped, so a measurement can
// supply any subset of counters.
func CompareCounts(pred, meas KernelCounts, tol float64) []string {
	var fails []string
	chk := func(name string, p, m int64) {
		if m == 0 {
			return
		}
		re := absF(float64(p-m)) / float64(m)
		if re > tol {
			fails = append(fails, fmt.Sprintf("%s: predicted %d, measured %d (%.2f%% off, tol %.2f%%)",
				name, p, m, 100*re, 100*tol))
		}
	}
	chk("blocks", pred.Blocks, meas.Blocks)
	chk("warps", pred.Warps, meas.Warps)
	chk("warp_insts", pred.WarpInsts, meas.WarpInsts)
	chk("thread_insts", pred.ThreadInsts, meas.ThreadInsts)
	chk("mem_warp_insts", pred.MemWarpInsts, meas.MemWarpInsts)
	chk("sectors", pred.Sectors, meas.Sectors)
	return fails
}
