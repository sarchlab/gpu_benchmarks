package ptrace

import "testing"

// loopWarp builds a warp with a prologue, a loop body run `iters` times, and an
// epilogue. The body's last instruction is the back-edge branch: it carries the
// taken mask (0xffffffff) on every iteration except the last, where it exits
// (mask 0) — exactly how a real trace records loop divergence.
func loopWarp(iters int) []DynInst {
	var w []DynInst
	for _, pc := range []int{0x00, 0x10, 0x20} { // prologue
		w = append(w, DynInst{Pc: pc, Mask: 0xffffffff})
	}
	for k := 0; k < iters; k++ {
		w = append(w,
			DynInst{Pc: 0x30, Mask: 0xffffffff},
			DynInst{Pc: 0x40, Mask: 0xffffffff})
		brMask := uint32(0xffffffff)
		if k == iters-1 {
			brMask = 0 // exit
		}
		w = append(w, DynInst{Pc: 0x50, Mask: brMask}) // back-edge branch
	}
	for _, pc := range []int{0x60, 0x70} { // epilogue
		w = append(w, DynInst{Pc: pc, Mask: 0xffffffff})
	}
	return w
}

func warpsEqual(a, b []DynInst) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Pc != b[i].Pc || a[i].Mask != b[i].Mask || a[i].Mode != b[i].Mode {
			return false
		}
	}
	return true
}

func TestScaleLoopReplay(t *testing.T) {
	isa := map[int]StaticInst{
		0x00: {}, 0x10: {}, 0x20: {}, 0x30: {}, 0x40: {}, 0x50: {}, 0x60: {}, 0x70: {},
	}
	// sample at 4 iterations; the real trace at 16 iterations is the ground truth.
	sample := Block{Warps: [][]DynInst{loopWarp(4)}}
	want := loopWarp(16)

	scaled, ok := ScaleBlocks([]Block{sample}, isa, 16, 4) // num/den = 16/4 = 4x
	if !ok {
		t.Fatal("sample with a 4-iteration loop should be scalable")
	}
	got := scaled[0].Warps[0]
	if !warpsEqual(got, want) {
		t.Fatalf("scaled warp != real 16-iteration warp\n got len=%d want len=%d", len(got), len(want))
	}
	// spot-check the back-edge masks: 15 taken + 1 exit
	taken, exit := 0, 0
	for _, d := range got {
		if d.Pc == 0x50 {
			if d.Mask == 0xffffffff {
				taken++
			} else {
				exit++
			}
		}
	}
	if taken != 15 || exit != 1 {
		t.Errorf("back-edge masks: got %d taken / %d exit, want 15/1", taken, exit)
	}
}

func TestScaleRejectsNoLoop(t *testing.T) {
	isa := map[int]StaticInst{0x00: {}, 0x10: {}, 0x20: {}}
	// a single straight-line iteration (no repetition) is not scalable.
	blk := Block{Warps: [][]DynInst{{
		{Pc: 0x00, Mask: 0xffffffff},
		{Pc: 0x10, Mask: 0xffffffff},
		{Pc: 0x20, Mask: 0xffffffff},
	}}}
	if _, ok := ScaleBlocks([]Block{blk}, isa, 8, 1); ok {
		t.Error("a warp with no detectable loop must not be reported scalable")
	}
}
