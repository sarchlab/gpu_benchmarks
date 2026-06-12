package ptrace

import (
	"fmt"
	"testing"
)

func loopIsa() map[int]StaticInst {
	return map[int]StaticInst{
		0x00: {}, 0x10: {}, 0x20: {}, 0x30: {}, 0x40: {}, 0x50: {}, 0x60: {}, 0x70: {},
	}
}

// loopMemWarp is loopWarp with a memory instruction in the body whose base
// advances 4 bytes per iteration from `base` — affine in occ, so the recipe
// fitter models it and cross-prediction can regenerate it at any trip count.
func loopMemWarp(iters int, base uint64) []DynInst {
	w := []DynInst{{Pc: 0x00, Mask: 0xffffffff}}
	for k := 0; k < iters; k++ {
		w = append(w, DynInst{Pc: 0x10, Mask: 0xffffffff, Mode: 1,
			Payload: []string{fmt.Sprintf("0x%x", base+uint64(k)*4), "4"}})
		brMask := uint32(0xffffffff)
		if k == iters-1 {
			brMask = 0
		}
		w = append(w, DynInst{Pc: 0x20, Mask: brMask})
	}
	return append(w, DynInst{Pc: 0x30, Mask: 0xffffffff})
}

func loopMemIsa() map[int]StaticInst {
	return map[int]StaticInst{
		0x00: {},
		0x10: {Middle: "1 R2 LDG.E 1 R4", IsMem: true, Width: "4", Imm: "0"},
		0x20: {},
		0x30: {},
	}
}

// Two runs at 4 and 8 iterations of the same loop: same regime, and scaling
// the small run to the large one's size must reproduce it exactly.
func TestRegimeSameLoop(t *testing.T) {
	isa := loopIsa()
	a := []Block{{Warps: [][]DynInst{loopWarp(4)}}}
	b := []Block{{Warps: [][]DynInst{loopWarp(8)}}}
	r := CheckRegime(a, isa, 4, b, isa, 8)
	if !r.ISAMatch || !r.PathMatch {
		t.Fatalf("same kernel at two sizes must match statically: %+v", r)
	}
	if !r.Scalable || !r.CrossOK || r.BlocksSame != 1 {
		t.Fatalf("cross-prediction 4->8 must reproduce run B: %+v", r)
	}
	if !r.Pass() {
		t.Fatalf("expected Pass: %+v", r)
	}
}

// Same instruction set, but run B executes it straight-line (no loop) — the
// fully-unrolled-vs-loop regime change. The ISA gate cannot see it; the path
// signature must.
func TestRegimePathChange(t *testing.T) {
	isa := loopIsa()
	a := []Block{{Warps: [][]DynInst{loopWarp(4)}}}
	var flat []DynInst
	for _, pc := range []int{0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70} {
		flat = append(flat, DynInst{Pc: pc, Mask: 0xffffffff})
	}
	b := []Block{{Warps: [][]DynInst{flat}}}
	r := CheckRegime(a, isa, 4, b, isa, 8)
	if !r.ISAMatch {
		t.Fatalf("ISA must match (same static program): %+v", r)
	}
	if r.PathMatch || r.Pass() {
		t.Fatalf("loop vs straight-line must fail the path gate: %+v", r)
	}
}

// Different static program at the second size (compile-time specialization):
// the ISA hash floor must fail it regardless of anything else.
func TestRegimeISAChange(t *testing.T) {
	isaA := loopIsa()
	isaB := loopIsa()
	isaB[0x40] = StaticInst{Middle: "1 R5 FFMA"} // different SASS at this PC
	a := []Block{{Warps: [][]DynInst{loopWarp(4)}}}
	b := []Block{{Warps: [][]DynInst{loopWarp(8)}}}
	r := CheckRegime(a, isaA, 4, b, isaB, 8)
	if r.ISAMatch || r.Pass() {
		t.Fatalf("different static programs must fail the ISA gate: %+v", r)
	}
}

// Cross-prediction across two *separate runs*: run B's allocation base differs
// (different cudaMalloc), which must be tolerated as run-specific, while the
// structure must still verify.
func TestRegimeCrossRunAddressTolerance(t *testing.T) {
	isa := loopMemIsa()
	a := []Block{{Warps: [][]DynInst{loopMemWarp(4, 0x7f0000001000)}}}
	b := []Block{{Warps: [][]DynInst{loopMemWarp(8, 0x7f0000002000)}}}
	r := CheckRegime(a, isa, 4, b, isa, 8)
	if !r.Scalable || !r.CrossOK {
		t.Fatalf("address-only diffs must not fail cross-prediction: %+v", r)
	}
	if r.AddrDiffs == 0 || r.StructDiffs != 0 {
		t.Fatalf("expected only run-specific address diffs, got addr=%d struct=%d", r.AddrDiffs, r.StructDiffs)
	}
	if !r.Pass() {
		t.Fatalf("expected Pass: %+v", r)
	}
}

// A structural change the static gates cannot see: same PCs, same modes, but a
// different stride at the second size. Cross-prediction must catch it.
func TestRegimeCrossPredictionCatchesStride(t *testing.T) {
	isa := loopMemIsa()
	a := []Block{{Warps: [][]DynInst{loopMemWarp(4, 0x7f0000001000)}}}
	bw := loopMemWarp(8, 0x7f0000001000)
	for i := range bw {
		if bw[i].Pc == 0x10 {
			bw[i].Payload[1] = "8" // stride changed: different access pattern
		}
	}
	b := []Block{{Warps: [][]DynInst{bw}}}
	r := CheckRegime(a, isa, 4, b, isa, 8)
	if !r.ISAMatch || !r.PathMatch {
		t.Fatalf("static gates cannot see a stride change: %+v", r)
	}
	if r.CrossOK || r.Pass() {
		t.Fatalf("stride change must fail cross-prediction: %+v", r)
	}
}

func TestPathSignatureSizeInvariant(t *testing.T) {
	s4 := PathSignature([]Block{{Warps: [][]DynInst{loopWarp(4)}}})
	s16 := PathSignature([]Block{{Warps: [][]DynInst{loopWarp(16)}}})
	if s4 != s16 {
		t.Fatalf("loop trip count must not change the path signature:\n%q\nvs\n%q", s4, s16)
	}
}
