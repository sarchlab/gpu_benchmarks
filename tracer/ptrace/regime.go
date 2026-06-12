package ptrace

// regime.go is the regime-consistency gate for trace generation. Extrapolating
// a trace from a base run (ScaleBlocks, ExtrapolateGrid) is only sound when the
// target size executes the same compiled program down the same control-flow
// regime — nvcc emits a remainder loop at tiny sizes and a fully-unrolled body
// at larger ones, and a compile-time-specialized size produces a different
// binary outright. A single base run cannot detect either.
//
// CheckRegime therefore compares two *small* runs of the same kernel at
// different scaling values (e.g. N and 2N — both cheap to collect) and gates on:
//
//  1. identical static program (ISA-table hash) — different SASS at a different
//     size (template/-DN= specialization, unroll decisions) is a hard failure;
//  2. identical control-flow shape — the per-warp loop prologue/body/epilogue
//     (pc, mode) signature with the size-dependent repeat count dropped. This
//     catches a remainder loop appearing or disappearing within one binary;
//  3. cross-prediction — scaling run A to run B's size must reproduce run B
//     structurally; only run-specific absolute addresses (cudaMalloc pointers)
//     may differ.
//
// Passing means extrapolation beyond B follows a line verified over one real
// step, instead of a line fit at a single point.

import (
	"strconv"
	"strings"
)

// RegimeResult is the verdict of CheckRegime for one kernel.
type RegimeResult struct {
	ISAMatch           bool   // static programs identical
	ISAHashA, ISAHashB string //
	PathMatch          bool   // control-flow shape identical (size-invariant part)
	Scalable           bool   // run A's loops detected and integer-scalable to B's size
	CrossOK            bool   // scaled A == B modulo run-specific addresses
	BlocksTotal        int    // blocks compared in cross-prediction
	BlocksSame         int    // byte-identical blocks
	AddrDiffs          int    // differing tokens that are address-shaped (expected across runs)
	StructDiffs        int    // differing tokens/blocks that are structural (a regime change)
}

// Pass reports whether extrapolation from these runs is justified. A kernel
// with no detectable loop cannot be cross-predicted on the temporal axis, but
// temporal extrapolation (ScaleBlocks) would refuse it anyway; the static and
// path gates still apply.
func (r RegimeResult) Pass() bool {
	if !r.ISAMatch || !r.PathMatch {
		return false
	}
	return !r.Scalable || r.CrossOK
}

// CheckRegime runs the three gates on two decoded runs of the same kernel at
// scaling values fromA and fromB. Pass fromA == fromB to use it as a pure
// run-to-run reproducibility check.
func CheckRegime(blocksA []Block, isaA map[int]StaticInst, fromA int,
	blocksB []Block, isaB map[int]StaticInst, fromB int) RegimeResult {
	var r RegimeResult
	ta, _ := buildISA(isaA)
	tb, _ := buildISA(isaB)
	r.ISAHashA, r.ISAHashB = ta.hash(), tb.hash()
	r.ISAMatch = r.ISAHashA == r.ISAHashB
	r.PathMatch = PathSignature(blocksA) == PathSignature(blocksB)
	if fromA <= 0 || fromB <= 0 {
		return r
	}
	scaled, ok := ScaleBlocks(blocksA, isaA, fromB, fromA)
	r.Scalable = ok
	if !ok {
		return r
	}
	r.BlocksTotal, r.BlocksSame, r.AddrDiffs, r.StructDiffs = compareBlocksStructural(scaled, blocksB)
	r.CrossOK = r.StructDiffs == 0
	return r
}

// PathSignature is the size-invariant control-flow fingerprint of a run: the
// set of distinct per-warp regime signatures. Two runs of the same kernel in
// the same regime produce the same signature regardless of loop trip count.
func PathSignature(blocks []Block) string {
	set := map[string]bool{}
	for _, blk := range blocks {
		for _, w := range blk.Warps {
			if len(w) == 0 {
				continue
			}
			set[warpRegimeSig(w)] = true
		}
	}
	sigs := make([]string, 0, len(set))
	for s := range set {
		sigs = append(sigs, s)
	}
	sortStrings(sigs)
	return strings.Join(sigs, "\n")
}

// warpRegimeSig fingerprints one warp's control path: the (pc, mode) sequences
// of the loop prologue, body, and epilogue when a loop is detected — dropping
// the repeat count, the only part that legitimately changes with problem size —
// or the full sequence for a straight-line warp. Masks and addresses are
// excluded for the same reason they are excluded from sameLoopInst.
func warpRegimeSig(warp []DynInst) string {
	pre, body, reps, ok := detectLoop(warp)
	if !ok {
		return "flat|" + segSig(warp)
	}
	return "loop|" + segSig(warp[:pre]) + "|" + segSig(warp[pre:pre+body]) + "|" + segSig(warp[pre+reps*body:])
}

func segSig(s []DynInst) string {
	var b strings.Builder
	for i, d := range s {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.Itoa(d.Pc))
		b.WriteByte(':')
		b.WriteString(strconv.Itoa(d.Mode))
	}
	return b.String()
}

// DiffBlocks is the exported entry to the structural block comparison: it
// pairs blocks by position and classifies every token difference as
// address-shaped (run-specific, expected across processes) or structural.
func DiffBlocks(gen, want []Block) (total, same, addrDiffs, structDiffs int) {
	return compareBlocksStructural(gen, want)
}

// compareBlocksStructural compares a generated run against a real run of the
// same shape, classifying every difference: address-shaped token differences
// are run-specific (different cudaMalloc pointers — expected when the runs are
// separate processes); anything else (shape, pc, mask, mode, non-address
// tokens, block-count mismatch) is structural and indicates a regime change.
func compareBlocksStructural(gen, want []Block) (total, same, addrDiffs, structDiffs int) {
	total = len(want)
	if len(gen) > total {
		total = len(gen)
	}
	n := len(want)
	if len(gen) < n {
		n = len(gen)
	}
	structDiffs += total - n // unmatched blocks are structural
	for i := 0; i < n; i++ {
		g, w := gen[i], want[i]
		identical := true
		if g.X != w.X || g.Y != w.Y || g.Z != w.Z {
			identical = false
			structDiffs++
		}
		if len(g.Warps) != len(w.Warps) {
			structDiffs++
			continue
		}
		for wi := range w.Warps {
			gw, ww := g.Warps[wi], w.Warps[wi]
			if len(gw) != len(ww) {
				identical = false
				structDiffs++
				continue
			}
			for k := range ww {
				gd, wd := gw[k], ww[k]
				if gd.Pc != wd.Pc || gd.Mask != wd.Mask || gd.Mode != wd.Mode ||
					len(gd.Payload) != len(wd.Payload) {
					identical = false
					structDiffs++
					continue
				}
				for t := range wd.Payload {
					if gd.Payload[t] == wd.Payload[t] {
						continue
					}
					identical = false
					if isAddrShaped(gd.Payload[t]) && isAddrShaped(wd.Payload[t]) {
						addrDiffs++
					} else {
						structDiffs++
					}
				}
			}
		}
		if identical {
			same++
		}
	}
	return
}

// isAddrShaped reports whether a token looks like an absolute memory address
// (wide hex) — i.e. a run-specific pointer rather than structural content.
func isAddrShaped(t string) bool {
	t = strings.TrimPrefix(t, "0x")
	if len(t) < 6 || len(t) > 16 {
		return false
	}
	for _, c := range t {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			return false
		}
	}
	return true
}

func sortStrings(a []string) {
	for i := 1; i < len(a); i++ {
		for j := i; j > 0 && a[j-1] > a[j]; j-- {
			a[j-1], a[j] = a[j], a[j-1]
		}
	}
}
