package ptrace

// synth.go implements P3: synthesizing the trace of a thread block that was
// never recorded, from (a) a template block giving the per-warp instruction
// shape (pc, mask, mode, tail) and (b) address recipes giving each memory
// instruction's base as an affine function of the block coordinates.
//
// This is the basis for the wall-time win: instrument only a few blocks of a
// large launch, fit recipes, and generate the rest. `Holdout` measures it
// directly — fit on a subset of blocks, synthesize the held-out blocks, and
// compare to ground truth.

import (
	"fmt"
	"os"
	"path/filepath"
)

// SynthBlock produces the block at (x,y,z) by replaying the template's shape and
// recomputing each modeled memory instruction's base at the new coordinates.
// Returns ok=false if the template contains a memory instruction that is not
// modeled (its addresses are data-dependent and cannot be synthesized).
func SynthBlock(tmpl Block, x, y, z int, recipes map[int]Recipe, isa map[int]StaticInst) (Block, bool) {
	out := Block{X: x, Y: y, Z: z, Warps: make([][]DynInst, len(tmpl.Warps))}
	for w := range tmpl.Warps {
		occ := map[int]int{}
		warp := make([]DynInst, len(tmpl.Warps[w]))
		for i, td := range tmpl.Warps[w] {
			o := occ[td.Pc]
			occ[td.Pc] = o + 1
			d := DynInst{Pc: td.Pc, Mask: td.Mask}
			if isa[td.Pc].IsMem {
				rec, ok := recipes[td.Pc]
				if !ok {
					return Block{}, false
				}
				ci := rec.classIndexFor(td.Mode, tailOf(td.Payload))
				if ci < 0 {
					return Block{}, false
				}
				cls := rec.Classes[ci]
				d.Mode = cls.Mode
				d.Payload = cls.payloadFor(x, y, z, w, o)
			}
			warp[i] = d
		}
		out.Warps[w] = warp
	}
	return out, true
}

func tailOf(payload []string) []string {
	if len(payload) <= 1 {
		return nil
	}
	return payload[1:]
}

// ExtrapolateGrid generates the trace of a launch over a *new* grid (gx,gy,gz)
// from the recipes fit on the given (smaller) blocks — the "generate a large
// trace from a small execution" goal. It fits recipes on the input blocks, uses
// the first block as the per-warp shape template, and synthesizes every block of
// the target grid. Blocks whose template contains an unmodeled memory access
// cannot be extrapolated and are counted in nFail (and omitted). This output is
// unverified by construction: there is no ground truth at the new size, so a
// committed ground-truth trace should gate any production use.
func ExtrapolateGrid(blocks []Block, isa map[int]StaticInst, header string, gx, gy, gz int) (content string, nSynth, nFail int, err error) {
	if len(blocks) == 0 {
		return "", 0, 0, fmt.Errorf("no template blocks")
	}
	recipes := fitRecipes(blocks, isa)
	tmpl := blocks[0]
	out := make([]Block, 0, gx*gy*gz)
	for z := 0; z < gz; z++ {
		for y := 0; y < gy; y++ {
			for x := 0; x < gx; x++ {
				b, ok := SynthBlock(tmpl, x, y, z, recipes, isa)
				if !ok {
					nFail++
					continue
				}
				out = append(out, b)
				nSynth++
			}
		}
	}
	content, err = EmitTraceg(header, out, isa)
	return content, nSynth, nFail, err
}

// HoldoutResult summarizes a synthesize-the-held-out-blocks experiment.
type HoldoutResult struct {
	TrainBlocks  int
	TestBlocks   int
	TestInsts    int // memory instructions in held-out blocks
	MatchInsts   int // synthesized byte-identically
	BlocksOK     int // held-out blocks reproduced fully
	NotSynthable bool
}

// Holdout fits recipes on the blocks selected by train(), then synthesizes every
// other block and compares it to ground truth.
func Holdout(blocks []Block, isa map[int]StaticInst, train func(b Block) bool) HoldoutResult {
	var trainBlocks, testBlocks []Block
	for _, b := range blocks {
		if train(b) {
			trainBlocks = append(trainBlocks, b)
		} else {
			testBlocks = append(testBlocks, b)
		}
	}
	res := HoldoutResult{TrainBlocks: len(trainBlocks), TestBlocks: len(testBlocks)}
	if len(trainBlocks) == 0 || len(testBlocks) == 0 {
		return res
	}
	recipes := fitRecipes(trainBlocks, isa)
	tmpl := trainBlocks[0]

	for _, real := range testBlocks {
		syn, ok := SynthBlock(tmpl, real.X, real.Y, real.Z, recipes, isa)
		if !ok {
			res.NotSynthable = true
			// still count its mem insts as not-matched
			res.TestInsts += countMem(real, isa)
			continue
		}
		nInst, nMatch, full := compareMem(syn, real, isa)
		res.TestInsts += nInst
		res.MatchInsts += nMatch
		if full {
			res.BlocksOK++
		}
	}
	return res
}

func countMem(b Block, isa map[int]StaticInst) int {
	n := 0
	for _, warp := range b.Warps {
		for _, d := range warp {
			if isa[d.Pc].IsMem {
				n++
			}
		}
	}
	return n
}

// compareMem compares the memory instructions of two blocks (assumed same shape).
// Returns (#mem insts, #byte-identical, allMatched).
func compareMem(syn, real Block, isa map[int]StaticInst) (nInst, nMatch int, full bool) {
	full = true
	if len(syn.Warps) != len(real.Warps) {
		return countMem(real, isa), 0, false
	}
	for w := range real.Warps {
		if len(syn.Warps[w]) != len(real.Warps[w]) {
			full = false
			nInst += countMemWarp(real.Warps[w], isa)
			continue
		}
		for i := range real.Warps[w] {
			rd := real.Warps[w][i]
			if !isa[rd.Pc].IsMem {
				continue
			}
			nInst++
			sd := syn.Warps[w][i]
			if sd.Pc == rd.Pc && sd.Mode == rd.Mode && eqStr(sd.Payload, rd.Payload) {
				nMatch++
			} else {
				full = false
			}
		}
	}
	return nInst, nMatch, full
}

func countMemWarp(warp []DynInst, isa map[int]StaticInst) int {
	n := 0
	for _, d := range warp {
		if isa[d.Pc].IsMem {
			n++
		}
	}
	return n
}

// ── decode a .ptrace invocation back to blocks (for holdout from a .ptrace) ────

// DecodeInvocationBlocks returns the decoded blocks + ISA map for an invocation.
func DecodeInvocationBlocks(ptraceDir string, inv Invocation, isaCache map[string]ISATable) ([]Block, map[int]StaticInst, error) {
	table, err := loadISA(ptraceDir, isaCache, inv.ISA)
	if err != nil {
		return nil, nil, err
	}
	comp, err := os.ReadFile(filepath.Join(ptraceDir, "dyn", inv.DynFile))
	if err != nil {
		return nil, nil, err
	}
	raw, err := zstdDecompress(comp)
	if err != nil {
		return nil, nil, err
	}
	blocks, err := decodeDyn(raw, table)
	if err != nil {
		return nil, nil, err
	}
	return blocks, table.toMap(), nil
}
