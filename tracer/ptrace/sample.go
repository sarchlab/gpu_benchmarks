package ptrace

// sample.go implements P3: sampled-block collection + verified synthesis.
//
// The wall-time thesis is that a regular kernel's trace can be reconstructed from
// only a handful of "instrumented" thread blocks. This file turns that into a
// concrete storage mode: a sampled .ptrace stores
//
//   - a corner sample of blocks, verbatim (these are the only ones a real tracer
//     would have to instrument), plus
//   - the affine address recipes fit from that sample, plus
//   - a plan: for every block of the launch, whether it was recorded or is to be
//     synthesized from the template + recipes.
//
// Construction is verify-or-fallback: every candidate synthesized block is
// regenerated and compared byte-for-byte against the real block; any block that
// does not reproduce exactly is demoted to "recorded". A sampled .ptrace is
// therefore still losslessly equal to the original — synthesis can only shrink
// it, never corrupt it. The instrumented fraction (recorded/total blocks) is the
// directly-measured proxy for collection wall-time.

import (
	"encoding/binary"
	"fmt"
)

const dynVersionSampled = 4

// samplePlan records, for one invocation, which blocks are stored verbatim and
// which are synthesized, plus which recorded block is the synthesis template.
type samplePlan struct {
	isSynth     []bool // per original block index
	recorded    []int  // original indices stored verbatim, ascending
	templatePos int    // index into recorded[] of the template block
}

// cornerSample returns the indices of the blocks to instrument: every block whose
// each block coordinate is one of the two smallest distinct values on that axis.
// Two values per varying axis is the minimum the exact-affine solver needs to
// recover a slope, so the recipes extrapolate to the whole grid (the P0 "4/16"
// result). The set is at most 2x2x2 = 8 blocks regardless of launch size, which
// is what decouples collection cost from problem size.
func cornerSample(blocks []Block) []int {
	xs := axisSmallest(blocks, func(b Block) int { return b.X })
	ys := axisSmallest(blocks, func(b Block) int { return b.Y })
	zs := axisSmallest(blocks, func(b Block) int { return b.Z })
	var idx []int
	for i, b := range blocks {
		if xs[b.X] && ys[b.Y] && zs[b.Z] {
			idx = append(idx, i)
		}
	}
	if len(idx) == 0 && len(blocks) > 0 {
		idx = []int{0}
	}
	return idx
}

// axisSmallest returns the set of the (up to) two smallest distinct values taken
// by sel over blocks.
func axisSmallest(blocks []Block, sel func(Block) int) map[int]bool {
	seen := map[int]bool{}
	for _, b := range blocks {
		seen[sel(b)] = true
	}
	vals := make([]int, 0, len(seen))
	for v := range seen {
		vals = append(vals, v)
	}
	sortInts(vals)
	out := map[int]bool{}
	for i := 0; i < len(vals) && i < 2; i++ {
		out[vals[i]] = true
	}
	return out
}

// buildSamplePlan picks the corner sample, fits recipes on it, then tries to
// synthesize every other block; blocks that do not reproduce byte-exactly are
// added to the recorded set (fallback). Returns the plan and the sample-fit
// recipes (keyed by PC) that decode will use for the synthesized blocks.
func buildSamplePlan(blocks []Block, isa map[int]StaticInst) (samplePlan, map[int]Recipe) {
	sampleIdx := cornerSample(blocks)
	inSample := map[int]bool{}
	sampleBlocks := make([]Block, 0, len(sampleIdx))
	for _, i := range sampleIdx {
		inSample[i] = true
		sampleBlocks = append(sampleBlocks, blocks[i])
	}
	recipes := fitRecipes(sampleBlocks, isa)

	recordedSet := map[int]bool{}
	for i := range inSample {
		recordedSet[i] = true
	}
	var tmpl Block
	if len(sampleBlocks) > 0 {
		tmpl = sampleBlocks[0]
	}
	for i := range blocks {
		if inSample[i] {
			continue
		}
		syn, ok := SynthBlock(tmpl, blocks[i].X, blocks[i].Y, blocks[i].Z, recipes, isa)
		if !ok || !blockEqual(syn, blocks[i]) {
			recordedSet[i] = true
		}
	}

	plan := samplePlan{isSynth: make([]bool, len(blocks))}
	for i := range blocks {
		if recordedSet[i] {
			plan.recorded = append(plan.recorded, i)
		} else {
			plan.isSynth[i] = true
		}
	}
	sortInts(plan.recorded)
	// template is the first corner-sample block; find its slot in recorded[].
	if len(sampleIdx) > 0 {
		t := sampleIdx[0]
		for pos, i := range plan.recorded {
			if i == t {
				plan.templatePos = pos
				break
			}
		}
	}
	return plan, recipes
}

// blockEqual reports whether two decoded blocks are identical in every field that
// the .traceg emitter reads (coordinates, warp shape, and each instruction's pc,
// mask, mode and payload). This is the synthesis verification gate.
func blockEqual(a, b Block) bool {
	if a.X != b.X || a.Y != b.Y || a.Z != b.Z || len(a.Warps) != len(b.Warps) {
		return false
	}
	for w := range a.Warps {
		if len(a.Warps[w]) != len(b.Warps[w]) {
			return false
		}
		for i := range a.Warps[w] {
			x, y := a.Warps[w][i], b.Warps[w][i]
			if x.Pc != y.Pc || x.Mask != y.Mask || x.Mode != y.Mode || !eqStr(x.Payload, y.Payload) {
				return false
			}
		}
	}
	return true
}

// ── sampled columnar payload (dyn version 4) ──────────────────────────────────

// encodeDynSampled serializes only the recorded blocks (verbatim, every memory
// instruction stored as a residual) plus the recipes and the block plan. A
// synthesized block costs only one flag bit plus its coordinate delta from the
// previous block — for a dense grid those streams are near-constant and zstd
// crushes them, so the payload size is dominated by the (size-independent) sample
// rather than by the grid. This is what lets the compression ratio grow with the
// trace instead of plateauing.
func encodeDynSampled(blocks []Block, plan samplePlan, recipes map[int]Recipe, pcIdx map[int]int, isa map[int]StaticInst) ([]byte, Coverage, error) {
	cov := Coverage{ModeledPCs: len(recipes), TotalBlocks: len(blocks), RecordedBlocks: len(plan.recorded)}
	memPCs := map[int]bool{}

	// A/B/C/D: recorded blocks only, WITHOUT coordinates (coords live in the
	// shared delta stream G so the recorded/synth split stays coherent for zstd).
	var a, colPc, colMask, colAddr []byte
	a = putUvarint(a, uint64(len(plan.recorded)))
	for _, bi := range plan.recorded {
		blk := blocks[bi]
		a = putUvarint(a, uint64(len(blk.Warps)))
		for w := 0; w < len(blk.Warps); w++ {
			warp := blk.Warps[w]
			a = putUvarint(a, uint64(len(warp)))
			for _, d := range warp {
				id, ok := pcIdx[d.Pc]
				if !ok {
					return nil, cov, fmt.Errorf("pc 0x%x not in ISA", d.Pc)
				}
				colPc = putUvarint(colPc, uint64(id))
				var mb [4]byte
				binary.LittleEndian.PutUint32(mb[:], d.Mask)
				colMask = append(colMask, mb[:]...)
				if !isa[d.Pc].IsMem {
					continue
				}
				memPCs[d.Pc] = true
				cov.MemInsts++
				var rerr error
				colAddr, rerr = encodeResidual(colAddr, d.Mode, d.Payload, d.Mask)
				if rerr != nil {
					return nil, cov, fmt.Errorf("residual pc 0x%x: %w", d.Pc, rerr)
				}
			}
		}
	}
	cov.MemPCs = len(memPCs)

	// F (flags): total, template slot, then a bitset (1 = synth, 0 = recorded).
	var colFlags []byte
	colFlags = putUvarint(colFlags, uint64(len(blocks)))
	colFlags = putUvarint(colFlags, uint64(plan.templatePos))
	bs := make([]byte, (len(blocks)+7)/8)
	for i := range blocks {
		if plan.isSynth[i] {
			bs[i/8] |= 1 << uint(i%8)
		}
	}
	colFlags = append(colFlags, bs...)

	// G (coords): zig-zag delta of (x,y,z) from the previous block, every block.
	var colCoord []byte
	px, py, pz := 0, 0, 0
	for _, blk := range blocks {
		colCoord = putVarint(colCoord, int64(blk.X-px))
		colCoord = putVarint(colCoord, int64(blk.Y-py))
		colCoord = putVarint(colCoord, int64(blk.Z-pz))
		px, py, pz = blk.X, blk.Y, blk.Z
	}

	colRec := encodeRecipes(recipes, pcIdx)
	out := []byte{dynVersionSampled}
	for _, sec := range [][]byte{a, colPc, colMask, colRec, colAddr, colFlags, colCoord} {
		out = putUvarint(out, uint64(len(sec)))
		out = append(out, sec...)
	}
	return out, cov, nil
}

// decodeDynSampled reconstructs the full ordered block list: recorded blocks are
// read verbatim from the columns (their coordinates taken from the shared delta
// stream); synthesized blocks are generated from the template and recipes at the
// coordinate the stream gives them.
func decodeDynSampled(data []byte, t ISATable) ([]Block, error) {
	isa := t.toMap()
	top := &cursor{b: data, pos: 1}
	sec := make([]*cursor, 7)
	for i := 0; i < 7; i++ {
		n, err := top.uvarint()
		if err != nil {
			return nil, err
		}
		sb, err := top.bytes(int(n))
		if err != nil {
			return nil, err
		}
		sec[i] = &cursor{b: sb}
	}
	ca, cPc, cMask, cRec, cAddr, cFlags, cCoord := sec[0], sec[1], sec[2], sec[3], sec[4], sec[5], sec[6]

	recById, err := decodeRecipes(cRec)
	if err != nil {
		return nil, err
	}
	recipes := make(map[int]Recipe, len(recById))
	for id, r := range recById {
		if id < 0 || id >= len(t.Pcs) {
			return nil, fmt.Errorf("recipe pc id %d out of range", id)
		}
		recipes[t.Pcs[id]] = r
	}

	// recorded blocks (coordinates filled in during the plan walk)
	nRec, err := ca.uvarint()
	if err != nil {
		return nil, err
	}
	recorded := make([]Block, nRec)
	for bi := range recorded {
		nWarps, err := ca.uvarint()
		if err != nil {
			return nil, err
		}
		blk := Block{Warps: make([][]DynInst, nWarps)}
		for wi := uint64(0); wi < nWarps; wi++ {
			nInsts, err := ca.uvarint()
			if err != nil {
				return nil, err
			}
			warp := make([]DynInst, nInsts)
			for ii := range warp {
				id, err := cPc.uvarint()
				if err != nil {
					return nil, err
				}
				if int(id) >= len(t.Pcs) {
					return nil, fmt.Errorf("pc id %d out of range", id)
				}
				mb, err := cMask.bytes(4)
				if err != nil {
					return nil, err
				}
				d := DynInst{Pc: t.Pcs[id], Mask: binary.LittleEndian.Uint32(mb)}
				if isa[d.Pc].IsMem {
					mode, payload, err := decodeResidual(cAddr, d.Mask)
					if err != nil {
						return nil, err
					}
					d.Mode = mode
					d.Payload = payload
				}
				warp[ii] = d
			}
			blk.Warps[wi] = warp
		}
		recorded[bi] = blk
	}

	// flags: total, template slot, synth/recorded bitset
	nTotal, err := cFlags.uvarint()
	if err != nil {
		return nil, err
	}
	tmplPos, err := cFlags.uvarint()
	if err != nil {
		return nil, err
	}
	bs, err := cFlags.bytes((int(nTotal) + 7) / 8)
	if err != nil {
		return nil, err
	}
	var tmpl Block
	if int(tmplPos) < len(recorded) {
		tmpl = recorded[tmplPos]
	}

	out := make([]Block, 0, nTotal)
	recPos := 0
	px, py, pz := 0, 0, 0
	for i := uint64(0); i < nTotal; i++ {
		dx, _ := cCoord.varint()
		dy, _ := cCoord.varint()
		dz, err := cCoord.varint()
		if err != nil {
			return nil, err
		}
		px, py, pz = px+int(dx), py+int(dy), pz+int(dz)
		isSynth := bs[i/8]&(1<<uint(i%8)) != 0
		if !isSynth {
			if recPos >= len(recorded) {
				return nil, fmt.Errorf("plan references more recorded blocks than stored")
			}
			blk := recorded[recPos]
			blk.X, blk.Y, blk.Z = px, py, pz
			out = append(out, blk)
			recPos++
			continue
		}
		syn, ok := SynthBlock(tmpl, px, py, pz, recipes, isa)
		if !ok {
			return nil, fmt.Errorf("synth block (%d,%d,%d) not modelable", px, py, pz)
		}
		out = append(out, syn)
	}
	return out, nil
}
