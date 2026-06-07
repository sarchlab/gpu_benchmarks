package ptrace

// recipe.go fits per-instruction address models (P2). A memory instruction is
// "modeled" when, for each of its address-encoding classes (mode + constant
// tail), the base address is an exact integer-affine function of the launch
// indices (bx, by, bz, warp, occ). Modeled instructions store only coefficients;
// the per-instance base addresses disappear. Fitting is self-verified: any class
// that does not reproduce the recorded payload exactly is dropped to "recorded",
// so a modeled .ptrace is always losslessly equal to the original.
//
// occ = the occurrence index of a PC within its warp (a loop-iteration proxy);
// it is recomputed identically on encode and decode.

import (
	"fmt"
	"math/big"
	"strconv"
	"strings"
)

const nFeat = 5 // bx, by, bz, warp, occ

// RecipeClass is one address-encoding class of a modeled instruction.
type RecipeClass struct {
	Mode int      // 1 base_stride, 2 base_delta
	Tail []string // payload tokens after the base (constant within the class)
	Coef [6]int64 // affine base: c0*bx + c1*by + c2*bz + c3*warp + c4*occ + c5
}

// Recipe models one static memory PC.
type Recipe struct {
	Classes    []RecipeClass
	Multiclass bool // if true, a per-instance class id is stored in the selector column
}

func (r RecipeClass) base(bx, by, bz, warp, occ int) int64 {
	return r.Coef[0]*int64(bx) + r.Coef[1]*int64(by) + r.Coef[2]*int64(bz) +
		r.Coef[3]*int64(warp) + r.Coef[4]*int64(occ) + r.Coef[5]
}

// payloadFor reconstructs the exact recorded payload tokens for an instance.
func (r RecipeClass) payloadFor(bx, by, bz, warp, occ int) []string {
	out := make([]string, 0, len(r.Tail)+1)
	out = append(out, fmt.Sprintf("0x%x", uint64(r.base(bx, by, bz, warp, occ))))
	return append(out, r.Tail...)
}

// ── exact integer-affine solver ───────────────────────────────────────────────

type point struct {
	feat [nFeat]int64
	y    int64
}

// fitAffine returns coefficients c[0..4] (per feature) and c[5] (intercept) such
// that y == sum(c[i]*feat[i]) + c[5] exactly for every point, or ok=false. Only
// columns that actually vary are solved; constant columns get coefficient 0 (their
// constant contribution folds into the intercept). Requires an integer solution.
func fitAffine(pts []point) (coef [6]int64, ok bool) {
	if len(pts) == 0 {
		return coef, false
	}
	// varying columns
	varying := make([]int, 0, nFeat)
	for c := 0; c < nFeat; c++ {
		v0 := pts[0].feat[c]
		for _, p := range pts {
			if p.feat[c] != v0 {
				varying = append(varying, c)
				break
			}
		}
	}
	cols := len(varying) + 1 // + intercept
	// augmented matrix rows: [varying feats..., 1, y] over rationals
	M := make([][]*big.Rat, len(pts))
	for i, p := range pts {
		row := make([]*big.Rat, cols+1)
		for j, c := range varying {
			row[j] = new(big.Rat).SetInt64(p.feat[c])
		}
		row[len(varying)] = big.NewRat(1, 1)
		row[cols] = new(big.Rat).SetInt64(p.y)
		M[i] = row
	}
	// Gaussian elimination to RREF
	pivotCol := make([]int, 0, cols)
	r := 0
	for c := 0; c < cols && r < len(M); c++ {
		sel := -1
		for i := r; i < len(M); i++ {
			if M[i][c].Sign() != 0 {
				sel = i
				break
			}
		}
		if sel < 0 {
			continue
		}
		M[r], M[sel] = M[sel], M[r]
		inv := new(big.Rat).Inv(M[r][c])
		for j := c; j <= cols; j++ {
			M[r][j].Mul(M[r][j], inv)
		}
		for i := 0; i < len(M); i++ {
			if i != r && M[i][c].Sign() != 0 {
				f := new(big.Rat).Set(M[i][c])
				for j := c; j <= cols; j++ {
					M[i][j].Sub(M[i][j], new(big.Rat).Mul(f, M[r][j]))
				}
			}
		}
		pivotCol = append(pivotCol, c)
		r++
	}
	// consistency: no row [0...0 | nonzero]
	for i := 0; i < len(M); i++ {
		allZero := true
		for c := 0; c < cols; c++ {
			if M[i][c].Sign() != 0 {
				allZero = false
				break
			}
		}
		if allZero && M[i][cols].Sign() != 0 {
			return coef, false
		}
	}
	// extract solution: free (non-pivot) unknowns = 0
	sol := make([]*big.Rat, cols)
	for i := range sol {
		sol[i] = new(big.Rat)
	}
	for ri, c := range pivotCol {
		sol[c] = new(big.Rat).Set(M[ri][cols])
	}
	// require integer solution
	for _, s := range sol {
		if !s.IsInt() {
			return coef, false
		}
	}
	for j, c := range varying {
		coef[c] = sol[j].Num().Int64()
	}
	coef[5] = sol[len(varying)].Num().Int64()
	// final exact check
	for _, p := range pts {
		got := coef[5]
		for c := 0; c < nFeat; c++ {
			got += coef[c] * p.feat[c]
		}
		if got != p.y {
			return coef, false
		}
	}
	return coef, true
}

// ── fitting recipes from a decoded invocation ─────────────────────────────────

type memInstance struct {
	bx, by, bz, warp, occ int
	mode                  int
	base                  int64
	baseTok               string // original base token, for hex-format verification
	tail                  []string
}

// fitRecipes builds the recipe table for an invocation. Returns recipes keyed by
// PC for the modeled instructions; all other mem instructions remain recorded.
func fitRecipes(blocks []Block, isa map[int]StaticInst) map[int]Recipe {
	// gather per-PC instances (only modes 1/2 are modelable; base = payload[0]).
	byPC := map[int][]memInstance{}
	walkMem(blocks, isa, func(pc int, mi memInstance) {
		if mi.mode == 1 || mi.mode == 2 {
			byPC[pc] = append(byPC[pc], mi)
		}
	})

	recipes := map[int]Recipe{}
	for pc, insts := range byPC {
		// must observe EVERY dynamic instance of this PC as modelable (mode 1/2);
		// otherwise (some list_all) we cannot model it.
		if countPC(blocks, isa, pc) != len(insts) {
			continue
		}
		classKey := func(m memInstance) string {
			return strconv.Itoa(m.mode) + "|" + strings.Join(m.tail, " ")
		}
		order := []string{}
		groups := map[string][]memInstance{}
		idxClass := map[[4]int]string{} // (bx,by,bz,warp) -> class key
		indexDetermined := true
		for _, m := range insts {
			k := classKey(m)
			if _, ok := groups[k]; !ok {
				order = append(order, k)
			}
			groups[k] = append(groups[k], m)
			ik := [4]int{m.bx, m.by, m.bz, m.warp}
			if prev, ok := idxClass[ik]; ok && prev != k {
				indexDetermined = false
			} else {
				idxClass[ik] = k
			}
		}
		// Gate: model only when the encoding class is index-determined and there
		// is real tail reuse. A PC whose class changes within one (block,warp) —
		// or that has a distinct class per instance — is data-dependent (gather /
		// PRNG): recording it is honest and avoids bloating the recipe table.
		if len(order) != 1 && !(indexDetermined && len(order)*2 <= len(insts)) {
			continue
		}
		classes := make([]RecipeClass, 0, len(order))
		ok := true
		for _, k := range order {
			grp := groups[k]
			pts := make([]point, len(grp))
			for i, m := range grp {
				pts[i] = point{feat: [nFeat]int64{int64(m.bx), int64(m.by), int64(m.bz), int64(m.warp), int64(m.occ)}, y: m.base}
			}
			coef, fit := fitAffine(pts)
			if !fit {
				ok = false
				break
			}
			cls := RecipeClass{Mode: grp[0].mode, Tail: grp[0].tail, Coef: coef}
			// Self-verify: the regenerated base token must match the original
			// exactly (guards the hex format), else fall back to recorded.
			for _, m := range grp {
				if fmt.Sprintf("0x%x", uint64(cls.base(m.bx, m.by, m.bz, m.warp, m.occ))) != m.baseTok {
					ok = false
					break
				}
			}
			if !ok {
				break
			}
			classes = append(classes, cls)
		}
		if !ok {
			continue
		}
		recipes[pc] = Recipe{Classes: classes, Multiclass: len(classes) > 1}
	}
	return recipes
}

// classIndexFor returns which class an instance belongs to (by mode+tail).
func (r Recipe) classIndexFor(mode int, tail []string) int {
	for i, c := range r.Classes {
		if c.Mode == mode && eqStr(c.Tail, tail) {
			return i
		}
	}
	return -1
}

func eqStr(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// walkMem invokes fn for every memory instruction in canonical order with its
// (bx,by,bz,warp,occ) context. base/tail are filled for modes 1/2.
func walkMem(blocks []Block, isa map[int]StaticInst, fn func(pc int, mi memInstance)) {
	for _, blk := range blocks {
		for w := 0; w < len(blk.Warps); w++ {
			occ := map[int]int{}
			for _, d := range blk.Warps[w] {
				o := occ[d.Pc]
				occ[d.Pc] = o + 1
				st := isa[d.Pc]
				if !st.IsMem {
					continue
				}
				mi := memInstance{bx: blk.X, by: blk.Y, bz: blk.Z, warp: w, occ: o, mode: d.Mode}
				if (d.Mode == 1 || d.Mode == 2) && len(d.Payload) >= 1 {
					if v, err := strconv.ParseUint(strings.TrimPrefix(d.Payload[0], "0x"), 16, 64); err == nil {
						mi.base = int64(v)
						mi.baseTok = d.Payload[0]
						mi.tail = d.Payload[1:]
					} else {
						mi.mode = 0 // force record on parse failure
					}
				}
				fn(d.Pc, mi)
			}
		}
	}
}

// ── recipe table serialization (keyed by dense pc id) ─────────────────────────

func encodeRecipes(recipes map[int]Recipe, pcIdx map[int]int) []byte {
	ids := make([]int, 0, len(recipes))
	for pc := range recipes {
		ids = append(ids, pcIdx[pc])
	}
	sortInts(ids)
	byId := map[int]Recipe{}
	for pc, r := range recipes {
		byId[pcIdx[pc]] = r
	}
	var b []byte
	b = putUvarint(b, uint64(len(ids)))
	for _, id := range ids {
		r := byId[id]
		b = putUvarint(b, uint64(id))
		if r.Multiclass {
			b = append(b, 1)
		} else {
			b = append(b, 0)
		}
		b = putUvarint(b, uint64(len(r.Classes)))
		for _, c := range r.Classes {
			b = append(b, byte(c.Mode))
			b = putUvarint(b, uint64(len(c.Tail)))
			for _, tok := range c.Tail {
				b = putUvarint(b, uint64(len(tok)))
				b = append(b, tok...)
			}
			for _, coef := range c.Coef {
				b = putVarint(b, coef)
			}
		}
	}
	return b
}

func decodeRecipes(c *cursor) (map[int]Recipe, error) {
	out := map[int]Recipe{}
	if len(c.b) == 0 {
		return out, nil
	}
	n, err := c.uvarint()
	if err != nil {
		return nil, err
	}
	for i := uint64(0); i < n; i++ {
		id, err := c.uvarint()
		if err != nil {
			return nil, err
		}
		mcB, err := c.bytes(1)
		if err != nil {
			return nil, err
		}
		nClasses, err := c.uvarint()
		if err != nil {
			return nil, err
		}
		r := Recipe{Multiclass: mcB[0] == 1, Classes: make([]RecipeClass, nClasses)}
		for ci := range r.Classes {
			modeB, err := c.bytes(1)
			if err != nil {
				return nil, err
			}
			nTail, err := c.uvarint()
			if err != nil {
				return nil, err
			}
			tail := make([]string, nTail)
			for ti := range tail {
				l, err := c.uvarint()
				if err != nil {
					return nil, err
				}
				tb, err := c.bytes(int(l))
				if err != nil {
					return nil, err
				}
				tail[ti] = string(tb)
			}
			var coef [6]int64
			for k := 0; k < 6; k++ {
				v, err := c.varint()
				if err != nil {
					return nil, err
				}
				coef[k] = v
			}
			r.Classes[ci] = RecipeClass{Mode: int(modeB[0]), Tail: tail, Coef: coef}
		}
		out[int(id)] = r
	}
	return out, nil
}

func sortInts(a []int) {
	for i := 1; i < len(a); i++ {
		for j := i; j > 0 && a[j-1] > a[j]; j-- {
			a[j-1], a[j] = a[j], a[j-1]
		}
	}
}

func countPC(blocks []Block, isa map[int]StaticInst, pc int) int {
	n := 0
	for _, blk := range blocks {
		for _, warp := range blk.Warps {
			for _, d := range warp {
				if d.Pc == pc && isa[d.Pc].IsMem {
					n++
				}
			}
		}
	}
	return n
}
