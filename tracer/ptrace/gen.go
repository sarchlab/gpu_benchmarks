package ptrace

// gen.go ingests live-sampled trace containers (P1b: the NVBit tracer recorded
// only the corner-sample blocks) and regenerates the full-grid trace: recorded
// blocks are emitted verbatim, every missing block is synthesized from recipes
// fit on the recorded ones.
//
// Unlike the P3 path (synthSampled), there is no local ground truth for the
// synthesized blocks — the tracer never recorded them. Soundness comes from
// the external gates instead: the regime check between two traced sizes
// (CheckRegime) and the dynamic-count fingerprint against an ncu profile
// (PredictCounts/CompareCounts). The self-check here only confirms that the
// fitted recipes reproduce the recorded blocks themselves (affine
// consistency), which catches non-affine kernels immediately.
//
// Known limitation: every synthesized block reuses the first recorded block's
// per-warp shape (instruction sequence + masks). Kernels whose far-edge blocks
// diverge (bounds-check tails, partial warps) are reproduced incorrectly at
// those coordinates; the counts gate is what catches this class.

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// GenResult reports what GenFromSampled did for one kernel container.
type GenResult struct {
	Grid        [3]int `json:"grid"`
	TotalBlocks int    `json:"total_blocks"`
	Recorded    int    `json:"recorded_blocks"`
	Synthesized int    `json:"synthesized_blocks"`
	Failed      int    `json:"failed_blocks"` // unmodeled mem PC — omitted from output
	Passthrough bool   `json:"passthrough"`   // container already had the full grid
	// Exceptions counts recorded blocks whose shape differs from the majority
	// (boundary blocks at the grid origin, partial warps). They are emitted
	// verbatim; recipes are fit on the majority-shape blocks only.
	Exceptions int `json:"exception_blocks"`
	// SelfCheckMismatch counts majority-shape recorded blocks the fitted
	// recipes cannot reproduce (affine inconsistency among the recorded
	// blocks themselves).
	SelfCheckMismatch int `json:"self_check_mismatch"`
}

// shapeSig fingerprints a block's control shape: per-warp (pc, mask, mode)
// sequences. Blocks in the same regime share it; boundary blocks (masked-off
// lanes, extra/missing iterations) differ.
func shapeSig(b Block) string {
	var sb strings.Builder
	for _, w := range b.Warps {
		for _, d := range w {
			sb.WriteString(strconv.Itoa(d.Pc))
			sb.WriteByte(':')
			sb.WriteString(strconv.FormatUint(uint64(d.Mask), 16))
			sb.WriteByte(':')
			sb.WriteString(strconv.Itoa(d.Mode))
			sb.WriteByte(',')
		}
		sb.WriteByte('|')
	}
	return sb.String()
}

var gridDimRe = regexp.MustCompile(`-grid dim = \((\d+),(\d+),(\d+)\)`)

// GenFromSampled regenerates the full-grid .traceg content from a sampled
// container's raw bytes.
func GenFromSampled(raw []byte) (string, GenResult, error) {
	var res GenResult

	header, blockTexts, err := DecodeContainer(raw)
	if err != nil {
		return "", res, err
	}
	isa := map[int]StaticInst{}
	_, blocks, err := ParseTraceg(header+strings.Join(blockTexts, ""), isa)
	if err != nil {
		return "", res, err
	}
	if len(blocks) == 0 {
		return "", res, fmt.Errorf("container has no blocks")
	}

	m := gridDimRe.FindStringSubmatch(header)
	if m == nil {
		return "", res, fmt.Errorf("no '-grid dim' in header")
	}
	for i := 0; i < 3; i++ {
		v, err := strconv.Atoi(m[i+1])
		if err != nil || v < 1 {
			return "", res, fmt.Errorf("bad grid dim %q", m[i+1])
		}
		res.Grid[i] = v
	}
	gx, gy, gz := res.Grid[0], res.Grid[1], res.Grid[2]
	res.TotalBlocks = gx * gy * gz
	res.Recorded = len(blocks)

	recorded := make(map[[3]int]Block, len(blocks))
	for _, b := range blocks {
		recorded[[3]int{b.X, b.Y, b.Z}] = b
	}

	if res.Recorded >= res.TotalBlocks {
		// Full trace — nothing to synthesize.
		res.Passthrough = true
		content, err := EmitTraceg(header, blocks, isa)
		return content, res, err
	}

	// Split recorded blocks into the majority shape class (the regime
	// synthesized blocks will follow) and exceptions (boundary blocks —
	// recorded verbatim, excluded from recipe fitting so they don't poison
	// the affine fit).
	bySig := map[string][]Block{}
	for _, b := range blocks {
		s := shapeSig(b)
		bySig[s] = append(bySig[s], b)
	}
	var majority []Block
	for _, grp := range bySig {
		if len(grp) > len(majority) {
			majority = grp
		}
	}
	res.Exceptions = len(blocks) - len(majority)

	recipes := fitRecipes(majority, isa)
	tmpl := majority[0]

	// Self-check: the recipes must reproduce the blocks they were fit on.
	for _, b := range majority {
		syn, ok := SynthBlock(tmpl, b.X, b.Y, b.Z, recipes, isa)
		if !ok {
			res.SelfCheckMismatch++
			continue
		}
		if _, _, full := compareMem(syn, b, isa); !full {
			res.SelfCheckMismatch++
		}
	}

	// Emit in the tracer's canonical linear order: x fastest, then y, then z.
	out := make([]Block, 0, res.TotalBlocks)
	for z := 0; z < gz; z++ {
		for y := 0; y < gy; y++ {
			for x := 0; x < gx; x++ {
				if b, ok := recorded[[3]int{x, y, z}]; ok {
					out = append(out, b)
					continue
				}
				b, ok := SynthBlock(tmpl, x, y, z, recipes, isa)
				if !ok {
					res.Failed++
					continue
				}
				out = append(out, b)
				res.Synthesized++
			}
		}
	}

	content, err := EmitTraceg(header, out, isa)
	return content, res, err
}
