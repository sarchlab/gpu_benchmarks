package ptrace

// accelsim.go parses and re-emits the Accel-Sim .traceg text losslessly.
//
// A .traceg instruction line (lineinfo off, no core-id) is a space-separated
// token list with a trailing space:
//
//	PC  mask  dst_num [Rdst]  opcode  src_num [Rsrc...]  W [mode payload...]  imm
//
// Everything except `mask` and (for memory ops) `mode`+`payload` is invariant
// for a given PC, so it is hoisted into the static ISA table; only mask and the
// address payload are stored per dynamic instruction.

import (
	"fmt"
	"math/bits"
	"strconv"
	"strings"
)

// StaticInst is the per-PC invariant content of an instruction.
type StaticInst struct {
	Middle string `json:"m"`           // dst tokens + opcode + src tokens, space-joined
	IsMem  bool   `json:"mem"`         //
	Width  string `json:"w,omitempty"` // memory width token (mem only)
	Imm    string `json:"i"`           // trailing immediate token
}

// DynInst is the per-dynamic-instruction varying content.
type DynInst struct {
	Pc      int      // virtual PC (key into the kernel's static table)
	Mask    uint32   // active & predicate mask
	Mode    int      // address compression mode (mem only): 0 list_all,1 stride,2 delta
	Payload []string // address payload tokens (mem only)
}

// Block is one thread block's decoded trace.
type Block struct {
	X, Y, Z int
	Warps   [][]DynInst // index = warp id; entry may be empty
}

// parseLine splits one instruction line into static and dynamic parts.
func parseLine(line string) (pc int, mask uint32, st StaticInst, dyn DynInst, err error) {
	t := strings.Fields(line)
	if len(t) < 7 {
		return 0, 0, st, dyn, fmt.Errorf("short instruction %q", line)
	}
	p, err := strconv.ParseInt(t[0], 16, 0)
	if err != nil || p < 0 {
		return 0, 0, st, dyn, fmt.Errorf("bad pc %q", t[0])
	}
	pc = int(p)
	m, err := strconv.ParseUint(t[1], 16, 32)
	if err != nil {
		return 0, 0, st, dyn, fmt.Errorf("bad mask %q", t[1])
	}
	mask = uint32(m)

	// Leave room for opcode, source count, width and immediate. Subtract from
	// the actual token count before adding any count supplied by the trace.
	dstNum, err := strconv.Atoi(t[2])
	if err != nil || dstNum < 0 || dstNum > len(t)-7 {
		return 0, 0, st, dyn, fmt.Errorf("bad destination count in %q", line)
	}
	srcIndex := 4 + dstNum
	srcNum, err := strconv.Atoi(t[srcIndex])
	if err != nil || srcNum < 0 || srcNum > len(t)-srcIndex-3 {
		return 0, 0, st, dyn, fmt.Errorf("bad source count in %q", line)
	}
	widthIndex := srcIndex + 1 + srcNum
	st.Middle = strings.Join(t[2:widthIndex], " ")
	st.Imm = t[len(t)-1]
	if t[widthIndex] == "0" {
		if len(t) != widthIndex+2 {
			return 0, 0, st, dyn, fmt.Errorf("extra non-memory operands in %q", line)
		}
	} else {
		width, err := strconv.Atoi(t[widthIndex])
		if err != nil || width <= 0 || len(t)-widthIndex < 3 {
			return 0, 0, st, dyn, fmt.Errorf("bad memory width or missing fields in %q", line)
		}
		mode, err := strconv.Atoi(t[widthIndex+1])
		if err != nil || mode < 0 || mode > 2 {
			return 0, 0, st, dyn, fmt.Errorf("bad memory mode in %q", line)
		}
		payload := t[widthIndex+2 : len(t)-1]
		if err := validateTracePayload(mode, payload, mask); err != nil {
			return 0, 0, st, dyn, fmt.Errorf("memory payload in %q: %w", line, err)
		}
		st.IsMem, st.Width = true, t[widthIndex]
		dyn.Mode, dyn.Payload = mode, payload
	}
	dyn.Pc, dyn.Mask = pc, mask
	return pc, mask, st, dyn, nil
}

// Validate token shape without regenerating addresses: zero-mask stride/base
// tokens and noncanonical numeric spellings must survive lossless conversion.
func validateTracePayload(mode int, payload []string, mask uint32) error {
	want := bits.OnesCount32(mask)
	if mode == 1 {
		want = 2 // base and stride, including when no lane is active
	} else if mode == 2 && want == 0 {
		want = 1 // base with no deltas
	}
	if len(payload) != want {
		return fmt.Errorf("mode %d has %d tokens, want %d", mode, len(payload), want)
	}
	for i, token := range payload {
		var err error
		if mode == 0 || i == 0 {
			_, err = parseHexAddr(token)
		} else {
			_, err = strconv.ParseInt(token, 10, 64)
		}
		if err != nil {
			return fmt.Errorf("invalid token %q", token)
		}
	}
	return nil
}

// emitLine reconstructs the exact .traceg line from static + dynamic parts.
func emitLine(st StaticInst, d DynInst) string {
	toks := []string{fmt.Sprintf("%04x", d.Pc), fmt.Sprintf("%08x", d.Mask)}
	if st.Middle != "" {
		toks = append(toks, strings.Split(st.Middle, " ")...)
	}
	if st.IsMem {
		toks = append(toks, st.Width, strconv.Itoa(d.Mode))
		toks = append(toks, d.Payload...)
	} else {
		toks = append(toks, "0")
	}
	toks = append(toks, st.Imm)
	return strings.Join(toks, " ") + " "
}

// ParseTraceg parses a full .traceg string into its header and blocks, and
// accumulates the static ISA into isa (pc -> StaticInst), validating that each
// PC's static content is consistent.
func ParseTraceg(content string, isa map[int]StaticInst) (header string, blocks []Block, err error) {
	if isa == nil {
		return "", nil, fmt.Errorf("nil ISA table")
	}
	hdr, segs := splitTracegBlocks(content)
	if len(segs) == 0 {
		return "", nil, fmt.Errorf("no thread blocks")
	}
	seen := make(map[[3]int]bool)
	for i, seg := range segs {
		blk, err := parseTraceBlock(seg, isa)
		if err != nil {
			return "", nil, fmt.Errorf("block %d: %w", i, err)
		}
		coords := [3]int{blk.X, blk.Y, blk.Z}
		if seen[coords] {
			return "", nil, fmt.Errorf("duplicate thread block %v", coords)
		}
		seen[coords] = true
		blocks = append(blocks, blk)
	}
	return hdr, blocks, nil
}

func parseTraceBlock(segment string, isa map[int]StaticInst) (Block, error) {
	blk := Block{}
	begun, haveCoords, ended := false, false, false
	haveWarp := false
	declared := -1
	var cur []DynInst
	finishWarp := func() error {
		if !haveWarp {
			return nil
		}
		if declared < 0 || declared != len(cur) {
			return fmt.Errorf("warp %d declares %d instructions, found %d", len(blk.Warps), declared, len(cur))
		}
		blk.Warps = append(blk.Warps, cur)
		haveWarp, cur = false, nil
		return nil
	}
	for _, line := range strings.Split(segment, "\n") {
		s := strings.TrimSpace(line)
		if s == "" {
			continue
		}
		if ended {
			return Block{}, fmt.Errorf("content after END_TB")
		}
		switch {
		case s == "#BEGIN_TB":
			if begun {
				return Block{}, fmt.Errorf("duplicate BEGIN_TB")
			}
			begun = true
		case !begun:
			return Block{}, fmt.Errorf("missing BEGIN_TB")
		case strings.HasPrefix(s, "thread block ="):
			if haveCoords {
				return Block{}, fmt.Errorf("duplicate block coordinates")
			}
			fields := strings.Split(strings.TrimPrefix(s, "thread block ="), ",")
			if len(fields) != 3 {
				return Block{}, fmt.Errorf("expected three block coordinates")
			}
			var coords [3]int
			for i, field := range fields {
				v, err := strconv.Atoi(strings.TrimSpace(field))
				if err != nil || v < 0 {
					return Block{}, fmt.Errorf("invalid block coordinate %q", field)
				}
				coords[i] = v
			}
			blk.X, blk.Y, blk.Z = coords[0], coords[1], coords[2]
			haveCoords = true
		case !haveCoords:
			return Block{}, fmt.Errorf("missing block coordinates")
		case strings.HasPrefix(s, "warp ="):
			if err := finishWarp(); err != nil {
				return Block{}, err
			}
			w, err := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(s, "warp =")))
			if err != nil || w != len(blk.Warps) {
				return Block{}, fmt.Errorf("expected warp %d, got %q", len(blk.Warps), s)
			}
			haveWarp, declared = true, -1
		case strings.HasPrefix(s, "insts ="):
			if !haveWarp || declared >= 0 {
				return Block{}, fmt.Errorf("unexpected instruction count")
			}
			n, err := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(s, "insts =")))
			if err != nil || n < 0 {
				return Block{}, fmt.Errorf("invalid instruction count %q", s)
			}
			declared = n
		case s == "#END_TB":
			if err := finishWarp(); err != nil {
				return Block{}, err
			}
			ended = true
		default:
			if !haveWarp || declared < 0 {
				return Block{}, fmt.Errorf("instruction outside a declared warp")
			}
			if len(cur) >= declared {
				return Block{}, fmt.Errorf("more instructions than declared for warp %d", len(blk.Warps))
			}
			pc, _, st, dyn, err := parseLine(line)
			if err != nil {
				return Block{}, err
			}
			if prev, ok := isa[pc]; ok && prev != st {
				return Block{}, fmt.Errorf("ISA inconsistency at pc 0x%x: %+v vs %+v", pc, prev, st)
			}
			isa[pc] = st
			cur = append(cur, dyn)
		}
	}
	if !ended {
		return Block{}, fmt.Errorf("missing END_TB")
	}
	return blk, nil
}

// EmitTraceg reconstructs the full .traceg string (header + blocks) byte-exactly.
func EmitTraceg(header string, blocks []Block, isa map[int]StaticInst) (string, error) {
	var b strings.Builder
	b.WriteString(header)
	for _, blk := range blocks {
		b.WriteString("\n#BEGIN_TB\n")
		b.WriteString(fmt.Sprintf("\nthread block = %d,%d,%d\n", blk.X, blk.Y, blk.Z))
		for w := 0; w < len(blk.Warps); w++ {
			b.WriteString(fmt.Sprintf("\nwarp = %d\n", w))
			b.WriteString(fmt.Sprintf("insts = %d\n", len(blk.Warps[w])))
			for _, d := range blk.Warps[w] {
				st, ok := isa[d.Pc]
				if !ok {
					return "", fmt.Errorf("missing ISA for pc 0x%x", d.Pc)
				}
				b.WriteString(emitLine(st, d))
				b.WriteString("\n")
			}
		}
		b.WriteString("\n#END_TB\n")
	}
	return b.String(), nil
}
