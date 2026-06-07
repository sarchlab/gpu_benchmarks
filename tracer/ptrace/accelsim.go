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
	if len(t) < 4 {
		return 0, 0, st, dyn, fmt.Errorf("short line %q", line)
	}
	p64, err := strconv.ParseInt(t[0], 16, 64)
	if err != nil {
		return 0, 0, st, dyn, fmt.Errorf("bad pc %q", t[0])
	}
	pc = int(p64)
	m64, err := strconv.ParseUint(t[1], 16, 64)
	if err != nil {
		return 0, 0, st, dyn, fmt.Errorf("bad mask %q", t[1])
	}
	mask = uint32(m64)

	i := 2
	dstNum, err := strconv.Atoi(t[i])
	if err != nil {
		return 0, 0, st, dyn, fmt.Errorf("bad dst_num %q", t[i])
	}
	i += 1 + dstNum // skip dst_num and its register tokens
	i += 1          // opcode
	if i >= len(t) {
		return 0, 0, st, dyn, fmt.Errorf("malformed (srcs) %q", line)
	}
	srcNum, err := strconv.Atoi(t[i])
	if err != nil {
		return 0, 0, st, dyn, fmt.Errorf("bad src_num %q", t[i])
	}
	i += 1 + srcNum // skip src_num and its register tokens
	if i >= len(t) {
		return 0, 0, st, dyn, fmt.Errorf("malformed (width) %q", line)
	}
	widthIdx := i
	st.Middle = strings.Join(t[2:widthIdx], " ")
	st.Imm = t[len(t)-1]

	if t[widthIdx] == "0" {
		st.IsMem = false
		// imm is the only token after the "0" marker
	} else {
		st.IsMem = true
		st.Width = t[widthIdx]
		mode, err := strconv.Atoi(t[widthIdx+1])
		if err != nil {
			return 0, 0, st, dyn, fmt.Errorf("bad mode %q", t[widthIdx+1])
		}
		dyn.Mode = mode
		dyn.Payload = t[widthIdx+2 : len(t)-1]
	}
	dyn.Pc = pc
	dyn.Mask = mask
	return pc, mask, st, dyn, nil
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
	hdr, segs := splitTracegBlocks(content)
	header = hdr
	for _, seg := range segs {
		blk := Block{}
		var cur []DynInst
		haveWarp := false
		for _, line := range strings.Split(seg, "\n") {
			s := strings.TrimSpace(line)
			switch {
			case s == "" || s == "#BEGIN_TB" || s == "#END_TB":
				continue
			case strings.HasPrefix(s, "thread block ="):
				f := strings.Split(strings.SplitN(s, "=", 2)[1], ",")
				blk.X, _ = strconv.Atoi(strings.TrimSpace(f[0]))
				blk.Y, _ = strconv.Atoi(strings.TrimSpace(f[1]))
				blk.Z, _ = strconv.Atoi(strings.TrimSpace(f[2]))
			case strings.HasPrefix(s, "warp ="):
				if haveWarp {
					blk.Warps = append(blk.Warps, cur)
				}
				cur = nil
				haveWarp = true
			case strings.HasPrefix(s, "insts ="):
				continue
			default:
				pc, _, st, dyn, perr := parseLine(line)
				if perr != nil {
					return "", nil, perr
				}
				if prev, ok := isa[pc]; ok {
					if prev != st {
						return "", nil, fmt.Errorf("ISA inconsistency at pc 0x%x: %+v vs %+v", pc, prev, st)
					}
				} else {
					isa[pc] = st
				}
				cur = append(cur, dyn)
			}
		}
		if haveWarp {
			blk.Warps = append(blk.Warps, cur)
		}
		blocks = append(blocks, blk)
	}
	return header, blocks, nil
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
