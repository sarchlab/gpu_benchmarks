package ptrace

// ptrace.go defines the on-disk .ptrace format and its (de)serialization.
//
// Layout (a directory):
//
//	<name>.ptrace/
//	  manifest.json         ordered kernelslist entries, contexts, invocations
//	  isa/<hash>.json       static per-PC table, deduped across invocations
//	  dyn/inv_<n>.zst       columnar dynamic payload (zstd of pc/mask/addr columns)
//
// P1 is lossless and does no address modeling: the address payload is stored
// verbatim in the addr column. The savings come from (1) hoisting the invariant
// opcode/registers/immediate into the shared ISA table, (2) deduping the ISA
// across relaunches, and (3) a single full-window zstd stream per invocation.

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"sort"

	"github.com/klauspost/compress/zstd"
)

const FormatVersion = 2

// ISATable is the static program of one kernel: pcs[i] has static content insts[i].
type ISATable struct {
	Pcs   []int        `json:"pcs"`
	Insts []StaticInst `json:"insts"`
}

func buildISA(isa map[int]StaticInst) (ISATable, map[int]int) {
	pcs := make([]int, 0, len(isa))
	for pc := range isa {
		pcs = append(pcs, pc)
	}
	sort.Ints(pcs)
	t := ISATable{Pcs: pcs, Insts: make([]StaticInst, len(pcs))}
	idx := make(map[int]int, len(pcs))
	for i, pc := range pcs {
		t.Insts[i] = isa[pc]
		idx[pc] = i
	}
	return t, idx
}

func (t ISATable) hash() string {
	b, _ := json.Marshal(t)
	h := sha256.Sum256(b)
	return fmt.Sprintf("%x", h[:16])
}

func (t ISATable) toMap() map[int]StaticInst {
	m := make(map[int]StaticInst, len(t.Pcs))
	for i, pc := range t.Pcs {
		m[pc] = t.Insts[i]
	}
	return m
}

// ── Manifest ────────────────────────────────────────────────────────────────

type Entry struct {
	Type string `json:"type"`          // "memcpy" | "kernel"
	Raw  string `json:"raw"`           // memcpy: verbatim line; kernel: trace filename ref
	Inv  int    `json:"inv,omitempty"` // kernel: index into Invocations
}

type Context struct {
	KernelslistName string  `json:"kernelslist"`
	StatsName       string  `json:"stats,omitempty"`
	StatsContent    string  `json:"stats_content,omitempty"`
	Entries         []Entry `json:"entries"`
}

type Invocation struct {
	TraceFile string   `json:"trace_file"` // original .trace.zst filename to regenerate
	Header    string   `json:"header"`     // verbatim .traceg header
	ISA       string   `json:"isa"`        // isa table hash
	DynFile   string   `json:"dyn"`        // dyn payload filename
	Cov       Coverage `json:"coverage"`   // modeled vs recorded summary
}

type Manifest struct {
	Version     int          `json:"version"`
	Source      string       `json:"source"`
	Contexts    []Context    `json:"contexts"`
	Invocations []Invocation `json:"invocations"`
}

// ── Columnar dynamic payload ──────────────────────────────────────────────────

func putUvarint(buf []byte, v uint64) []byte {
	var tmp [binary.MaxVarintLen64]byte
	n := binary.PutUvarint(tmp[:], v)
	return append(buf, tmp[:n]...)
}

func putVarint(buf []byte, v int64) []byte {
	var tmp [binary.MaxVarintLen64]byte
	n := binary.PutVarint(tmp[:], v)
	return append(buf, tmp[:n]...)
}

const dynVersion = 3

// Coverage summarizes how much of an invocation's memory traffic is modeled.
// TotalBlocks/RecordedBlocks are set only in sampled (P3) mode: the recorded
// fraction is the directly-measured proxy for collection wall-time.
type Coverage struct {
	MemPCs         int `json:"mem_pcs"`
	ModeledPCs     int `json:"modeled_pcs"`
	MemInsts       int `json:"mem_insts"`
	ModeledInsts   int `json:"modeled_insts"`
	TotalBlocks    int `json:"total_blocks,omitempty"`
	RecordedBlocks int `json:"recorded_blocks,omitempty"`
}

// encodeDyn serializes blocks into the uncompressed columnar payload (v2).
// Sections: [A struct][B pc][C mask][R recipes][S selector][D residual addr].
// Memory instructions whose PC is modeled store nothing in D (and only a class id
// in S when multiclass); their addresses are regenerated from the recipe.
func encodeDyn(blocks []Block, pcIdx map[int]int, isa map[int]StaticInst) ([]byte, Coverage, error) {
	recipes := fitRecipes(blocks, isa)
	cov := Coverage{ModeledPCs: len(recipes)}
	memPCs := map[int]bool{}

	var a, colPc, colMask, colSel, colAddr []byte
	a = putUvarint(a, uint64(len(blocks)))
	for _, blk := range blocks {
		a = putUvarint(a, uint64(blk.X))
		a = putUvarint(a, uint64(blk.Y))
		a = putUvarint(a, uint64(blk.Z))
		a = putUvarint(a, uint64(len(blk.Warps)))
		for w := 0; w < len(blk.Warps); w++ {
			warp := blk.Warps[w]
			a = putUvarint(a, uint64(len(warp)))
			occ := map[int]int{}
			for _, d := range warp {
				o := occ[d.Pc]
				occ[d.Pc] = o + 1
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
				if rec, ok := recipes[d.Pc]; ok {
					cov.ModeledInsts++
					if rec.Multiclass {
						ci := rec.classIndexFor(d.Mode, d.Payload[1:])
						colSel = putUvarint(colSel, uint64(ci))
					}
					continue // addresses regenerated from recipe
				}
				var rerr error
				colAddr, rerr = encodeResidual(colAddr, d.Mode, d.Payload, d.Mask)
				if rerr != nil {
					return nil, cov, fmt.Errorf("residual pc 0x%x: %w", d.Pc, rerr)
				}
			}
		}
	}
	colRec := encodeRecipes(recipes, pcIdx)
	cov.MemPCs = len(memPCs)

	out := []byte{dynVersion}
	for _, sec := range [][]byte{a, colPc, colMask, colRec, colSel, colAddr} {
		out = putUvarint(out, uint64(len(sec)))
		out = append(out, sec...)
	}
	return out, cov, nil
}

type cursor struct {
	b   []byte
	pos int
}

func (c *cursor) uvarint() (uint64, error) {
	v, n := binary.Uvarint(c.b[c.pos:])
	if n <= 0 {
		return 0, fmt.Errorf("bad uvarint at %d", c.pos)
	}
	c.pos += n
	return v, nil
}
func (c *cursor) varint() (int64, error) {
	v, n := binary.Varint(c.b[c.pos:])
	if n <= 0 {
		return 0, fmt.Errorf("bad varint at %d", c.pos)
	}
	c.pos += n
	return v, nil
}
func (c *cursor) bytes(n int) ([]byte, error) {
	if c.pos+n > len(c.b) {
		return nil, fmt.Errorf("short read")
	}
	r := c.b[c.pos : c.pos+n]
	c.pos += n
	return r, nil
}

func decodeDyn(data []byte, t ISATable) ([]Block, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("empty dyn payload")
	}
	if data[0] == dynVersionSampled {
		return decodeDynSampled(data, t)
	}
	isa := t.toMap()
	if data[0] != dynVersion {
		return nil, fmt.Errorf("unsupported dyn version")
	}
	top := &cursor{b: data, pos: 1}
	sec := make([]*cursor, 6)
	for i := 0; i < 6; i++ {
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
	ca, cPc, cMask, cRec, cSel, cAddr := sec[0], sec[1], sec[2], sec[3], sec[4], sec[5]

	recipes, err := decodeRecipes(cRec)
	if err != nil {
		return nil, err
	}

	nBlocks, err := ca.uvarint()
	if err != nil {
		return nil, err
	}
	blocks := make([]Block, nBlocks)
	for bi := range blocks {
		x, _ := ca.uvarint()
		y, _ := ca.uvarint()
		z, _ := ca.uvarint()
		nWarps, err := ca.uvarint()
		if err != nil {
			return nil, err
		}
		blk := Block{X: int(x), Y: int(y), Z: int(z), Warps: make([][]DynInst, nWarps)}
		for wi := uint64(0); wi < nWarps; wi++ {
			nInsts, err := ca.uvarint()
			if err != nil {
				return nil, err
			}
			occ := map[int]int{}
			warp := make([]DynInst, nInsts)
			for ii := range warp {
				id, err := cPc.uvarint()
				if err != nil {
					return nil, err
				}
				if int(id) >= len(t.Pcs) {
					return nil, fmt.Errorf("pc id %d out of range", id)
				}
				pc := t.Pcs[id]
				mb, err := cMask.bytes(4)
				if err != nil {
					return nil, err
				}
				d := DynInst{Pc: pc, Mask: binary.LittleEndian.Uint32(mb)}
				o := occ[pc]
				occ[pc] = o + 1
				if isa[pc].IsMem {
					if rec, ok := recipes[int(id)]; ok {
						ci := 0
						if rec.Multiclass {
							v, err := cSel.uvarint()
							if err != nil {
								return nil, err
							}
							ci = int(v)
						}
						cls := rec.Classes[ci]
						d.Mode = cls.Mode
						d.Payload = cls.payloadFor(blk.X, blk.Y, blk.Z, int(wi), o)
					} else {
						mode, payload, err := decodeResidual(cAddr, d.Mask)
						if err != nil {
							return nil, err
						}
						d.Mode = mode
						d.Payload = payload
					}
				}
				warp[ii] = d
			}
			blk.Warps[wi] = warp
		}
		blocks[bi] = blk
	}
	return blocks, nil
}

// ── zstd helpers for the dyn payload ──────────────────────────────────────────

func zstdCompress(data []byte) ([]byte, error) {
	enc, err := zstd.NewWriter(nil, zstd.WithEncoderLevel(zstd.SpeedBetterCompression))
	if err != nil {
		return nil, err
	}
	defer enc.Close()
	return enc.EncodeAll(data, nil), nil
}

func zstdDecompress(data []byte) ([]byte, error) {
	dec, err := zstd.NewReader(nil)
	if err != nil {
		return nil, err
	}
	defer dec.Close()
	return dec.DecodeAll(data, nil)
}
