package ptrace

import (
	"strconv"
	"strings"
	"testing"
)

// a minimal but structurally faithful .traceg: header + two thread blocks, each
// with a non-memory and a memory instruction (base_stride payload).
func sampleTraceg() string {
	header := "-kernel name = test\n-block dim = (64,1,1)\n\n" +
		"#traces format = [line_num] PC mask ...\n\n"
	block := func(x int) string {
		return "\n#BEGIN_TB\n" +
			"\nthread block = " + strconv.Itoa(x) + ",0,0\n" +
			"\nwarp = 0\n" +
			"insts = 2\n" +
			"0000 ffffffff 1 R1 LDC 0 0 0 \n" +
			"0010 ffffffff 1 R2 LDG.E 1 R4 4 1 0x1000 4 0 \n" +
			"\nwarp = 1\n" +
			"insts = 1\n" +
			"0010 0000000f 1 R2 LDG.E 1 R4 4 0 0x0000000000000001 0x0000000000000002 0x0000000000000003 0x0000000000000004 0 \n" +
			"\n#END_TB\n"
	}
	return header + block(0) + block(1)
}

func TestTracegRoundTrip(t *testing.T) {
	content := sampleTraceg()

	isa := map[int]StaticInst{}
	hdr, blocks, err := ParseTraceg(content, isa)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	got, err := EmitTraceg(hdr, blocks, isa)
	if err != nil {
		t.Fatalf("emit: %v", err)
	}
	if got != content {
		t.Fatalf("traceg round-trip mismatch:\n got=%q\nwant=%q", got, content)
	}
}

func TestDynColumnRoundTrip(t *testing.T) {
	content := sampleTraceg()
	isa := map[int]StaticInst{}
	_, blocks, err := ParseTraceg(content, isa)
	if err != nil {
		t.Fatal(err)
	}
	table, pcIdx := buildISA(isa)
	enc, _, err := encodeDyn(blocks, pcIdx, isa)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := decodeDyn(enc, table)
	if err != nil {
		t.Fatal(err)
	}
	reHdr := "-kernel name = test\n-block dim = (64,1,1)\n\n#traces format = [line_num] PC mask ...\n\n"
	out, err := EmitTraceg(reHdr, dec, table.toMap())
	if err != nil {
		t.Fatal(err)
	}
	if out != content {
		t.Fatalf("dyn column round-trip mismatch:\n got=%q\nwant=%q", out, content)
	}
}

// affineGrid builds nx 1-D thread blocks with one non-memory and one memory
// instruction whose base address is exactly affine in blockIdx.x
// (base = 0x1000 + 0x100*bx). Such a kernel is fully synthesizable: a corner
// sample of two blocks fixes the slope and the rest extrapolate.
func affineGrid(nx int) ([]Block, map[int]StaticInst) {
	const ldc, ldg = 0x10, 0x20
	isa := map[int]StaticInst{
		ldc: {Middle: "1 R1 LDC 0", IsMem: false, Imm: "0"},
		ldg: {Middle: "1 R2 LDG.E 1 R4", IsMem: true, Width: "4", Imm: "0"},
	}
	blocks := make([]Block, nx)
	for bx := 0; bx < nx; bx++ {
		base := 0x1000 + 0x100*bx
		warp := []DynInst{
			{Pc: ldc, Mask: 0xffffffff},
			{Pc: ldg, Mask: 0xffffffff, Mode: 1, Payload: []string{strconv.FormatInt(int64(base), 10), "4"}},
		}
		// store base as 0x-hex the way the tracer would (mode 1 base token).
		warp[1].Payload[0] = "0x" + strconv.FormatInt(int64(base), 16)
		blocks[bx] = Block{X: bx, Warps: [][]DynInst{warp}}
	}
	return blocks, isa
}

func TestSampledRoundTripAndSynthesis(t *testing.T) {
	blocks, isa := affineGrid(8)
	table, pcIdx := buildISA(isa)

	plan, recipes := buildSamplePlan(blocks, isa)
	enc, cov, err := encodeDynSampled(blocks, plan, recipes, pcIdx, isa)
	if err != nil {
		t.Fatal(err)
	}
	// corner sample is the two smallest bx; the other six must be synthesized.
	if cov.TotalBlocks != 8 || cov.RecordedBlocks != 2 {
		t.Fatalf("expected 2/8 recorded, got %d/%d", cov.RecordedBlocks, cov.TotalBlocks)
	}

	dec, err := decodeDyn(enc, table) // routes to the v4 sampled path
	if err != nil {
		t.Fatal(err)
	}
	if len(dec) != len(blocks) {
		t.Fatalf("decoded %d blocks, want %d", len(dec), len(blocks))
	}
	for i := range blocks {
		if !blockEqual(dec[i], blocks[i]) {
			t.Fatalf("block %d mismatch after sampled round-trip:\n got=%+v\nwant=%+v", i, dec[i], blocks[i])
		}
	}
	// spot-check a synthesized block regenerated the right affine base.
	if got := dec[5].Warps[0][1].Payload[0]; got != "0x1500" {
		t.Fatalf("synthesized base = %q, want 0x1500", got)
	}
}

func TestExtrapolateGrid(t *testing.T) {
	blocks, isa := affineGrid(4)
	content, nSynth, nFail, err := ExtrapolateGrid(blocks, isa, "-kernel name = t\n\n", 16, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if nSynth != 16 || nFail != 0 {
		t.Fatalf("extrapolate 16x1x1: synth=%d fail=%d", nSynth, nFail)
	}
	// the extrapolated trace must contain the affine base for a block beyond the
	// original 4 (bx=15 -> 0x1000 + 0x100*15 = 0x1f00).
	if !strings.Contains(content, "0x1f00") {
		t.Fatalf("extrapolated trace missing bx=15 base 0x1f00")
	}
}

func TestResidualRoundTrip(t *testing.T) {
	cases := []struct {
		mode    int
		mask    uint32
		payload []string
	}{
		{0, 0xff, []string{
			"0x0000000000001000", "0x0000000000001004", "0x0000000000001008", "0x000000000000100c",
			"0x0000000000001010", "0x0000000000001014", "0x0000000000001018", "0x000000000000101c"}},
		{1, 0xff, []string{"0x2000", "4"}},
		{2, 0x0f, []string{"0x3000", "8", "16", "8"}},
	}
	for _, c := range cases {
		buf, err := encodeResidual(nil, c.mode, c.payload, c.mask)
		if err != nil {
			t.Fatalf("mode %d encode: %v", c.mode, err)
		}
		cur := &cursor{b: buf}
		mode, payload, err := decodeResidual(cur, c.mask)
		if err != nil {
			t.Fatalf("mode %d decode: %v", c.mode, err)
		}
		if mode != c.mode || !eqStr(payload, c.payload) {
			t.Fatalf("mode %d round-trip: got (%d,%v) want (%d,%v)", c.mode, mode, payload, c.mode, c.payload)
		}
	}
}

func TestContainerRoundTrip(t *testing.T) {
	content := sampleTraceg()
	comp, err := EncodeContainer(content)
	if err != nil {
		t.Fatal(err)
	}
	hdr, blocks, err := DecodeContainer(comp)
	if err != nil {
		t.Fatal(err)
	}
	got := hdr + strings.Join(blocks, "")
	if got != content {
		t.Fatalf("container round-trip mismatch:\n got=%q\nwant=%q", got, content)
	}
}
