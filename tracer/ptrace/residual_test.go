package ptrace

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestResidualPreservesUnrecoverableTokens(t *testing.T) {
	cases := []struct {
		name    string
		mode    int
		mask    uint32
		payload []string
	}{
		{"zero lanes stride", 1, 0, []string{"0x0", "0"}},
		{"zero lanes nonzero base", 1, 0, []string{"0x1234", "8"}},
		{"one lane stride", 1, 1, []string{"0x1000", "16"}},
		{"one high lane negative stride", 1, 0x80000000, []string{"0x1000", "-8"}},
		{"stride spelling", 1, 3, []string{"0X00001000", "+4"}},
		{"list spelling", 0, 0x81, []string{"0x1000", "0X1004"}},
		{"delta spelling", 2, 3, []string{"0X00001000", "+4"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			buf, err := encodeResidual(nil, tc.mode, tc.payload, tc.mask)
			if err != nil {
				t.Fatal(err)
			}
			// A following compact record must remain aligned after variable-length tokens.
			buf, err = encodeResidual(buf, 1, []string{"0x2000", "4"}, 3)
			if err != nil {
				t.Fatal(err)
			}
			c := &cursor{b: buf}
			mode, payload, err := decodeResidual(c, tc.mask, true)
			if err != nil || mode != tc.mode || !eqStr(payload, tc.payload) {
				t.Fatalf("round trip = (%d, %v, %v), want (%d, %v)", mode, payload, err, tc.mode, tc.payload)
			}
			mode, payload, err = decodeResidual(c, 3, true)
			if err != nil || mode != 1 || !eqStr(payload, []string{"0x2000", "4"}) || c.pos != len(buf) {
				t.Fatalf("following record = (%d, %v, %v), consumed %d/%d", mode, payload, err, c.pos, len(buf))
			}
		})
	}
}

func TestResidualRejectsMalformedLiteral(t *testing.T) {
	valid, err := encodeResidual(nil, 1, []string{"0x0", "0"}, 0)
	if err != nil {
		t.Fatal(err)
	}
	for n := 0; n < len(valid); n++ {
		if _, _, err := decodeResidual(&cursor{b: valid[:n]}, 0, true); err == nil {
			t.Fatalf("accepted truncated literal of %d bytes", n)
		}
	}
	bad := [][]byte{
		{residualLiteral, 3, 0}, // unknown embedded mode
		append([]byte{residualLiteral, 1}, putUvarint(nil, ^uint64(0))...),    // impossible token count
		append([]byte{residualLiteral, 1, 1}, putUvarint(nil, ^uint64(0))...), // impossible token length
		{3}, // invalid compact mode, including an empty active mask
	}
	for _, buf := range bad {
		if _, _, err := decodeResidual(&cursor{b: buf}, 0, true); err == nil {
			t.Fatalf("accepted invalid residual %x", buf)
		}
	}
	if _, _, err := decodeResidual(&cursor{b: valid}, 0, false); err == nil {
		t.Fatal("accepted literal in legacy format")
	}
	for _, mode := range []int{-1, 3, 256} {
		if _, err := encodeResidual(nil, mode, nil, 0); err == nil {
			t.Fatalf("encoded unknown mode %d", mode)
		}
	}
}

func TestResidualDynamicRoundTrips(t *testing.T) {
	content := strings.ReplaceAll(sampleTraceg(),
		"0010 ffffffff 1 R2 LDG.E 1 R4 4 1 0x1000 4 0",
		"0010 00000000 1 R2 LDG.E 1 R4 4 1 0x0 0 0")
	isa := map[int]StaticInst{}
	_, blocks, err := ParseTraceg(content, isa)
	if err != nil {
		t.Fatal(err)
	}
	table, idx := buildISA(isa)
	for _, sampled := range []bool{false, true} {
		var data []byte
		if sampled {
			plan, recipes := buildSamplePlan(blocks, isa)
			data, _, err = encodeDynSampled(blocks, plan, recipes, idx, isa)
		} else {
			data, _, err = encodeDyn(blocks, idx, isa)
		}
		if err != nil {
			t.Fatal(err)
		}
		version := byte(5)
		legacyVersion := byte(3)
		if sampled {
			version = 6
			legacyVersion = 4
		}
		if data[0] != version {
			t.Fatalf("sampled=%v: writer version %d, want %d", sampled, data[0], version)
		}
		decoded, err := decodeDyn(data, table)
		if err != nil {
			t.Fatal(err)
		}
		if len(decoded) != len(blocks) {
			t.Fatalf("sampled=%v: got %d blocks, want %d", sampled, len(decoded), len(blocks))
		}
		for i := range blocks {
			if !blockEqual(decoded[i], blocks[i]) {
				t.Fatalf("sampled=%v block %d changed", sampled, i)
			}
		}
		// A new payload mislabeled as an old format must fail, not expand corrupt text.
		data[0] = legacyVersion
		if _, err := decodeDyn(data, table); err == nil {
			t.Fatalf("sampled=%v: accepted literal residual in a legacy format", sampled)
		}
	}
}

func TestLegacyResidualDynamicFixtures(t *testing.T) {
	dir := filepath.Join("testdata", "residual")
	read := func(name string) []byte {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	var table ISATable
	if err := json.Unmarshal(read("legacy-isa.json"), &table); err != nil {
		t.Fatal(err)
	}
	var want string
	if err := json.Unmarshal(read("legacy-traceg.json"), &want); err != nil {
		t.Fatal(err)
	}
	isa := map[int]StaticInst{}
	header, _, err := ParseTraceg(want, isa)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"legacy-full-v3.bin", "legacy-sampled-v4.bin"} {
		t.Run(name, func(t *testing.T) {
			blocks, err := decodeDyn(read(name), table)
			if err != nil {
				t.Fatal(err)
			}
			got, err := EmitTraceg(header, blocks, isa)
			if err != nil {
				t.Fatal(err)
			}
			if got != want {
				t.Fatal("legacy fixture no longer expands byte-identically")
			}
		})
	}
}
