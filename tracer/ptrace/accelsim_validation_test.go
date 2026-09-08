package ptrace

import (
	"reflect"
	"strconv"
	"strings"
	"testing"
)

const minimalInstruction = "0000 ffffffff 0 EXIT 0 0 0 "
const minimalTraceBlock = "\n#BEGIN_TB\n\nthread block = 0,0,0\n\nwarp = 0\ninsts = 1\n" + minimalInstruction + "\n\n#END_TB\n"
const minimalTrace = "-kernel name = test\n-block dim = (32,1,1)\n\n" + minimalTraceBlock

func TestInstructionRejectsMalformedFields(t *testing.T) {
	maxInt := strconv.Itoa(int(^uint(0) >> 1))
	cases := []struct{ name, line string }{
		{"empty", ""},
		{"short", "0000 ffffffff 0 EXIT"},
		{"negative pc", "-001 ffffffff 0 EXIT 0 0 0"},
		{"overflow pc", "8000000000000000 ffffffff 0 EXIT 0 0 0"},
		{"invalid pc", "nope ffffffff 0 EXIT 0 0 0"},
		{"mask overflow", "0000 100000000 0 EXIT 0 0 0"},
		{"negative mask", "0000 -1 0 EXIT 0 0 0"},
		{"invalid mask", "0000 nope 0 EXIT 0 0 0"},
		{"negative destinations", "0000 ffffffff -5 R1 LDC 0 0 0"},
		{"huge destinations", "0000 ffffffff " + maxInt + " R1 LDC 0 0 0"},
		{"noninteger destinations", "0000 ffffffff 0.5 EXIT 0 0 0"},
		{"missing destination", "0000 ffffffff 1 EXIT 0 0 0"},
		{"negative sources", "0000 ffffffff 1 R1 LDC -8 0 0"},
		{"huge sources", "0000 ffffffff 0 EXIT " + maxInt + " 0 0"},
		{"noninteger sources", "0000 ffffffff 0 EXIT x 0 0"},
		{"missing source", "0000 ffffffff 0 EXIT 1 0 0"},
		{"missing width", "0000 ffffffff 1 R1 LDC 0 0"},
		{"extra nonmemory operand", "0000 ffffffff 0 EXIT 0 0 extra 0"},
		{"negative width", "0000 ffffffff 0 LOAD 0 -4 1 0x0 4 0"},
		{"invalid width", "0000 ffffffff 0 LOAD 0 nope 1 0x0 4 0"},
		{"overflow width", "0000 ffffffff 0 LOAD 0 9223372036854775808 1 0x0 4 0"},
		{"missing mode", "0000 ffffffff 1 R1 LDG.E 0 4"},
		{"missing memory immediate", "0000 ffffffff 1 R1 LDG.E 0 4 1"},
		{"unknown mode", "0000 ffffffff 0 LOAD 0 4 3 0x0 4 0"},
		{"negative mode", "0000 ffffffff 0 LOAD 0 4 -1 0x0 4 0"},
		{"nonnumeric mode", "0000 ffffffff 0 LOAD 0 4 nope 0x0 4 0"},
		{"missing stride", "0000 ffffffff 0 LOAD 0 4 1 0x0 0"},
		{"extra stride", "0000 ffffffff 0 LOAD 0 4 1 0x0 4 4 0"},
		{"invalid base", "0000 ffffffff 0 LOAD 0 4 1 nope 4 0"},
		{"invalid stride", "0000 ffffffff 0 LOAD 0 4 1 0x0 nope 0"},
		{"overflow stride", "0000 ffffffff 0 LOAD 0 4 1 0x0 9223372036854775808 0"},
		{"short address list", "0000 00000003 0 LOAD 0 4 0 0x0 0"},
		{"extra address list", "0000 00000001 0 LOAD 0 4 0 0x0 0x4 0"},
		{"overflow address", "0000 00000001 0 LOAD 0 4 0 0x10000000000000000 0"},
		{"short deltas", "0000 00000003 0 LOAD 0 4 2 0x0 0"},
		{"bad delta", "0000 00000003 0 LOAD 0 4 2 0x0 nope 0"},
		{"zero mask missing stride", "0000 00000000 0 LOAD 0 4 1 0"},
		{"zero mask bad stride", "0000 00000000 0 LOAD 0 4 1 0x0 nope 0"},
		{"zero mask extra list", "0000 00000000 0 LOAD 0 4 0 0x0 0"},
		{"zero mask missing delta base", "0000 00000000 0 LOAD 0 4 2 0"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, _, _, err := parseLine(tc.line); err == nil {
				t.Fatal("malformed instruction accepted")
			}
		})
	}
}

func TestInstructionPreservesValidPayloads(t *testing.T) {
	lines := []string{
		minimalInstruction,
		"0010 00000000 1 R2 LDG.E 1 R4 4 0 0 ",
		"0010 00000000 1 R2 LDG.E 1 R4 4 1 0x0 0 0 ",
		"0010 00000000 1 R2 LDG.E 1 R4 4 1 0X00001234 +8 0 ",
		"0010 00000000 1 R2 LDG.E 1 R4 4 2 0X00001000 0 ",
		"0010 00000081 1 R2 LDG.E 1 R4 4 0 0x1000 0X00001004 0 ",
		"0010 80000000 1 R2 LDG.E 1 R4 4 1 0x1000 -8 0 ",
		"0010 00000003 1 R2 LDG.E 1 R4 4 2 0X00001000 +4 0 ",
		"0010 ffffffff 1 R2 LDG.E 1 R4 4 1 0x1000 -4 0 ",
	}
	for _, line := range lines {
		_, _, st, dyn, err := parseLine(line)
		if err != nil {
			t.Fatalf("valid instruction rejected: %v", err)
		}
		if got := emitLine(st, dyn); got != line {
			t.Fatalf("changed valid tokens:\n got %q\nwant %q", got, line)
		}
	}
}

func TestTracegRejectsMalformedFrames(t *testing.T) {
	cases := []struct{ name, old, replacement string }{
		{"short coordinates", "0,0,0", "0,0"},
		{"extra coordinates", "0,0,0", "0,0,0,1"},
		{"invalid coordinate", "0,0,0", "nope,0,0"},
		{"negative coordinate", "0,0,0", "0,-1,0"},
		{"overflow coordinate", "0,0,0", "0,0,9223372036854775808"},
		{"missing coordinates", "thread block = 0,0,0\n", ""},
		{"duplicate coordinates", "thread block = 0,0,0", "thread block = 0,0,0\nthread block = 1,0,0"},
		{"first warp is not zero", "warp = 0", "warp = 1"},
		{"negative warp", "warp = 0", "warp = -1"},
		{"nonnumeric warp", "warp = 0", "warp = nope"},
		{"overflow warp", "warp = 0", "warp = 9223372036854775808"},
		{"missing warp", "warp = 0\n", ""},
		{"missing instruction count", "insts = 1\n", ""},
		{"duplicate instruction count", "insts = 1", "insts = 1\ninsts = 1"},
		{"negative count", "insts = 1", "insts = -1"},
		{"invalid count", "insts = 1", "insts = nope"},
		{"overflow count", "insts = 1", "insts = 9223372036854775808"},
		{"too few instructions", "insts = 1", "insts = 99"},
		{"too many instructions", "insts = 1", "insts = 0"},
		{"missing end marker", "#END_TB", ""},
		{"duplicate end marker", "#END_TB", "#END_TB\n#END_TB"},
		{"data after end marker", "#END_TB", "#END_TB\n" + minimalInstruction},
		{"unfinished block before next", "#END_TB", minimalTraceBlock},
		{"malformed instruction", minimalInstruction, "0000 100000000 0 EXIT 0 0 0"},
		{"warp gap", "#END_TB", "warp = 2\ninsts = 0\n#END_TB"},
		{"repeated warp", "#END_TB", "warp = 0\ninsts = 0\n#END_TB"},
		{"warp missing count before next", "insts = 1", "warp = 1\ninsts = 1"},
		{"ISA inconsistency", "insts = 1\n" + minimalInstruction, "insts = 2\n" + minimalInstruction + "\n0000 ffffffff 0 OTHER 0 0 0 "},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if !strings.Contains(minimalTrace, tc.old) {
				t.Fatal("bad fixture edit")
			}
			s := strings.Replace(minimalTrace, tc.old, tc.replacement, 1)
			h, blocks, err := ParseTraceg(s, map[int]StaticInst{})
			if err == nil {
				t.Fatal("malformed text accepted")
			}
			if h != "" || blocks != nil {
				t.Fatal("failed parse returned partial output")
			}
		})
	}
	for _, text := range []string{"", "-kernel name = test\n", minimalTrace + minimalTraceBlock} {
		if _, _, err := ParseTraceg(text, map[int]StaticInst{}); err == nil {
			t.Fatal("empty/duplicate blocks accepted")
		}
	}
	if _, _, err := ParseTraceg(minimalTrace, nil); err == nil {
		t.Fatal("nil ISA table accepted")
	}
}

func TestTracegPreservesSparseAndEmptyWarps(t *testing.T) {
	texts := []string{
		minimalTrace,
		sampleTraceg(),
		strings.Replace(minimalTrace, "insts = 1\n"+minimalInstruction, "insts = 0", 1),
		strings.Replace(minimalTrace, "warp = 0\ninsts = 1\n"+minimalInstruction+"\n", "", 1),
		minimalTrace + strings.Replace(minimalTraceBlock, "0,0,0", "5,7,9", 1),
		strings.Replace(minimalTrace, "#END_TB", "warp = 1\ninsts = 0\n\n#END_TB", 1),
	}
	for _, text := range texts {
		isa := map[int]StaticInst{}
		h, blocks, err := ParseTraceg(text, isa)
		if err != nil {
			t.Fatal(err)
		}
		out, err := EmitTraceg(h, blocks, isa)
		if err != nil {
			t.Fatal(err)
		}
		other := map[int]StaticInst{}
		h2, blocks2, err := ParseTraceg(out, other)
		if err != nil || h2 != h || !reflect.DeepEqual(blocks, blocks2) || !reflect.DeepEqual(isa, other) {
			t.Fatalf("changed valid parsed content: %v", err)
		}
	}
}

func FuzzTraceInstruction(f *testing.F) {
	f.Add(minimalInstruction)
	f.Add("0010 00000000 1 R2 LDG.E 1 R4 4 1 0x0 0 0 ")
	f.Fuzz(func(t *testing.T, line string) {
		if len(line) > 4096 {
			t.Skip()
		}
		pc, mask, st, dyn, err := parseLine(line)
		if err != nil {
			return
		}
		pc2, mask2, st2, dyn2, err := parseLine(emitLine(st, dyn))
		if err != nil || pc != pc2 || mask != mask2 || st != st2 || !reflect.DeepEqual(dyn, dyn2) {
			t.Fatalf("parsed instruction changed on emission: %v", err)
		}
	})
}

func FuzzTracegFrames(f *testing.F) {
	f.Add(minimalTrace)
	f.Add(sampleTraceg())
	f.Add("-kernel name = empty\n")
	f.Fuzz(func(t *testing.T, text string) {
		if len(text) > 16384 {
			t.Skip()
		}
		isa := map[int]StaticInst{}
		h, blocks, err := ParseTraceg(text, isa)
		if err != nil {
			return
		}
		out, err := EmitTraceg(h, blocks, isa)
		if err != nil {
			t.Fatal(err)
		}
		other := map[int]StaticInst{}
		h2, blocks2, err := ParseTraceg(out, other)
		if err != nil || h != h2 || !reflect.DeepEqual(blocks, blocks2) || !reflect.DeepEqual(isa, other) {
			t.Fatalf("parsed trace changed on emission: %v", err)
		}
	})
}
