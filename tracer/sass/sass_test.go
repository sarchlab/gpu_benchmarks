package sass

import "testing"

// A minimal but format-faithful SASS snippet (cuobjdump -sass style) exercising:
// an index-guarded EXIT, an affine load (arg0[tid]), an indirect gather
// (arg1[loaded]), a data-dependent loop back-edge, and a second kernel with an
// atomic (nondeterministic).
const sample = `
		Function : _Z4testPiS_i
        /*0000*/                   S2R R0, SR_TID.X ;                    /* 0x... */
        /*0010*/                   ISETP.GE.AND P0, PT, R0, c[0x0][0x238], PT ;
        /*0020*/               @P0 EXIT ;
        /*0030*/                   LDC.64 R4, c[0x0][0x210] ;
        /*0040*/                   IMAD.WIDE R4, R0, 0x4, R4 ;
        /*0050*/                   LDG.E R5, desc[UR4][R4.64] ;
        /*0060*/                   LDC.64 R6, c[0x0][0x218] ;
        /*0070*/                   IMAD.WIDE R6, R5, 0x4, R6 ;
        /*0080*/                   LDG.E R7, desc[UR4][R6.64] ;
        /*0090*/                   ISETP.NE.AND P1, PT, R7, RZ, PT ;
        /*00a0*/               @P1 BRA 0x50 ;
        /*00b0*/                   EXIT ;
		Function : _Z5atomkPii
        /*0000*/                   S2R R0, SR_TID.X ;
        /*0010*/                   ATOMG.ADD R2, desc[UR4][R4.64], R0 ;
        /*0020*/                   EXIT ;
`

func TestAnalyzeSpMVShape(t *testing.T) {
	ks := ParseKernels(sample)
	if len(ks) != 2 {
		t.Fatalf("parsed %d kernels, want 2", len(ks))
	}
	a := ks[0].Analyze()

	// memory tiers
	if len(a.Mem) != 2 {
		t.Fatalf("mem accesses = %d, want 2", len(a.Mem))
	}
	if a.Mem[0].Tier != "affine" || a.Mem[0].Base != ProvIndex {
		t.Errorf("load0: got tier=%s base=%s, want affine/index", a.Mem[0].Tier, a.Mem[0].Base)
	}
	if !hasArg(a.Mem[0].Args, 0x210) {
		t.Errorf("affine load should reference arg0 (0x210), got %v", a.Mem[0].Args)
	}
	if a.Mem[1].Tier != "indirect" || a.Mem[1].Base != ProvLoaded {
		t.Errorf("gather: got tier=%s base=%s, want indirect/loaded", a.Mem[1].Tier, a.Mem[1].Base)
	}
	if !hasArg(a.Mem[1].Args, 0x218) {
		t.Errorf("gather should reference arg1 (0x218), got %v", a.Mem[1].Args)
	}

	// branch classes
	cls := map[uint64]Branch{}
	for _, b := range a.Branches {
		cls[b.Addr] = b
	}
	if cls[0x20].Class != "index" {
		t.Errorf("EXIT guard: got %s, want index", cls[0x20].Class)
	}
	if cls[0xa0].Class != "data" || !cls[0xa0].BackEdge {
		t.Errorf("loop branch: got class=%s backedge=%v, want data/true", cls[0xa0].Class, cls[0xa0].BackEdge)
	}

	// determinism
	if !a.Deterministic {
		t.Errorf("kernel 0 should be deterministic, reasons=%v", a.NonDetReasons)
	}
	if a.DataDependent != true {
		t.Errorf("kernel 0 has a gather + data loop, should be DataDependent")
	}

	// kernel 1: atomic => nondeterministic
	a2 := ks[1].Analyze()
	if a2.Deterministic {
		t.Errorf("kernel with ATOMG should be nondeterministic")
	}
}

func hasArg(args []uint64, off uint64) bool {
	for _, a := range args {
		if a == off {
			return true
		}
	}
	return false
}
