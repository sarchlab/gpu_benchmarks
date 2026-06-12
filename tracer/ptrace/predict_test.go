package ptrace

import "testing"

func TestPredictCounts(t *testing.T) {
	isa := map[int]StaticInst{}
	_, blocks, err := ParseTraceg(sampleTraceg(), isa)
	if err != nil {
		t.Fatal(err)
	}
	c := PredictCounts(blocks, isa)
	// sampleTraceg: 2 blocks, each with warp0 (LDC + LDG, full mask) and
	// warp1 (LDG, 4 lanes).
	if c.Blocks != 2 || c.Warps != 4 || c.WarpInsts != 6 {
		t.Fatalf("shape: %+v", c)
	}
	// thread insts: per block 32 (LDC) + 32 (LDG) + 4 (warp1 LDG) = 68
	if c.ThreadInsts != 136 {
		t.Fatalf("thread_insts = %d, want 136", c.ThreadInsts)
	}
	if c.MemWarpInsts != 4 {
		t.Fatalf("mem_warp_insts = %d, want 4", c.MemWarpInsts)
	}
	// sectors: warp0 LDG = 0x1000 stride 4 over 32 lanes = 128B = 4 sectors;
	// warp1 LDG = addrs 1..4, one sector. 5 per block.
	if c.Sectors != 10 {
		t.Fatalf("sectors = %d, want 10", c.Sectors)
	}
}

func TestCompareCounts(t *testing.T) {
	pred := KernelCounts{WarpInsts: 1000, ThreadInsts: 32000, Sectors: 4000}
	// within tolerance
	meas := KernelCounts{WarpInsts: 1010, ThreadInsts: 32100}
	if fails := CompareCounts(pred, meas, 0.05); len(fails) != 0 {
		t.Fatalf("within-tolerance comparison failed: %v", fails)
	}
	// zero measured field (sectors not profiled) must be skipped
	meas = KernelCounts{WarpInsts: 1010}
	if fails := CompareCounts(pred, meas, 0.05); len(fails) != 0 {
		t.Fatalf("unmeasured fields must be skipped: %v", fails)
	}
	// a remainder-loop regime change shifts instruction counts well past tol
	meas = KernelCounts{WarpInsts: 1500, ThreadInsts: 32000}
	if fails := CompareCounts(pred, meas, 0.05); len(fails) != 1 {
		t.Fatalf("expected exactly the warp_insts violation, got %v", fails)
	}
}
