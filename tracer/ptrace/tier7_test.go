package ptrace

import (
	"math/rand"
	"testing"
)

// buildGather makes nWarps warp-loads of `lanes` addresses each, with each lane's
// index produced by idx() and address = base + idx*4.
func buildGather(nWarps, lanes int, base uint64, idx func(r *rand.Rand) uint64) [][]uint64 {
	r := rand.New(rand.NewSource(7))
	out := make([][]uint64, nWarps)
	for w := range out {
		al := make([]uint64, lanes)
		for i := range al {
			al[i] = base + idx(r)*4
		}
		out[w] = al
	}
	return out
}

func TestTier7UniformAccepted(t *testing.T) {
	// Uniform random gather over 1024 elements — the SHOC spmv case.
	insts := buildGather(400, 32, 0x7f000000, func(r *rand.Rand) uint64 { return uint64(r.Intn(1024)) })
	m, f, ok := FitAndGate(insts, 0.06, 1)
	if !ok {
		t.Fatalf("uniform gather rejected: missΔ=%.3f txnErr=%.3f", f.MaxMissDelta, f.TxnRelErr)
	}
	if m.Freq != nil {
		t.Errorf("uniform stream should fit the uniform model (Freq=nil), got histogram")
	}
	if f.ModelBytes > 40 {
		t.Errorf("uniform model should be tiny, got %d bytes", f.ModelBytes)
	}
	if f.RawBytes <= f.ModelBytes*10 {
		t.Errorf("expected large compression, raw=%d model=%d", f.RawBytes, f.ModelBytes)
	}
}

func TestTier7SkewEscalatesToHistogram(t *testing.T) {
	// Highly skewed: ~99% of accesses in a 2-line hot region, a few outliers set a
	// large range. A uniform model over that range mispredicts the miss curve; the
	// histogram captures the concentration.
	insts := buildGather(400, 32, 0x80000000, func(r *rand.Rand) uint64 {
		if r.Intn(100) == 0 {
			return uint64(3900 + r.Intn(100)) // rare outliers -> large range
		}
		return uint64(r.Intn(64)) // hot region (2 cache lines)
	})

	// the uniform model alone should miss the cache behavior
	uni := FitStatModel(insts, false, 1).Validate(insts)
	if uni.MaxMissDelta <= 0.06 {
		t.Fatalf("expected uniform model to fail on skewed stream, missΔ=%.3f", uni.MaxMissDelta)
	}

	m, f, ok := FitAndGate(insts, 0.06, 1)
	if !ok {
		t.Fatalf("skewed gather rejected even with histogram: missΔ=%.3f txnErr=%.3f", f.MaxMissDelta, f.TxnRelErr)
	}
	if m.Freq == nil {
		t.Errorf("skewed stream should escalate to the histogram model")
	}
}

func TestStatModelSerializeRoundTrip(t *testing.T) {
	insts := buildGather(50, 16, 0x1000, func(r *rand.Rand) uint64 { return uint64(r.Intn(500)) })
	m := FitStatModel(insts, true, 99)
	dec, err := DecodeStatModel(m.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if dec.Base != m.Base || dec.Elem != m.Elem || dec.Range != m.Range || dec.Seed != m.Seed {
		t.Fatalf("scalar mismatch: %+v vs %+v", dec, m)
	}
	if len(dec.Freq) != len(m.Freq) {
		t.Fatalf("freq len %d vs %d", len(dec.Freq), len(m.Freq))
	}
	for k, v := range m.LaneHist {
		if dec.LaneHist[k] != v {
			t.Fatalf("lanehist[%d]=%d, want %d", k, dec.LaneHist[k], v)
		}
	}
}

func TestCacheModelMissCurve(t *testing.T) {
	// Cyclic walk over 100 distinct lines: each revisit has reuse distance 99, so a
	// cache smaller than 100 lines misses every revisit; a larger one hits.
	var seq []uint64
	for rep := 0; rep < 50; rep++ {
		for line := 0; line < 100; line++ {
			seq = append(seq, uint64(line)<<7)
		}
	}
	rds := LineReuseDistances(seq, 7)
	small := MissRate(rds, 50)  // cache < working set
	large := MissRate(rds, 128) // cache >= working set
	if !(small > 0.9) {
		t.Errorf("expected ~all-miss below working set, got %.2f", small)
	}
	if !(large < 0.05) {
		t.Errorf("expected ~all-hit above working set, got %.2f", large)
	}
}
