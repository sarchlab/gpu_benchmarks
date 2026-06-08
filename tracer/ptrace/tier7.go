package ptrace

// tier7.go is the lossy "statistical residual" encoder. A data-dependent gather
// (e.g. SpMV's x[col[]]) carries near-maximal address entropy, so it cannot be
// compressed losslessly. But for cache/performance simulation the *exact*
// addresses do not matter — only the access *pattern* the memory system sees.
//
// tier-7 replaces the recorded index stream with a compact statistical model plus
// a seed. The synthesizer regenerates a stream that is NOT the original but
// stresses the caches equivalently: indices are drawn i.i.d. from a distribution
// over a working set, and each warp-load's lanes are drawn independently so the
// coalescing (memory-transaction count) is reproduced. It is opt-in and lossy,
// gated by cache-behavior equivalence (cache.go), and must never be used for
// functional replay.
//
// Model class (escalated by the gate, simplest first):
//   - uniform:   just the working-set range          (~tens of bytes)
//   - histogram: a frequency histogram over the range (captures hot/cold skew)
// Higher-order correlation (bursty/banded reuse) is future work.

import (
	"math/bits"
	"math/rand"
)

const (
	statLineBits   = 7  // 128 B cache line (temporal-locality granularity)
	statSectorBits = 5  // 32 B sector (memory-transaction granularity)
	statBuckets    = 64 // frequency-histogram resolution
)

// statCacheSizesB are the cache sizes (bytes) the equivalence gate checks the
// miss-rate curve at, spanning GPU L1 through L2.
var statCacheSizesB = []int{512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576}

// StatModel is a compact, regenerable description of a data-dependent address
// stream's cache-relevant statistics. An address is Base + index*Elem, with index
// drawn i.i.d. from a distribution over [0,Range): uniform when Freq==nil, else
// the Freq histogram. Per-warp active-lane counts replay LaneHist.
type StatModel struct {
	Base, Elem, Range uint64
	NWarps            int
	LaneHist          map[int]int // active-lane-count -> #warp-loads
	Freq              []uint32    // index-frequency histogram (len statBuckets), or nil for uniform
	Seed              uint64
}

// GatherStreams extracts, per static memory PC, the per-warp-load active-lane
// address lists of its data-dependent accesses (mode-2 base_delta with a full-ish
// warp) — the candidates for tier-7 statistical modeling.
func GatherStreams(blocks []Block, isa map[int]StaticInst) map[int][][]uint64 {
	out := map[int][][]uint64{}
	for _, blk := range blocks {
		for _, warp := range blk.Warps {
			for _, d := range warp {
				if !isa[d.Pc].IsMem || d.Mode != 2 {
					continue
				}
				if bits.OnesCount32(d.Mask) < 8 {
					continue
				}
				al, err := payloadToAddrs(d.Mode, d.Payload, d.Mask)
				if err != nil {
					continue
				}
				out[d.Pc] = append(out[d.Pc], al)
			}
		}
	}
	return out
}

// FitStatModel measures the working set, element size, per-warp lane counts, and
// (optionally) the index-frequency histogram of an address stream.
func FitStatModel(insts [][]uint64, withHist bool, seed uint64) StatModel {
	m := StatModel{Seed: seed, NWarps: len(insts), LaneHist: map[int]int{}}
	var lo uint64 = ^uint64(0)
	var hi uint64
	for _, al := range insts {
		m.LaneHist[len(al)]++
		for _, a := range al {
			if a < lo {
				lo = a
			}
			if a > hi {
				hi = a
			}
		}
	}
	if lo == ^uint64(0) {
		lo = 0
	}
	var g uint64
	for _, al := range insts {
		for _, a := range al {
			g = gcdU(g, a-lo)
		}
	}
	if g == 0 {
		g = 1
	}
	m.Base, m.Elem = lo, g
	m.Range = (hi-lo)/g + 1
	if withHist && m.Range > 0 {
		m.Freq = make([]uint32, statBuckets)
		for _, al := range insts {
			for _, a := range al {
				idx := (a - lo) / g
				b := int(idx * uint64(statBuckets) / m.Range)
				if b >= statBuckets {
					b = statBuckets - 1
				}
				m.Freq[b]++
			}
		}
	}
	return m
}

// Synthesize regenerates a cache-equivalent address stream from the model + seed.
func (m StatModel) Synthesize() [][]uint64 {
	rng := rand.New(rand.NewSource(int64(m.Seed)))
	// Expand the lane-count histogram into NWarps loads (deterministic order; for
	// i.i.d. draws the order does not affect the reuse-distance distribution).
	var laneCounts []int
	for _, c := range sortedIntKeys(m.LaneHist) {
		for i := 0; i < m.LaneHist[c]; i++ {
			laneCounts = append(laneCounts, c)
		}
	}
	var cum []uint32
	var total uint32
	if m.Freq != nil {
		cum = make([]uint32, len(m.Freq))
		for i, f := range m.Freq {
			total += f
			cum[i] = total
		}
	}
	rangeMax := m.Range
	if rangeMax < 1 {
		rangeMax = 1
	}
	out := make([][]uint64, len(laneCounts))
	for w, lc := range laneCounts {
		al := make([]uint64, lc)
		for i := 0; i < lc; i++ {
			var idx uint64
			if total == 0 {
				idx = uint64(rng.Int63n(int64(rangeMax)))
			} else {
				r := uint32(rng.Int63n(int64(total)))
				b := upperBound(cum, r)
				blo := uint64(b) * m.Range / uint64(len(m.Freq))
				bhi := uint64(b+1) * m.Range / uint64(len(m.Freq))
				if bhi <= blo {
					bhi = blo + 1
				}
				idx = blo + uint64(rng.Int63n(int64(bhi-blo)))
			}
			al[i] = m.Base + idx*m.Elem
		}
		out[w] = al
	}
	return out
}

// Fidelity reports how well a model's synthesized stream reproduces the original's
// cache behavior, plus its compression.
type Fidelity struct {
	MaxMissDelta float64 // max |miss_real - miss_synth| over the checked cache sizes
	TxnRelErr    float64 // relative error in mean memory-transactions per warp-load
	ModelBytes   int     // serialized model size
	RawBytes     int     // lossless entropy-floor estimate of the stream
}

// Validate synthesizes from the model and compares cache behavior against the real
// stream: the LRU miss-rate curve (temporal locality) and the per-warp
// transaction count (coalescing).
func (m StatModel) Validate(real [][]uint64) Fidelity {
	synth := m.Synthesize()
	rRds := LineReuseDistances(flattenAddrs(real), statLineBits)
	sRds := LineReuseDistances(flattenAddrs(synth), statLineBits)
	maxd := 0.0
	for _, b := range statCacheSizesB {
		c := b >> statLineBits
		if c < 1 {
			c = 1
		}
		if d := absF(MissRate(rRds, c) - MissRate(sRds, c)); d > maxd {
			maxd = d
		}
	}
	tr := MeanSectorsPerLoad(real, statSectorBits)
	ts := MeanSectorsPerLoad(synth, statSectorBits)
	txn := 0.0
	if tr > 0 {
		txn = absF(tr-ts) / tr
	}
	raw := 0
	for _, al := range real {
		raw += len(al)
	}
	raw = raw * bitsForRange(m.Range) / 8
	return Fidelity{MaxMissDelta: maxd, TxnRelErr: txn, ModelBytes: len(m.Bytes()), RawBytes: raw}
}

// FitAndGate fits the simplest statistical model that reproduces the stream's
// cache behavior within tol (both the max miss-rate delta and the transaction
// relative error), escalating uniform -> frequency-histogram. ok=false means no
// lossy model qualified — keep the lossless residual.
func FitAndGate(insts [][]uint64, tol float64, seed uint64) (m StatModel, f Fidelity, ok bool) {
	for _, withHist := range []bool{false, true} {
		m = FitStatModel(insts, withHist, seed)
		f = m.Validate(insts)
		if f.MaxMissDelta <= tol && f.TxnRelErr <= tol {
			return m, f, true
		}
	}
	return m, f, false
}

// ── serialization ─────────────────────────────────────────────────────────────

// Bytes serializes the model (the on-disk tier-7 residual for one gather PC).
func (m StatModel) Bytes() []byte {
	var b []byte
	b = putUvarint(b, m.Base)
	b = putUvarint(b, m.Elem)
	b = putUvarint(b, m.Range)
	b = putUvarint(b, m.Seed)
	ks := sortedIntKeys(m.LaneHist)
	b = putUvarint(b, uint64(len(ks)))
	for _, k := range ks {
		b = putUvarint(b, uint64(k))
		b = putUvarint(b, uint64(m.LaneHist[k]))
	}
	if m.Freq == nil {
		b = putUvarint(b, 0)
	} else {
		b = putUvarint(b, uint64(len(m.Freq)))
		for _, fr := range m.Freq {
			b = putUvarint(b, uint64(fr))
		}
	}
	return b
}

// DecodeStatModel is the inverse of Bytes.
func DecodeStatModel(data []byte) (StatModel, error) {
	c := &cursor{b: data}
	var m StatModel
	var err error
	if m.Base, err = c.uvarint(); err != nil {
		return m, err
	}
	if m.Elem, err = c.uvarint(); err != nil {
		return m, err
	}
	if m.Range, err = c.uvarint(); err != nil {
		return m, err
	}
	if m.Seed, err = c.uvarint(); err != nil {
		return m, err
	}
	nk, err := c.uvarint()
	if err != nil {
		return m, err
	}
	m.LaneHist = make(map[int]int, nk)
	for i := uint64(0); i < nk; i++ {
		k, err := c.uvarint()
		if err != nil {
			return m, err
		}
		v, err := c.uvarint()
		if err != nil {
			return m, err
		}
		m.LaneHist[int(k)] = int(v)
		m.NWarps += int(v)
	}
	nf, err := c.uvarint()
	if err != nil {
		return m, err
	}
	if nf > 0 {
		m.Freq = make([]uint32, nf)
		for i := range m.Freq {
			v, err := c.uvarint()
			if err != nil {
				return m, err
			}
			m.Freq[i] = uint32(v)
		}
	}
	return m, nil
}

// ── small helpers ─────────────────────────────────────────────────────────────

func gcdU(a, b uint64) uint64 {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

func absF(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func bitsForRange(r uint64) int {
	if r <= 1 {
		return 1
	}
	r--
	n := 0
	for r > 0 {
		n++
		r >>= 1
	}
	return n
}

func sortedIntKeys(m map[int]int) []int {
	ks := make([]int, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	sortInts(ks)
	return ks
}

// upperBound returns the smallest index i with cum[i] > v (cum is ascending).
func upperBound(cum []uint32, v uint32) int {
	lo, hi := 0, len(cum)
	for lo < hi {
		mid := (lo + hi) / 2
		if cum[mid] <= v {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo >= len(cum) {
		lo = len(cum) - 1
	}
	return lo
}
