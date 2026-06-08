package ptrace

// cache.go is a small LRU / stack-distance cache model. It is the *gate* for the
// lossy tier-7 statistical encoder (tier7.go): a regenerated address stream is
// accepted only if it reproduces the original's cache behavior — the LRU
// miss-rate curve across cache sizes (temporal locality) and the per-warp
// memory-transaction count (coalescing) — within tolerance. The metric we
// preserve is the metric we validate.

// fenwick is a Fenwick/BIT tree for O(n log n) stack-distance computation.
type fenwick struct {
	n int
	t []int
}

func newFenwick(n int) *fenwick { return &fenwick{n: n, t: make([]int, n+1)} }
func (f *fenwick) add(i, v int) {
	for i++; i <= f.n; i += i & -i {
		f.t[i] += v
	}
}
func (f *fenwick) sum(i int) int {
	s := 0
	for i++; i > 0; i -= i & -i {
		s += f.t[i]
	}
	return s
}
func (f *fenwick) rangeSum(l, r int) int {
	if l > r {
		return 0
	}
	if l == 0 {
		return f.sum(r)
	}
	return f.sum(r) - f.sum(l-1)
}

// LineReuseDistances returns, for each access in seq, the number of distinct cache
// lines touched since that line's previous access (the LRU stack distance), or -1
// for a first (compulsory) touch. lineBits = log2(line size in bytes). For a
// fully-associative LRU cache of C lines, an access is a hit iff 0 <= distance < C.
func LineReuseDistances(seq []uint64, lineBits uint) []int {
	rds := make([]int, len(seq))
	last := make(map[uint64]int, len(seq))
	f := newFenwick(len(seq))
	for t, a := range seq {
		L := a >> lineBits
		if p, ok := last[L]; ok {
			rds[t] = f.rangeSum(p+1, t-1)
			f.add(p, -1)
		} else {
			rds[t] = -1
		}
		f.add(t, 1)
		last[L] = t
	}
	return rds
}

// MissRate of a fully-associative LRU cache of cacheLines lines, from precomputed
// stack distances: the fraction of accesses with distance >= cacheLines (capacity
// miss) or distance < 0 (compulsory miss).
func MissRate(rds []int, cacheLines int) float64 {
	if len(rds) == 0 {
		return 0
	}
	m := 0
	for _, d := range rds {
		if d < 0 || d >= cacheLines {
			m++
		}
	}
	return float64(m) / float64(len(rds))
}

// MeanSectorsPerLoad is the average number of distinct sectors a warp-load's
// active lanes touch — i.e. memory transactions per load (a fully coalesced load
// is 1; an uncoalesced gather approaches the active-lane count). sectorBits =
// log2(sector size in bytes).
func MeanSectorsPerLoad(insts [][]uint64, sectorBits uint) float64 {
	if len(insts) == 0 {
		return 0
	}
	tot := 0
	for _, al := range insts {
		s := map[uint64]bool{}
		for _, a := range al {
			s[a>>sectorBits] = true
		}
		tot += len(s)
	}
	return float64(tot) / float64(len(insts))
}

func flattenAddrs(insts [][]uint64) []uint64 {
	n := 0
	for _, al := range insts {
		n += len(al)
	}
	seq := make([]uint64, 0, n)
	for _, al := range insts {
		seq = append(seq, al...)
	}
	return seq
}
