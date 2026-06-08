// Command ptrace converts Accel-Sim traces to the .ptrace format and back.
//
//	ptrace convert <accelsim-traces-dir> <out.ptrace>
//	ptrace expand  <in.ptrace> <out-traces-dir>
//	ptrace verify  <accelsim-traces-dir>     # lossless round-trip + size report
//	ptrace stat    <in.ptrace>               # size / dedup summary
//	ptrace synth   <accelsim-traces-dir>     # sampled collect, report wall-time saved
//	ptrace synth   <traces-dir> --grid GxHxD <out-traces-dir>  # extrapolate to a new size
//	ptrace statfit <accelsim-traces-dir>     # lossy tier-7 fit of data-dependent gathers
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/sarchlab/gpu_benchmarks/tracer/ptrace"
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	var err error
	switch os.Args[1] {
	case "convert":
		need(4)
		err = ptrace.Convert(os.Args[2], os.Args[3])
	case "expand":
		need(4)
		err = ptrace.Expand(os.Args[2], os.Args[3])
	case "verify":
		need(3)
		err = verify(os.Args[2])
	case "stat":
		need(3)
		err = stat(os.Args[2])
	case "holdout":
		need(3)
		err = holdout(os.Args[2])
	case "synth":
		need(3)
		err = synth(os.Args[2:])
	case "statfit":
		need(3)
		err = statfit(os.Args[2])
	case "scale":
		need(5)
		err = scaleVerify(os.Args[2:])
	default:
		usage()
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func need(n int) {
	if len(os.Args) < n {
		usage()
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: ptrace convert|expand|verify|stat|holdout|synth|statfit|scale ...")
	os.Exit(2)
}

func dirSize(dir string) int64 {
	var total int64
	filepath.Walk(dir, func(_ string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() {
			total += info.Size()
		}
		return nil
	})
	return total
}

// verify converts a traces dir to .ptrace, expands it, and checks that every
// kernel's decoded .traceg content is byte-identical to the original.
func verify(tracesDir string) error {
	tmp, err := os.MkdirTemp("", "ptrace-verify-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmp)
	pdir := filepath.Join(tmp, "out.ptrace")
	if err := ptrace.Convert(tracesDir, pdir); err != nil {
		return err
	}

	m, err := ptrace.ReadManifest(pdir)
	if err != nil {
		return err
	}
	isaCache := map[string]ptrace.ISATable{}
	nPass, nFail := 0, 0
	for _, inv := range m.Invocations {
		orig, err := os.ReadFile(filepath.Join(tracesDir, inv.TraceFile))
		if err != nil {
			return err
		}
		oh, ob, err := ptrace.DecodeContainer(orig)
		if err != nil {
			return err
		}
		want := oh + strings.Join(ob, "")
		got, err := ptrace.ExpandTraceg(pdir, m, inv, isaCache)
		if err != nil {
			return err
		}
		if got == want {
			nPass++
		} else {
			nFail++
			fmt.Printf("  FAIL %s (len got=%d want=%d, first diff at %d)\n",
				inv.TraceFile, len(got), len(want), firstDiff(got, want))
		}
	}

	var memInsts, modeledInsts, memPCs, modeledPCs int
	for _, inv := range m.Invocations {
		memInsts += inv.Cov.MemInsts
		modeledInsts += inv.Cov.ModeledInsts
		memPCs += inv.Cov.MemPCs
		modeledPCs += inv.Cov.ModeledPCs
	}
	orig := dirSize(tracesDir)
	pt := dirSize(pdir)
	fmt.Printf("round-trip: %d/%d kernels byte-identical\n", nPass, nPass+nFail)
	if memInsts > 0 {
		fmt.Printf("modeled: %.1f%% of mem instructions (%d/%d PCs)\n",
			100*float64(modeledInsts)/float64(memInsts), modeledPCs, memPCs)
	}
	fmt.Printf("size: accelsim=%s  ptrace=%s  ratio=%.2fx (%.1f%% of original)\n",
		human(orig), human(pt), float64(orig)/float64(pt), 100*float64(pt)/float64(orig))
	if nFail > 0 {
		return fmt.Errorf("%d kernels failed round-trip", nFail)
	}
	return nil
}

func stat(pdir string) error {
	m, err := ptrace.ReadManifest(pdir)
	if err != nil {
		return err
	}
	isaSet := map[string]bool{}
	for _, inv := range m.Invocations {
		isaSet[inv.ISA] = true
	}
	fmt.Printf("invocations: %d\n", len(m.Invocations))
	fmt.Printf("distinct ISA tables: %d (dedup across relaunches)\n", len(isaSet))
	fmt.Printf("contexts: %d\n", len(m.Contexts))
	fmt.Printf("total size: %s\n", human(dirSize(pdir)))
	return nil
}

// holdout converts a traces dir to .ptrace, then for each invocation fits
// recipes on a corner of the grid and synthesizes the held-out blocks, reporting
// how much of the trace could be reconstructed from the sampled blocks.
func holdout(tracesDir string) error {
	tmp, err := os.MkdirTemp("", "ptrace-holdout-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmp)
	pdir := filepath.Join(tmp, "out.ptrace")
	if err := ptrace.Convert(tracesDir, pdir); err != nil {
		return err
	}
	m, err := ptrace.ReadManifest(pdir)
	if err != nil {
		return err
	}
	isaCache := map[string]ptrace.ISATable{}
	var totTest, totInst, totMatch, totBlocksOK, totTestBlocks, totTrain int
	for _, inv := range m.Invocations {
		blocks, isa, err := ptrace.DecodeInvocationBlocks(pdir, inv, isaCache)
		if err != nil {
			return err
		}
		// train = lower-corner quadrant of the grid (≈ a few blocks)
		bxs, bys := axisValues(blocks)
		tcx := halfSet(bxs)
		tcy := halfSet(bys)
		train := func(b ptrace.Block) bool { return tcx[b.X] && tcy[b.Y] }
		r := ptrace.Holdout(blocks, isa, train)
		totTrain += r.TrainBlocks
		totTestBlocks += r.TestBlocks
		totBlocksOK += r.BlocksOK
		totInst += r.TestInsts
		totMatch += r.MatchInsts
		totTest++
	}
	pct := 0.0
	if totInst > 0 {
		pct = 100 * float64(totMatch) / float64(totInst)
	}
	fmt.Printf("trained on lower-quadrant blocks; synthesized the rest:\n")
	fmt.Printf("  held-out blocks fully reconstructed: %d/%d\n", totBlocksOK, totTestBlocks)
	fmt.Printf("  held-out mem instructions byte-exact: %d/%d = %.1f%%\n", totMatch, totInst, pct)
	return nil
}

// synth runs the P3 sampled-collection path. Without --grid it converts the
// traces with corner sampling, verifies the reconstruction is byte-identical, and
// reports the recorded-block fraction (the collection-wall-time proxy) and size.
// With `--grid GxHxD <out-dir>` it instead extrapolates the recipes to a new,
// larger grid and writes the synthesized (unverified) traces there.
func synth(args []string) error {
	tracesDir := args[0]
	rest := args[1:]
	if len(rest) >= 1 && rest[0] == "--grid" {
		if len(rest) < 3 {
			return fmt.Errorf("usage: ptrace synth <traces-dir> --grid GxHxD <out-traces-dir>")
		}
		gx, gy, gz, err := parseGrid(rest[1])
		if err != nil {
			return err
		}
		return synthGrid(tracesDir, rest[2], gx, gy, gz)
	}
	return synthSampled(tracesDir)
}

func synthSampled(tracesDir string) error {
	tmp, err := os.MkdirTemp("", "ptrace-synth-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmp)
	pdir := filepath.Join(tmp, "out.ptrace")
	if err := ptrace.ConvertSampled(tracesDir, pdir); err != nil {
		return err
	}
	m, err := ptrace.ReadManifest(pdir)
	if err != nil {
		return err
	}
	isaCache := map[string]ptrace.ISATable{}
	nPass, nFail := 0, 0
	var total, recorded int
	for _, inv := range m.Invocations {
		orig, err := os.ReadFile(filepath.Join(tracesDir, inv.TraceFile))
		if err != nil {
			return err
		}
		oh, ob, err := ptrace.DecodeContainer(orig)
		if err != nil {
			return err
		}
		want := oh + strings.Join(ob, "")
		got, err := ptrace.ExpandTraceg(pdir, m, inv, isaCache)
		if err != nil {
			return err
		}
		if got == want {
			nPass++
		} else {
			nFail++
			fmt.Printf("  FAIL %s (first diff at %d)\n", inv.TraceFile, firstDiff(got, want))
		}
		total += inv.Cov.TotalBlocks
		recorded += inv.Cov.RecordedBlocks
	}
	fmt.Printf("sampled reconstruction: %d/%d kernels byte-identical\n", nPass, nPass+nFail)
	if total > 0 {
		synthFrac := 100 * float64(total-recorded) / float64(total)
		fmt.Printf("instrumented blocks: %d/%d (%.1f%% synthesized, not traced)\n",
			recorded, total, synthFrac)
		fmt.Printf("projected collection wall-time: %.1f%% of full (~%.2fx speedup)\n",
			100*float64(recorded)/float64(total), float64(total)/float64(max1(recorded)))
	}
	orig := dirSize(tracesDir)
	pt := dirSize(pdir)
	fmt.Printf("size: accelsim=%s  ptrace(sampled)=%s  ratio=%.2fx\n",
		human(orig), human(pt), float64(orig)/float64(pt))
	if nFail > 0 {
		return fmt.Errorf("%d kernels failed round-trip", nFail)
	}
	return nil
}

func synthGrid(tracesDir, outDir string, gx, gy, gz int) error {
	tmp, err := os.MkdirTemp("", "ptrace-grid-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmp)
	pdir := filepath.Join(tmp, "in.ptrace")
	if err := ptrace.Convert(tracesDir, pdir); err != nil {
		return err
	}
	m, err := ptrace.ReadManifest(pdir)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}
	isaCache := map[string]ptrace.ISATable{}
	fmt.Printf("extrapolating to grid %dx%dx%d (unverified — no ground truth at this size):\n", gx, gy, gz)
	for _, inv := range m.Invocations {
		blocks, isa, err := ptrace.DecodeInvocationBlocks(pdir, inv, isaCache)
		if err != nil {
			return err
		}
		content, nSynth, nFail, err := ptrace.ExtrapolateGrid(blocks, isa, inv.Header, gx, gy, gz)
		if err != nil {
			return err
		}
		comp, err := ptrace.EncodeContainer(content)
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(outDir, inv.TraceFile), comp, 0o644); err != nil {
			return err
		}
		status := "fully synthesized"
		if nFail > 0 {
			status = fmt.Sprintf("%d/%d blocks NOT synthesizable (data-dependent)", nFail, nSynth+nFail)
		}
		fmt.Printf("  %s: %d blocks — %s\n", inv.TraceFile, nSynth, status)
	}
	return nil
}

func parseGrid(s string) (gx, gy, gz int, err error) {
	parts := strings.Split(strings.ToLower(s), "x")
	if len(parts) < 1 || len(parts) > 3 {
		return 0, 0, 0, fmt.Errorf("bad grid %q, want GxHxD", s)
	}
	dims := []int{1, 1, 1}
	for i, p := range parts {
		v, e := strconv.Atoi(p)
		if e != nil || v < 1 {
			return 0, 0, 0, fmt.Errorf("bad grid dim %q", p)
		}
		dims[i] = v
	}
	return dims[0], dims[1], dims[2], nil
}

func max1(n int) int {
	if n < 1 {
		return 1
	}
	return n
}

// statfit fits the lossy tier-7 statistical model to each data-dependent gather,
// validates it against cache-behavior equivalence (LRU miss-rate curve + per-warp
// transaction count), and reports the per-gather compression and fidelity. This
// is the encoder for irregular accesses that resist lossless compression (SpMV
// gather, etc.): it never affects functional/byte-exact paths.
func statfit(tracesDir string) error {
	const tol = 0.06 // 6% max miss-rate delta and transaction error
	tmp, err := os.MkdirTemp("", "ptrace-statfit-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmp)
	pdir := filepath.Join(tmp, "out.ptrace")
	if err := ptrace.Convert(tracesDir, pdir); err != nil {
		return err
	}
	m, err := ptrace.ReadManifest(pdir)
	if err != nil {
		return err
	}
	isaCache := map[string]ptrace.ISATable{}
	var rawTotal, modelTotal, accepted, gathers int
	fmt.Printf("%-8s %9s %9s %9s %8s %8s  %s\n", "pc", "accesses", "raw", "model", "missΔ", "txnErr", "model")
	for _, inv := range m.Invocations {
		blocks, isa, err := ptrace.DecodeInvocationBlocks(pdir, inv, isaCache)
		if err != nil {
			return err
		}
		streams := ptrace.GatherStreams(blocks, isa)
		for _, pc := range sortedPCs(streams) {
			insts := streams[pc]
			gathers++
			model, fid, ok := ptrace.FitAndGate(insts, tol, 1)
			rawTotal += fid.RawBytes
			kind := "uniform"
			if model.Freq != nil {
				kind = "histogram"
			}
			status := kind
			if ok {
				accepted++
				modelTotal += fid.ModelBytes
			} else {
				modelTotal += fid.RawBytes // rejected -> stays lossless
				status = "REJECT (lossless): " + kind
			}
			fmt.Printf("0x%04x %9d %8dB %8dB %7.1f%% %7.1f%%  %s\n",
				pc, countAddrs(insts), fid.RawBytes, fid.ModelBytes,
				100*fid.MaxMissDelta, 100*fid.TxnRelErr, status)
		}
	}
	if gathers == 0 {
		fmt.Println("no data-dependent gathers found")
		return nil
	}
	ratio := 0.0
	if modelTotal > 0 {
		ratio = float64(rawTotal) / float64(modelTotal)
	}
	fmt.Printf("\n%d/%d gathers accepted (lossy, cache-equivalent within %.0f%%)\n", accepted, gathers, 100*tol)
	fmt.Printf("gather bytes: lossless ~%s -> tier-7 ~%s  (%.0fx on the gather)\n",
		human(int64(rawTotal)), human(int64(modelTotal)), ratio)
	return nil
}

// scaleVerify generates the trace of a larger per-thread-work size from a small
// sample by loop-replay, and verifies it byte-for-byte against a real trace of
// the target size — proving "one small run -> many sizes" for regular kernels.
//
//	ptrace scale <sample-traces> <fromScaling> <toScaling> <truth-traces>
func scaleVerify(args []string) error {
	sampleDir, truthDir := args[0], args[3]
	from, err := strconv.Atoi(args[1])
	if err != nil {
		return fmt.Errorf("bad fromScaling %q", args[1])
	}
	to, err := strconv.Atoi(args[2])
	if err != nil {
		return fmt.Errorf("bad toScaling %q", args[2])
	}
	sfs, tfs := traceFiles(sampleDir), traceFiles(truthDir)
	if len(sfs) == 0 || len(tfs) == 0 {
		return fmt.Errorf("no .trace.zst found (sample=%d truth=%d)", len(sfs), len(tfs))
	}
	nPass, nFail, notScalable := 0, 0, 0
	var sampleBytes, truthBytes int64
	for i, sf := range sfs {
		sraw, _ := os.ReadFile(sf)
		sampleBytes += int64(len(sraw))
		sh, sb, err := ptrace.DecodeContainer(sraw)
		if err != nil {
			return err
		}
		isa := map[int]ptrace.StaticInst{}
		hdr, blocks, err := ptrace.ParseTraceg(sh+strings.Join(sb, ""), isa)
		if err != nil {
			return err
		}
		scaled, ok := ptrace.ScaleBlocks(blocks, isa, to, from)
		if !ok {
			notScalable++
			fmt.Printf("  %s: not regularly scalable (data-dependent loop / no loop)\n", filepath.Base(sf))
			continue
		}
		gen, err := ptrace.EmitTraceg(hdr, scaled, isa)
		if err != nil {
			return err
		}
		if i >= len(tfs) {
			break
		}
		traw, _ := os.ReadFile(tfs[i])
		truthBytes += int64(len(traw))
		th, tb, err := ptrace.DecodeContainer(traw)
		if err != nil {
			return err
		}
		want := th + strings.Join(tb, "")
		// Compare per thread block, ignoring the run-specific header (shmem/local
		// base addresses are assigned per process, not per problem size). Remaining
		// token differences are classified: anything that is not an absolute memory
		// address (also run-specific, e.g. a cudaMalloc'd output pointer) is a real
		// structural error.
		gblk, wblk := splitTBs(gen), splitTBs(want)
		blkSame, tokDiff, addrDiff := compareTBs(gblk, wblk)
		if len(gblk) == len(wblk) && tokDiff == addrDiff { // every diff is a run-specific address
			nPass++
			fmt.Printf("  %s: %d/%d blocks identical; %d run-specific address tokens differ (expected)\n",
				filepath.Base(sf), blkSame, len(wblk), addrDiff)
		} else {
			nFail++
			fmt.Printf("  %s: STRUCTURAL MISMATCH — %d/%d blocks identical, %d non-address token diffs\n",
				filepath.Base(sf), blkSame, len(wblk), tokDiff-addrDiff)
		}
	}
	fmt.Printf("scaled %d->%d : %d/%d kernels reproduced (structure byte-exact modulo run-specific addresses)",
		from, to, nPass, nPass+nFail)
	if notScalable > 0 {
		fmt.Printf(" (%d not scalable)", notScalable)
	}
	fmt.Printf("\nsample trace: %s   (target real trace: %s — never collected; %.0fx less data + compute)\n",
		human(sampleBytes), human(truthBytes), float64(truthBytes)/float64(maxI64(sampleBytes, 1)))
	if nFail > 0 {
		return fmt.Errorf("%d kernels failed structural scaling verification", nFail)
	}
	return nil
}

// splitTBs splits a .traceg into its per-thread-block segments (dropping the
// run-specific header before the first block).
func splitTBs(s string) []string {
	parts := strings.Split(s, "\n#BEGIN_TB\n")
	if len(parts) <= 1 {
		return nil
	}
	return parts[1:]
}

// compareTBs compares thread blocks token-by-token, returning the count of
// byte-identical blocks, the total differing tokens, and how many of those are
// absolute-address-shaped (run-specific, expected to differ).
func compareTBs(gen, want []string) (blkIdentical, tokDiff, addrDiff int) {
	n := len(gen)
	if len(want) < n {
		n = len(want)
	}
	for i := 0; i < n; i++ {
		if gen[i] == want[i] {
			blkIdentical++
			continue
		}
		gt, wt := strings.Fields(gen[i]), strings.Fields(want[i])
		m := len(gt)
		if len(wt) < m {
			m = len(wt)
		}
		for j := 0; j < m; j++ {
			if gt[j] != wt[j] {
				tokDiff++
				if isAddrToken(gt[j]) && isAddrToken(wt[j]) {
					addrDiff++
				}
			}
		}
		tokDiff += abs(len(gt) - len(wt)) // length mismatch counts as structural
	}
	return
}

// isAddrToken reports whether a token looks like an absolute memory address
// (hex, wide), i.e. a run-specific pointer rather than structural content.
func isAddrToken(t string) bool {
	t = strings.TrimPrefix(t, "0x")
	if len(t) < 6 || len(t) > 16 {
		return false
	}
	for _, c := range t {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			return false
		}
	}
	return true
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
func maxI64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func traceFiles(dir string) []string {
	var fs []string
	filepath.Walk(dir, func(p string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() && strings.HasSuffix(p, ".trace.zst") {
			fs = append(fs, p)
		}
		return nil
	})
	sortStrs(fs)
	return fs
}

func sortStrs(a []string) {
	for i := 1; i < len(a); i++ {
		for j := i; j > 0 && a[j-1] > a[j]; j-- {
			a[j-1], a[j] = a[j], a[j-1]
		}
	}
}

func sortedPCs(m map[int][][]uint64) []int {
	ks := make([]int, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	for i := 1; i < len(ks); i++ {
		for j := i; j > 0 && ks[j-1] > ks[j]; j-- {
			ks[j-1], ks[j] = ks[j], ks[j-1]
		}
	}
	return ks
}

func countAddrs(insts [][]uint64) int {
	n := 0
	for _, al := range insts {
		n += len(al)
	}
	return n
}

func axisValues(blocks []ptrace.Block) ([]int, []int) {
	xs, ys := map[int]bool{}, map[int]bool{}
	for _, b := range blocks {
		xs[b.X] = true
		ys[b.Y] = true
	}
	return keys(xs), keys(ys)
}

func keys(m map[int]bool) []int {
	out := make([]int, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j-1] > out[j]; j-- {
			out[j-1], out[j] = out[j], out[j-1]
		}
	}
	return out
}

func halfSet(vals []int) map[int]bool {
	s := map[int]bool{}
	n := len(vals) / 2
	if n == 0 {
		n = 1
	}
	for _, v := range vals[:n] {
		s[v] = true
	}
	return s
}

func firstDiff(a, b string) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

func human(n int64) string {
	const u = 1024
	if n < u {
		return fmt.Sprintf("%dB", n)
	}
	div, exp := int64(u), 0
	for x := n / u; x >= u; x /= u {
		div *= u
		exp++
	}
	return fmt.Sprintf("%.1f%cB", float64(n)/float64(div), "KMGT"[exp])
}
