// Package sass is an offline SASS (cubin disassembly) analyzer. It classifies,
// per kernel, what a sampling tracer can synthesize from indices versus what it
// must capture at run time:
//
//   - each memory instruction's address tier (constant / affine-in-index /
//     indirect-through-a-loaded-value), recovered by backward-slicing the address
//     register to its provenance (builtin index, kernel arg, or a memory load);
//   - each branch's divergence class (uniform / index-determined / data-dependent),
//     so "what warp needs to take" is computed from indices where possible and
//     sampled only where it depends on data; and
//   - whether the kernel is deterministic across launches (no device-side PRNG,
//     clock, or atomics), which decides whether relaunches can be deduped.
//
// It is a conservative classifier, not a prover: an address or predicate whose
// provenance cannot be resolved (optimized datapaths, spills) is classified down
// to the safe side (data-dependent / must-capture). The dynamic verify-or-fallback
// gate remains the correctness backstop.
//
// The slicing is a linear last-writer walk, which is exact for the straight-line
// address arithmetic these kernels emit and approximate across control flow.
package sass

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// ── instruction model ─────────────────────────────────────────────────────────

// Inst is one decoded SASS instruction.
type Inst struct {
	Addr   uint64
	Pred   string // guard predicate: "", "P0", "!P0"
	Op     string // base mnemonic, e.g. IMAD, LDG, ISETP, BRA
	Mods   []string
	Ops    []string // operands, in textual order (dest first for writes)
	Target uint64   // branch target (absolute), if any
	HasTgt bool
	Raw    string
}

// Dst returns the written register/predicate (first operand), or "" if the
// instruction writes nothing addressable.
func (in Inst) Dst() string {
	switch in.Op {
	case "EXIT", "BRA", "BRX", "JMP", "RET", "NOP", "BSSY", "BSYNC", "BAR",
		"STG", "ST", "STS", "STL", "RED", "MEMBAR", "WARPSYNC", "CALL":
		return ""
	}
	if len(in.Ops) == 0 {
		return ""
	}
	return regName(in.Ops[0])
}

var (
	reAddr  = regexp.MustCompile(`/\*([0-9a-fA-F]+)\*/`)
	reReg   = regexp.MustCompile(`\bU?R\d+\b|\bP\d+\b`)
	reConst = regexp.MustCompile(`c\[0x[0-9a-fA-F]+\]\[0x([0-9a-fA-F]+)\]`)
	reHex   = regexp.MustCompile(`\b0x[0-9a-fA-F]+\b`)
)

// regName returns the leading register token of an operand (strips .reuse, .64,
// negation, brackets), or "" if the operand is not a plain register.
func regName(op string) string {
	op = strings.TrimSpace(op)
	op = strings.TrimPrefix(op, "-")
	op = strings.TrimPrefix(op, "|")
	m := reReg.FindString(op)
	return m
}

// addrReg returns the register inside the final [...] of a memory operand, e.g.
// "desc[UR4][R4.64+0x4]" -> "R4". That register holds the byte address.
func addrReg(op string) string {
	l := strings.LastIndex(op, "[")
	r := strings.LastIndex(op, "]")
	if l < 0 || r < 0 || r < l {
		return ""
	}
	inner := op[l+1 : r]
	// the address register is the first R-register in the bracket
	for _, tok := range reReg.FindAllString(inner, -1) {
		if strings.HasPrefix(tok, "R") && !strings.HasPrefix(tok, "RZ") {
			return tok
		}
	}
	return ""
}

// ParseKernels splits cuobjdump -sass output into kernels and parses each one's
// instruction list. Kernel boundaries are "Function : <mangled>" lines.
func ParseKernels(sass string) []*Kernel {
	var kernels []*Kernel
	var cur *Kernel
	for _, line := range strings.Split(sass, "\n") {
		if i := strings.Index(line, "Function :"); i >= 0 {
			name := strings.TrimSpace(line[i+len("Function :"):])
			cur = &Kernel{Name: name}
			kernels = append(kernels, cur)
			continue
		}
		if cur == nil || !reAddr.MatchString(line) {
			continue
		}
		if in, ok := parseLine(line); ok {
			cur.Insts = append(cur.Insts, in)
		}
	}
	return kernels
}

func parseLine(line string) (Inst, bool) {
	m := reAddr.FindStringSubmatch(line)
	if m == nil {
		return Inst{}, false
	}
	addr, _ := strconv.ParseUint(m[1], 16, 64)

	// body = text between the address comment and the trailing encoding comment.
	body := line[reAddr.FindStringIndex(line)[1]:]
	if i := strings.Index(body, "/*"); i >= 0 { // strip trailing /* 0x... */
		body = body[:i]
	}
	body = strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(body), ";"))
	if body == "" {
		return Inst{}, false
	}

	in := Inst{Addr: addr, Raw: body}
	fields := strings.Fields(body)
	idx := 0
	if strings.HasPrefix(fields[0], "@") {
		in.Pred = strings.TrimPrefix(fields[0], "@")
		idx = 1
	}
	if idx >= len(fields) {
		return Inst{}, false
	}
	opTok := fields[idx]
	parts := strings.Split(opTok, ".")
	in.Op = parts[0]
	in.Mods = parts[1:]

	rest := strings.TrimSpace(strings.Join(fields[idx+1:], " "))
	if rest != "" {
		for _, o := range strings.Split(rest, ",") {
			in.Ops = append(in.Ops, strings.TrimSpace(o))
		}
	}
	if in.Op == "BRA" || in.Op == "BRX" || in.Op == "JMP" {
		if t := reHex.FindString(rest); t != "" {
			v, err := strconv.ParseUint(strings.TrimPrefix(t, "0x"), 16, 64)
			if err == nil {
				in.Target, in.HasTgt = v, true
			}
		}
	}
	return in, true
}

// ── provenance / dataflow ─────────────────────────────────────────────────────

// Prov is where a value ultimately comes from.
type Prov int

const (
	ProvUnknown Prov = iota
	ProvConst        // immediate / RZ
	ProvArg          // kernel parameter (constant bank)
	ProvIndex        // builtin: ctaid / tid / laneid
	ProvLoaded       // value read from memory (data-dependent)
	ProvClock        // clock / SR_CLOCK (nondeterministic)
)

func (p Prov) String() string {
	switch p {
	case ProvConst:
		return "const"
	case ProvArg:
		return "arg"
	case ProvIndex:
		return "index"
	case ProvLoaded:
		return "loaded"
	case ProvClock:
		return "clock"
	}
	return "unknown"
}

// join combines source provenances for an ALU result. "Stronger" = more varying /
// less synthesizable: loaded/clock dominate index, which dominates arg/const.
func join(a, b Prov) Prov {
	rank := func(p Prov) int {
		switch p {
		case ProvClock:
			return 5
		case ProvLoaded:
			return 4
		case ProvIndex:
			return 3
		case ProvArg:
			return 2
		case ProvConst:
			return 1
		}
		return 0
	}
	if rank(a) >= rank(b) {
		return a
	}
	return b
}

// Kernel holds the parsed body and analysis results.
type Kernel struct {
	Name  string
	Insts []Inst
}

// lastWriter returns the index of the most recent instruction before `before`
// that writes reg, or -1. Linear scan (path-insensitive).
func (k *Kernel) lastWriter(before int, reg string) int {
	for j := before - 1; j >= 0; j-- {
		if k.Insts[j].Dst() == reg {
			return j
		}
	}
	return -1
}

// classifyReg backward-slices a register's provenance from instruction `at`.
// argOff collects the constant-bank offsets (kernel-arg bases) it passes through.
func (k *Kernel) classifyReg(at int, reg string, depth int, argOff map[uint64]bool) Prov {
	if depth > 64 || reg == "" || reg == "RZ" || reg == "URZ" || reg == "PT" {
		return ProvConst
	}
	w := k.lastWriter(at, reg)
	if w < 0 {
		return ProvUnknown // live-in; conservatively unknown
	}
	in := k.Insts[w]
	switch in.Op {
	case "S2R":
		src := strings.Join(in.Ops, " ")
		switch {
		case strings.Contains(src, "SR_CLOCK"):
			return ProvClock
		case strings.Contains(src, "CTAID"), strings.Contains(src, "TID"),
			strings.Contains(src, "LANEID"), strings.Contains(src, "CTAID"):
			return ProvIndex
		}
		return ProvIndex
	case "LDC", "ULDC", "MOV32I":
		if m := reConst.FindStringSubmatch(in.Raw); m != nil {
			off, _ := strconv.ParseUint(m[1], 16, 64)
			argOff[off] = true
			return ProvArg
		}
		// MOV32I / immediate
		return ProvConst
	case "LDG", "LD", "LDS", "LDL", "LDGSTS", "LDSM":
		return ProvLoaded
	}
	// ALU / MOV: combine source-register provenances.
	res := ProvConst
	srcs := in.Ops
	if len(srcs) > 0 {
		srcs = srcs[1:] // skip dest
	}
	any := false
	for _, o := range srcs {
		r := regName(o)
		if r == "" {
			if reConst.MatchString(o) { // direct const-bank source (e.g. IMAD ..., c[0x0][..])
				if m := reConst.FindStringSubmatch(o); m != nil {
					off, _ := strconv.ParseUint(m[1], 16, 64)
					argOff[off] = true
				}
				res = join(res, ProvArg)
				any = true
			}
			continue
		}
		// A source equal to the dest is a read-modify-write accumulator (e.g.
		// IMAD.WIDE R4, R0, 0x4, R4): lastWriter looks strictly before w, so
		// recursing finds the prior definition, not this instruction.
		res = join(res, k.classifyReg(w, r, depth+1, argOff))
		any = true
	}
	if !any {
		return ProvConst
	}
	return res
}

// nonlinearOps are register operations that take an index/arg out of the affine
// regime: bitwise logic and bit permutes are the signature of PRNGs (xorshift /
// Philox) and hashes. A scaling shift folds into an affine coefficient, so plain
// SHL/SHF are not flagged; XOR/AND/OR and permutes are.
var nonlinearOps = map[string]bool{
	"LOP": true, "LOP3": true, "LOP32I": true,
	"PRMT": true, "BREV": true, "POPC": true, "FLO": true,
}

// sliceNonlinear reports whether the backward slice of reg (from instruction at)
// passes through a non-affine register op — i.e. the address is computed, not
// affine-in-index. Used to split tier 0–3 (affine) from tier 4 (PRNG/computed).
func (k *Kernel) sliceNonlinear(at int, reg string, depth int) bool {
	if depth > 64 || reg == "" || reg == "RZ" || reg == "URZ" || reg == "PT" {
		return false
	}
	w := k.lastWriter(at, reg)
	if w < 0 {
		return false
	}
	in := k.Insts[w]
	if nonlinearOps[in.Op] {
		return true
	}
	switch in.Op {
	case "S2R", "LDC", "ULDC", "MOV32I", "LDG", "LD", "LDS", "LDL", "LDGSTS", "LDSM":
		return false // leaves of the slice
	}
	for _, o := range in.Ops[1:] {
		if r := regName(o); r != "" && k.sliceNonlinear(w, r, depth+1) {
			return true
		}
	}
	return false
}

// ── analysis output ───────────────────────────────────────────────────────────

// MemAccess is the classification of one LDG/STG.
type MemAccess struct {
	Addr uint64
	Op   string
	Tier string   // "const" | "affine" | "indirect" | "unknown"
	Base Prov     // provenance of the address register
	Args []uint64 // constant-bank offsets (arg bases) the address depends on
}

// Branch is the classification of one guarded branch.
type Branch struct {
	Addr     uint64
	Pred     string
	Class    string // "uniform" | "index" | "data" | "unknown"
	BackEdge bool   // target < addr => loop
	Target   uint64
}

// Analysis is the per-kernel result.
type Analysis struct {
	Name          string
	Mem           []MemAccess
	Branches      []Branch
	Deterministic bool
	NonDetReasons []string
	DataDependent bool // any indirect address or data-dependent branch
	PRNGMixing    bool // loop-carried bitwise mixing (xorshift/Philox signature)
}

func argOffsets(m map[uint64]bool) []uint64 {
	out := make([]uint64, 0, len(m))
	for o := range m {
		out = append(out, o)
	}
	for i := 1; i < len(out); i++ { // insertion sort, small
		for j := i; j > 0 && out[j-1] > out[j]; j-- {
			out[j-1], out[j] = out[j], out[j-1]
		}
	}
	return out
}

// Analyze classifies a kernel's memory accesses, branches, and determinism.
func (k *Kernel) Analyze() Analysis {
	a := Analysis{Name: k.Name, Deterministic: true}

	for i, in := range k.Insts {
		switch {
		case isLoad(in.Op) || isStore(in.Op):
			areg := addrReg(memOperand(in))
			argOff := map[uint64]bool{}
			p := k.classifyReg(i, areg, 0, argOff)
			tier := "unknown"
			switch {
			case p == ProvLoaded:
				tier = "indirect"
			case k.sliceNonlinear(i, areg, 0):
				tier = "computed" // non-affine register math: PRNG/hash/bit-reverse (tier 4)
			case p == ProvIndex:
				tier = "affine"
			case p == ProvArg, p == ProvConst:
				tier = "const"
			}
			if tier == "indirect" {
				a.DataDependent = true
			}
			a.Mem = append(a.Mem, MemAccess{Addr: in.Addr, Op: in.Op + dot(in.Mods), Tier: tier, Base: p, Args: argOffsets(argOff)})

		case in.Pred != "" && (in.Op == "BRA" || in.Op == "EXIT" || in.Op == "BRX"):
			reg := strings.TrimPrefix(in.Pred, "!")
			argOff := map[uint64]bool{}
			p := k.predProv(i, reg, argOff)
			class := map[Prov]string{ProvConst: "uniform", ProvArg: "uniform", ProvIndex: "index", ProvLoaded: "data", ProvClock: "data", ProvUnknown: "unknown"}[p]
			be := in.HasTgt && in.Target <= in.Addr
			if class == "data" {
				a.DataDependent = true
			}
			a.Branches = append(a.Branches, Branch{Addr: in.Addr, Pred: in.Pred, Class: class, BackEdge: be, Target: in.Target})
		}

		// determinism scan
		switch {
		case strings.HasPrefix(in.Op, "ATOM"), in.Op == "RED", in.Op == "ATOMG", in.Op == "ATOMS":
			a.Deterministic = false
			a.NonDetReasons = appendUniq(a.NonDetReasons, "atomic ("+in.Op+")")
		case in.Op == "S2R" && strings.Contains(strings.Join(in.Ops, " "), "SR_CLOCK"):
			a.Deterministic = false
			a.NonDetReasons = appendUniq(a.NonDetReasons, "clock read")
		}
	}

	// Kernel-level PRNG/hash signal: a loop plus dense XOR/bitwise mixing is the
	// xorshift/Philox signature. This catches loop-carried PRNG state that the
	// path-insensitive address slice cannot follow (e.g. gups), where the
	// per-instruction tier comes back affine but the addresses are really tier-4.
	hasLoop, mix := false, 0
	for _, in := range k.Insts {
		if (in.Op == "BRA" || in.Op == "BRX") && in.HasTgt && in.Target <= in.Addr {
			hasLoop = true
		}
		if in.Op == "LOP3" || in.Op == "LOP" || in.Op == "LOP32I" {
			mix++
		}
	}
	a.PRNGMixing = hasLoop && mix >= 4
	return a
}

// predProv classifies a predicate by slicing the comparison that defined it.
func (k *Kernel) predProv(at int, pred string, argOff map[uint64]bool) Prov {
	w := k.lastWriter(at, pred)
	if w < 0 {
		return ProvUnknown
	}
	in := k.Insts[w]
	res := ProvConst
	any := false
	for _, o := range in.Ops[1:] { // skip dest predicate
		r := regName(o)
		if r == "" || strings.HasPrefix(r, "P") {
			if reConst.MatchString(o) {
				res = join(res, ProvArg)
				any = true
			}
			continue
		}
		res = join(res, k.classifyReg(w, r, 0, argOff))
		any = true
	}
	if !any {
		return ProvConst
	}
	return res
}

func isLoad(op string) bool {
	switch op {
	case "LDG", "LD", "LDGSTS":
		return true
	}
	return false
}
func isStore(op string) bool {
	switch op {
	case "STG", "ST":
		return true
	}
	return false
}

// memOperand returns the operand carrying the address (the one with [...]).
func memOperand(in Inst) string {
	for _, o := range in.Ops {
		if strings.Contains(o, "[") {
			return o
		}
	}
	return ""
}

func dot(mods []string) string {
	if len(mods) == 0 {
		return ""
	}
	return "." + strings.Join(mods, ".")
}

func appendUniq(s []string, v string) []string {
	for _, x := range s {
		if x == v {
			return s
		}
	}
	return append(s, v)
}

// Verdict summarizes a kernel for the sampler: how much is synthesizable from
// indices, and what must be captured.
func (a Analysis) Verdict() string {
	var indirect, computed, dataBr, loopDataBr int
	for _, m := range a.Mem {
		switch m.Tier {
		case "indirect":
			indirect++
		case "computed":
			computed++
		}
	}
	for _, b := range a.Branches {
		if b.Class == "data" {
			dataBr++
			if b.BackEdge {
				loopDataBr++
			}
		}
	}
	reg := "REGULAR (synthesizable from indices)"
	switch {
	case indirect > 0 || loopDataBr > 0:
		reg = "IRREGULAR (must capture: data-dependent addresses/loops)"
	case computed > 0 || a.PRNGMixing:
		reg = "COMPUTED (non-affine addresses: PRNG/hash — needs tier-4, affine fit will fail)"
	}
	if a.PRNGMixing && computed == 0 {
		reg += " [loop-carried PRNG mixing detected]"
	}
	det := "deterministic across launches (relaunch-dedup safe)"
	if !a.Deterministic {
		det = "NONDETERMINISTIC: " + strings.Join(a.NonDetReasons, ", ")
	}
	return fmt.Sprintf("%s; %d indirect, %d computed addr, %d data-branches (%d loop); %s",
		reg, indirect, computed, dataBr, loopDataBr, det)
}
