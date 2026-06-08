package ptrace

// scale.go implements *scaling extrapolation*: generate the trace of a larger
// per-thread-work problem size from a small run, by replaying the kernel's loop
// body more times. Where `synth --grid` extrapolates the spatial axis (more
// blocks), this extrapolates the temporal axis (a longer loop) — the axis that
// drives the expensive compute configs (fp32 at 65536 ops is the 64-op loop body
// run 16384 times instead of 16).
//
// A regular kernel's warp stream is prologue + body*k + epilogue. We detect the
// dominant consecutively-repeating segment (the body) and its count k in the
// sample, then re-emit it k' times for the target size. Memory instructions in
// the body get their per-iteration address from the recipe's `occ` (iteration)
// term, so this composes with P2 address modeling. It is gated by byte-exact
// verification against a real trace at the target size (see `ptrace scale`).
//
// Scope: regular kernels with a loop-invariant body and a trip count that scales
// with the parameter (the tier-1 compute microbenchmarks). Data-dependent trip
// counts (pointer chase, sparse inner loops) are not synthesizable this way.

// sameLoopInst compares two instructions for loop-body identity by PC and
// address-encoding mode only. Neither mask nor address is compared: the loop's
// back-edge branch is recorded with the taken mask on every iteration except the
// last (where it exits, mask 0), and memory addresses advance per iteration (the
// `occ` term). Those differences are reproduced at replay time, not matched here.
func sameLoopInst(a, b DynInst) bool {
	return a.Pc == b.Pc && a.Mode == b.Mode
}

func segEq(w []DynInst, a, b, p int) bool {
	for i := 0; i < p; i++ {
		if !sameLoopInst(w[a+i], w[b+i]) {
			return false
		}
	}
	return true
}

// detectLoop finds the dominant consecutively-repeating segment of a warp stream:
// the (prologue length, body length, repeat count) maximizing body*reps, with
// reps >= 2. ok=false when no repetition is found (the sample ran < 2 iterations).
func detectLoop(warp []DynInst) (preLen, bodyLen, reps int, ok bool) {
	n := len(warp)
	best := 0
	for s := 0; s < n; s++ {
		for p := 1; s+2*p <= n; p++ {
			r := 1
			for s+(r+1)*p <= n && segEq(warp, s, s+r*p, p) {
				r++
			}
			if r >= 2 && r*p > best {
				best = r * p
				preLen, bodyLen, reps, ok = s, p, r, true
			}
		}
	}
	return
}

// scaleWarp re-emits a warp with the loop run targetReps times instead of its
// detected `reps`. The body is replayed as (taken iteration)×(targetReps-1) +
// (exit iteration), so the back-edge masks land exactly as a real run would.
// Prologue and epilogue are copied verbatim. Memory instructions inside the loop
// body get their per-iteration address from the recipe's `occ` term; an unmodeled
// (data-dependent) body access makes the warp unscalable (ok=false). Memory in
// the prologue/epilogue is iteration-independent and copied verbatim.
func scaleWarp(warp []DynInst, bx, by, bz, wi, preLen, bodyLen, reps, targetReps int, recipes map[int]Recipe, isa map[int]StaticInst) ([]DynInst, bool) {
	taken := warp[preLen : preLen+bodyLen]                      // iteration 0 (branches back)
	exit := warp[preLen+(reps-1)*bodyLen : preLen+reps*bodyLen] // last iteration (falls through)
	epilogue := warp[preLen+reps*bodyLen:]

	// Build the target source stream, tagging which instructions are loop body.
	type tagged struct {
		d    DynInst
		body bool
	}
	src := make([]tagged, 0, preLen+bodyLen*targetReps+len(epilogue))
	for i := 0; i < preLen; i++ {
		src = append(src, tagged{warp[i], false})
	}
	for r := 0; r < targetReps-1; r++ {
		for i := range taken {
			src = append(src, tagged{taken[i], true})
		}
	}
	for i := range exit {
		src = append(src, tagged{exit[i], true})
	}
	for i := range epilogue {
		src = append(src, tagged{epilogue[i], false})
	}

	occ := map[int]int{}
	out := make([]DynInst, len(src))
	for i, s := range src {
		d := s.d
		nd := DynInst{Pc: d.Pc, Mask: d.Mask, Mode: d.Mode, Payload: d.Payload}
		o := occ[d.Pc]
		occ[d.Pc] = o + 1
		if s.body && isa[d.Pc].IsMem {
			rec, has := recipes[d.Pc]
			if !has {
				return nil, false
			}
			ci := rec.classIndexFor(d.Mode, tailOf(d.Payload))
			if ci < 0 {
				return nil, false
			}
			nd.Payload = rec.Classes[ci].payloadFor(bx, by, bz, wi, o)
		}
		out[i] = nd
	}
	return out, true
}

// ScaleBlocks re-emits a decoded invocation at a per-thread work scaled by
// num/den (e.g. from=8, to=64 -> num=64, den=8). Every warp's loop is replayed
// reps*num/den times. ok=false if any warp has no detectable loop, a non-integer
// target count, or an unmodeled data-dependent body — i.e. it is not regularly
// scalable and must be collected directly.
func ScaleBlocks(blocks []Block, isa map[int]StaticInst, num, den int) ([]Block, bool) {
	if den == 0 {
		return nil, false
	}
	recipes := fitRecipes(blocks, isa)
	out := make([]Block, len(blocks))
	for bi, blk := range blocks {
		nb := Block{X: blk.X, Y: blk.Y, Z: blk.Z, Warps: make([][]DynInst, len(blk.Warps))}
		for wi, warp := range blk.Warps {
			preLen, bodyLen, reps, ok := detectLoop(warp)
			if !ok {
				return nil, false
			}
			if (reps*num)%den != 0 {
				return nil, false
			}
			tr := reps * num / den
			if tr < 1 {
				return nil, false
			}
			sw, ok := scaleWarp(warp, blk.X, blk.Y, blk.Z, wi, preLen, bodyLen, reps, tr, recipes, isa)
			if !ok {
				return nil, false
			}
			nb.Warps[wi] = sw
		}
		out[bi] = nb
	}
	return out, true
}
