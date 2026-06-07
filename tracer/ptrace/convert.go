package ptrace

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// traceFileFromRef maps a kernelslist reference (".trace.zsim") to the on-disk
// compressed file name (".trace.zst").
func traceFileFromRef(ref string) string {
	ref = filepath.Base(strings.TrimSpace(ref))
	switch {
	case strings.HasSuffix(ref, ".trace.zsim"):
		return strings.TrimSuffix(ref, ".trace.zsim") + ".trace.zst"
	case strings.HasSuffix(ref, ".trace.zst"):
		return ref
	default:
		return ""
	}
}

func isKernelRef(line string) bool {
	return strings.HasSuffix(line, ".trace.zsim") || strings.HasSuffix(line, ".trace.zst")
}

// Convert reads an Accel-Sim traces/ directory and writes a .ptrace directory
// (P1/P2: every block stored, addresses modeled where possible).
func Convert(tracesDir, outDir string) error {
	return convert(tracesDir, outDir, false)
}

// ConvertSampled writes a sampled (P3) .ptrace: only a corner sample of blocks
// plus blocks that fail synthesis are stored; the rest are reconstructed from
// recipes on expand. Still lossless (verify-or-fallback), and the recorded block
// fraction approximates the collection wall-time a real sampled tracer would pay.
func ConvertSampled(tracesDir, outDir string) error {
	return convert(tracesDir, outDir, true)
}

func convert(tracesDir, outDir string, sampled bool) error {
	klists, _ := filepath.Glob(filepath.Join(tracesDir, "kernelslist*"))
	sort.Strings(klists)
	if len(klists) == 0 {
		return fmt.Errorf("no kernelslist* in %s", tracesDir)
	}

	if err := os.MkdirAll(filepath.Join(outDir, "isa"), 0o755); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Join(outDir, "dyn"), 0o755); err != nil {
		return err
	}

	man := Manifest{Version: FormatVersion, Source: "accelsim-convert"}
	invByFile := map[string]int{} // trace filename -> invocation index
	isaWritten := map[string]bool{}

	for _, kl := range klists {
		data, err := os.ReadFile(kl)
		if err != nil {
			return err
		}
		ctx := Context{KernelslistName: filepath.Base(kl)}

		// matching stats file: kernelslist_ctx_X -> stats_ctx_X
		statsName := strings.Replace(filepath.Base(kl), "kernelslist", "stats", 1)
		if sb, err := os.ReadFile(filepath.Join(tracesDir, statsName)); err == nil {
			ctx.StatsName = statsName
			ctx.StatsContent = string(sb)
		}

		for _, line := range strings.Split(string(data), "\n") {
			s := strings.TrimSpace(line)
			if s == "" {
				continue
			}
			if !isKernelRef(s) {
				ctx.Entries = append(ctx.Entries, Entry{Type: "memcpy", Raw: s})
				continue
			}
			tf := traceFileFromRef(s)
			inv, ok := invByFile[tf]
			if !ok {
				inv, err = addInvocation(tracesDir, outDir, tf, &man, isaWritten, sampled)
				if err != nil {
					return fmt.Errorf("invocation %s: %w", tf, err)
				}
				invByFile[tf] = inv
			}
			ctx.Entries = append(ctx.Entries, Entry{Type: "kernel", Raw: s, Inv: inv})
		}
		man.Contexts = append(man.Contexts, ctx)
	}

	return writeJSON(filepath.Join(outDir, "manifest.json"), man)
}

func addInvocation(tracesDir, outDir, traceFile string, man *Manifest, isaWritten map[string]bool, sampled bool) (int, error) {
	raw, err := os.ReadFile(filepath.Join(tracesDir, traceFile))
	if err != nil {
		return 0, err
	}
	header, blocksText, err := DecodeContainer(raw)
	if err != nil {
		return 0, err
	}
	content := header + strings.Join(blocksText, "")

	isaMap := map[int]StaticInst{}
	hdr, blocks, err := ParseTraceg(content, isaMap)
	if err != nil {
		return 0, err
	}
	table, pcIdx := buildISA(isaMap)
	hash := table.hash()
	if !isaWritten[hash] {
		if err := writeJSON(filepath.Join(outDir, "isa", hash+".json"), table); err != nil {
			return 0, err
		}
		isaWritten[hash] = true
	}

	var dyn []byte
	var cov Coverage
	if sampled {
		plan, recipes := buildSamplePlan(blocks, isaMap)
		dyn, cov, err = encodeDynSampled(blocks, plan, recipes, pcIdx, isaMap)
	} else {
		dyn, cov, err = encodeDyn(blocks, pcIdx, isaMap)
	}
	if err != nil {
		return 0, err
	}
	comp, err := zstdCompress(dyn)
	if err != nil {
		return 0, err
	}
	invIdx := len(man.Invocations)
	dynName := fmt.Sprintf("inv_%d.zst", invIdx)
	if err := os.WriteFile(filepath.Join(outDir, "dyn", dynName), comp, 0o644); err != nil {
		return 0, err
	}
	man.Invocations = append(man.Invocations, Invocation{
		TraceFile: traceFile,
		Header:    hdr,
		ISA:       hash,
		DynFile:   dynName,
		Cov:       cov,
	})
	return invIdx, nil
}

func writeJSON(path string, v any) error {
	b, err := jsonMarshal(v)
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}
