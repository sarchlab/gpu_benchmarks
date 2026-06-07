package ptrace

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func jsonMarshal(v any) ([]byte, error) {
	return json.MarshalIndent(v, "", "  ")
}

func readJSON(path string, v any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}

// ReadManifest loads a .ptrace manifest.
func ReadManifest(ptraceDir string) (*Manifest, error) {
	var m Manifest
	if err := readJSON(filepath.Join(ptraceDir, "manifest.json"), &m); err != nil {
		return nil, err
	}
	return &m, nil
}

// loadISA loads and caches ISA tables by hash.
func loadISA(ptraceDir string, cache map[string]ISATable, hash string) (ISATable, error) {
	if t, ok := cache[hash]; ok {
		return t, nil
	}
	var t ISATable
	if err := readJSON(filepath.Join(ptraceDir, "isa", hash+".json"), &t); err != nil {
		return ISATable{}, err
	}
	cache[hash] = t
	return t, nil
}

// ExpandTraceg reconstructs the full .traceg content for one invocation.
func ExpandTraceg(ptraceDir string, m *Manifest, inv Invocation, isaCache map[string]ISATable) (string, error) {
	table, err := loadISA(ptraceDir, isaCache, inv.ISA)
	if err != nil {
		return "", err
	}
	comp, err := os.ReadFile(filepath.Join(ptraceDir, "dyn", inv.DynFile))
	if err != nil {
		return "", err
	}
	raw, err := zstdDecompress(comp)
	if err != nil {
		return "", err
	}
	blocks, err := decodeDyn(raw, table)
	if err != nil {
		return "", err
	}
	return EmitTraceg(inv.Header, blocks, table.toMap())
}

// Expand reconstructs an Accel-Sim traces/ directory from a .ptrace directory,
// re-emitting kernelslist, stats, and the per-kernel .trace.zst files.
func Expand(ptraceDir, outDir string) error {
	m, err := ReadManifest(ptraceDir)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}
	isaCache := map[string]ISATable{}

	// Re-emit each invocation's .trace.zst.
	for _, inv := range m.Invocations {
		content, err := ExpandTraceg(ptraceDir, m, inv, isaCache)
		if err != nil {
			return fmt.Errorf("expand %s: %w", inv.TraceFile, err)
		}
		comp, err := EncodeContainer(content)
		if err != nil {
			return fmt.Errorf("encode %s: %w", inv.TraceFile, err)
		}
		if err := os.WriteFile(filepath.Join(outDir, inv.TraceFile), comp, 0o644); err != nil {
			return err
		}
	}

	// Re-emit kernelslist + stats per context.
	for _, ctx := range m.Contexts {
		var b strings.Builder
		for _, e := range ctx.Entries {
			b.WriteString(e.Raw)
			b.WriteString("\n")
		}
		if err := os.WriteFile(filepath.Join(outDir, ctx.KernelslistName), []byte(b.String()), 0o644); err != nil {
			return err
		}
		if ctx.StatsName != "" {
			if err := os.WriteFile(filepath.Join(outDir, ctx.StatsName), []byte(ctx.StatsContent), 0o644); err != nil {
				return err
			}
		}
	}
	return nil
}
