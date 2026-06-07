package ptrace

import (
	"os"
	"path/filepath"
	"testing"
)

// TestConvertSampledEndToEnd writes a synthetic Accel-Sim traces directory for a
// fully-affine kernel, runs the P3 sampled conversion through the filesystem, and
// checks that (a) expand reproduces the original .traceg byte-for-byte and (b) the
// manifest records that most blocks were synthesized rather than recorded.
func TestConvertSampledEndToEnd(t *testing.T) {
	blocks, isa := affineGrid(8)
	content, err := EmitTraceg("-kernel name = test\n-block dim = (32,1,1)\n\n#traces format = [PC] ...\n\n", blocks, isa)
	if err != nil {
		t.Fatal(err)
	}
	container, err := EncodeContainer(content)
	if err != nil {
		t.Fatal(err)
	}

	src := t.TempDir()
	if err := os.WriteFile(filepath.Join(src, "kernel-1.trace.zst"), container, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(src, "kernelslist"), []byte("MemcpyHtoD,0x7f00,1024\nkernel-1.trace.zsim\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	out := filepath.Join(t.TempDir(), "k.ptrace")
	if err := ConvertSampled(src, out); err != nil {
		t.Fatal(err)
	}
	m, err := ReadManifest(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Invocations) != 1 {
		t.Fatalf("invocations = %d, want 1", len(m.Invocations))
	}
	cov := m.Invocations[0].Cov
	if cov.TotalBlocks != 8 || cov.RecordedBlocks != 2 {
		t.Fatalf("coverage: recorded %d/%d, want 2/8", cov.RecordedBlocks, cov.TotalBlocks)
	}

	isaCache := map[string]ISATable{}
	got, err := ExpandTraceg(out, m, m.Invocations[0], isaCache)
	if err != nil {
		t.Fatal(err)
	}
	if got != content {
		t.Fatalf("sampled expand not byte-identical:\n got=%q\nwant=%q", got, content)
	}
}
