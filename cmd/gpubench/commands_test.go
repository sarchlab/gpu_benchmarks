package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeTestFile(t *testing.T, path, contents string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatalf("mkdir %s: %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, []byte(contents), 0644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func TestTraceComplete(t *testing.T) {
	t.Run("missing traces dir", func(t *testing.T) {
		complete, exists, _ := traceComplete(t.TempDir())
		if complete || exists {
			t.Fatalf("traceComplete() = complete %v exists %v, want false false", complete, exists)
		}
	})

	t.Run("metadata without trace file is incomplete", func(t *testing.T) {
		dir := t.TempDir()
		traces := filepath.Join(dir, "traces")
		writeTestFile(t, filepath.Join(traces, "kernelslist_ctx_0xabc"),
			"MemcpyHtoD,0x0,128\nkernel-1-ctx_0xabc.trace.zsim\n")
		writeTestFile(t, filepath.Join(traces, "stats_ctx_0xabc"),
			"header\nkernel-1-ctx_0xabc.trace.zsim,kernel,1,1,1\n")

		complete, exists, _ := traceComplete(dir)
		if complete || !exists {
			t.Fatalf("traceComplete() = complete %v exists %v, want false true", complete, exists)
		}
	})

	t.Run("matching kernelslist stats and trace file is complete", func(t *testing.T) {
		dir := t.TempDir()
		traces := filepath.Join(dir, "traces")
		writeTestFile(t, filepath.Join(traces, "kernelslist_ctx_0xabc"),
			"MemcpyHtoD,0x0,128\nkernel-1-ctx_0xabc.trace.zsim\n")
		writeTestFile(t, filepath.Join(traces, "stats_ctx_0xabc"),
			"header\nkernel-1-ctx_0xabc.trace.zsim,kernel,1,1,1\n")
		writeTestFile(t, filepath.Join(traces, "kernel-1-ctx_0xabc.trace.zst"), "trace")

		complete, exists, detail := traceComplete(dir)
		if !complete || !exists {
			t.Fatalf("traceComplete() = complete %v exists %v detail %q, want true true", complete, exists, detail)
		}
	})

	t.Run("one complete context is enough", func(t *testing.T) {
		dir := t.TempDir()
		traces := filepath.Join(dir, "traces")
		writeTestFile(t, filepath.Join(traces, "kernelslist_ctx_0xbad"),
			"kernel-1-ctx_0xbad.trace.zsim\n")
		writeTestFile(t, filepath.Join(traces, "stats_ctx_0xbad"),
			"header\nkernel-1-ctx_0xbad.trace.zsim,kernel,1,1,1\n")
		writeTestFile(t, filepath.Join(traces, "kernelslist_ctx_0xgood"),
			"kernel-1-ctx_0xgood.trace.zsim\n")
		writeTestFile(t, filepath.Join(traces, "stats_ctx_0xgood"),
			"header\nkernel-1-ctx_0xgood.trace.zsim,kernel,1,1,1\n")
		writeTestFile(t, filepath.Join(traces, "kernel-1-ctx_0xgood.trace.zst"), "trace")

		complete, exists, detail := traceComplete(dir)
		if !complete || !exists {
			t.Fatalf("traceComplete() = complete %v exists %v detail %q, want true true", complete, exists, detail)
		}
	})
}

func TestMarkTraceTimedOut(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, filepath.Join(dir, "traces", "kernel-1-ctx_0xabc.trace.zst"), "large trace")

	if err := markTraceTimedOut(dir, 3600); err != nil {
		t.Fatalf("markTraceTimedOut() error = %v", err)
	}
	if !traceTimedOut(dir) {
		t.Fatalf("traceTimedOut() = false, want true")
	}
	if _, err := os.Stat(filepath.Join(dir, "traces", "kernel-1-ctx_0xabc.trace.zst")); !os.IsNotExist(err) {
		t.Fatalf("trace file still exists after timeout cleanup, stat err = %v", err)
	}
	marker, err := os.ReadFile(traceTimeoutMarkerPath(dir))
	if err != nil {
		t.Fatalf("read timeout marker: %v", err)
	}
	if !strings.Contains(string(marker), `"status": "timeout"`) {
		t.Fatalf("timeout marker missing status: %s", marker)
	}
}
