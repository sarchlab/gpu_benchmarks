package main

import (
	"os"
	"path/filepath"
	"testing"
)

func writeTemp(t *testing.T, name, content string) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestParseRocprofCSV_DurationColumn_LastDispatch(t *testing.T) {
	// rocprof v1 style: KernelName + DurationNs, dispatch order (warmups then
	// the timed launch). The LAST dispatch per kernel must win.
	csv := "Index,KernelName,DurationNs\n" +
		"0,smem_bw_no_conflict,1000000\n" + // warmup
		"1,smem_bw_no_conflict,2000000\n" + // timed (last) -> 2.0 ms
		"2,smem_bw_conflict,5000000\n" // -> 5.0 ms
	got := parseRocprofCSV(writeTemp(t, "results.csv", csv))
	if got["smem_bw_no_conflict"] != 2.0 {
		t.Errorf("no_conflict: want 2.0 ms, got %v", got["smem_bw_no_conflict"])
	}
	if got["smem_bw_conflict"] != 5.0 {
		t.Errorf("conflict: want 5.0 ms, got %v", got["smem_bw_conflict"])
	}
}

func TestParseRocprofCSV_BeginEndFallback_AndBaseName(t *testing.T) {
	// rocprofv3 style: no duration column, demangled names with signatures.
	csv := "Kernel_Name,Start_Timestamp,End_Timestamp\n" +
		"\"smem_bw_no_conflict(float const*, float*, int)\",100,1100\n" // 1000 ns = 0.001 ms
	got := parseRocprofCSV(writeTemp(t, "results.csv", csv))
	if v := got["smem_bw_no_conflict"]; v != 0.001 {
		t.Errorf("want 0.001 ms (base name + end-begin), got %v (map=%v)", v, got)
	}
}

func TestParseRocprofCSV_Unusable(t *testing.T) {
	if got := parseRocprofCSV(writeTemp(t, "x.csv", "a,b\n1,2\n")); got != nil {
		t.Errorf("expected nil for CSV without kernel/duration columns, got %v", got)
	}
	if got := parseRocprofCSV(filepath.Join(t.TempDir(), "missing.csv")); got != nil {
		t.Errorf("expected nil for missing file, got %v", got)
	}
}

func TestKernelBaseName(t *testing.T) {
	cases := map[string]string{
		"smem_bw_no_conflict":                            "smem_bw_no_conflict",
		"smem_bw_no_conflict(float const*, float*, int)": "smem_bw_no_conflict",
		"ns::foo(int)":                                   "foo",
		"  spaced  ":                                     "spaced",
	}
	for in, want := range cases {
		if got := kernelBaseName(in); got != want {
			t.Errorf("kernelBaseName(%q) = %q, want %q", in, got, want)
		}
	}
}
