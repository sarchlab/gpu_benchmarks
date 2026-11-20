package main

import (
	"path/filepath"
	"testing"
)

func TestTraceCompleteV2(t *testing.T) {
	t.Run("raw trace without post-processing is incomplete", func(t *testing.T) {
		dir := t.TempDir()
		traces := filepath.Join(dir, "traces")
		writeTestFile(t, filepath.Join(traces, "kernelslist"), "kernel-1-ctx_0xabc.trace.xz\n")
		writeTestFile(t, filepath.Join(traces, "kernel-1-ctx_0xabc.trace.xz"), "xz")

		complete, exists, _ := traceCompleteV2(dir)
		if complete || !exists {
			t.Fatalf("traceCompleteV2() = complete %v exists %v, want false true", complete, exists)
		}
	})

	t.Run("missing tracez is incomplete", func(t *testing.T) {
		dir := t.TempDir()
		traces := filepath.Join(dir, "traces")
		writeTestFile(t, filepath.Join(traces, "kernelslist.g"),
			"MemcpyHtoD,0x0,128\nkernel-1-ctx_0xabc.tracez\nkernel-2-ctx_0xabc.tracez\n")
		writeTestFile(t, filepath.Join(traces, "kernel-1-ctx_0xabc.tracez"), "z")

		complete, _, detail := traceCompleteV2(dir)
		if complete {
			t.Fatalf("traceCompleteV2() complete with %q missing", "kernel-2-ctx_0xabc.tracez")
		}
		if detail != "kernelslist.g references missing kernel-2-ctx_0xabc.tracez" {
			t.Fatalf("detail = %q", detail)
		}
	})

	t.Run("post-processed trace is complete without raw files", func(t *testing.T) {
		dir := t.TempDir()
		traces := filepath.Join(dir, "traces")
		writeTestFile(t, filepath.Join(traces, "kernelslist.g"),
			"MemcpyHtoD,0x0,128\nkernel-1-ctx_0xabc.tracez\n")
		writeTestFile(t, filepath.Join(traces, "kernel-1-ctx_0xabc.tracez"), "z")

		complete, exists, detail := traceCompleteV2(dir)
		if !complete || !exists {
			t.Fatalf("traceCompleteV2() = complete %v exists %v detail %q, want true true", complete, exists, detail)
		}
	})

	t.Run("v1 rule does not accept v2.0 output", func(t *testing.T) {
		dir := t.TempDir()
		traces := filepath.Join(dir, "traces")
		writeTestFile(t, filepath.Join(traces, "kernelslist.g"), "kernel-1-ctx_0xabc.tracez\n")
		writeTestFile(t, filepath.Join(traces, "kernel-1-ctx_0xabc.tracez"), "z")

		if complete, _, _ := traceCompleteFor(TracerV1, dir); complete {
			t.Fatal("traceCompleteFor(v1) accepted .tracez output")
		}
		if complete, _, _ := traceCompleteFor(TracerV2, dir); !complete {
			t.Fatal("traceCompleteFor(v2.0) rejected .tracez output")
		}
	})
}

func TestParamPins(t *testing.T) {
	base := &BenchmarkParams{
		Benchmark:    "b",
		ScalingParam: ScalingParam{Name: "n", Type: "int", Values: []interface{}{float64(64), float64(128)}},
		NonScalingParams: []NonScalingParam{
			{Name: "block_size", Type: "int", Default: float64(256),
				Values: []interface{}{float64(128), float64(256)}},
			{Name: "theta", Type: "float", Default: 0.5,
				Values: []interface{}{0.3, 0.5}},
			{Name: "vector_add_warmup", Type: "enum", Default: "true",
				Values: []interface{}{"true", "false"}},
		},
	}

	t.Run("parse", func(t *testing.T) {
		pins, err := parseParamPins([]string{"block_size=128", " theta = 0.3 "})
		if err != nil || len(pins) != 2 || pins[1] != (paramPin{"theta", "0.3"}) {
			t.Fatalf("parseParamPins() = %v, %v", pins, err)
		}
		for _, bad := range [][]string{{"block_size"}, {"=1"}, {"a=1", "a=2"}} {
			if _, err := parseParamPins(bad); err == nil {
				t.Fatalf("parseParamPins(%q) accepted", bad)
			}
		}
	})

	t.Run("pinned value replaces sweep and default", func(t *testing.T) {
		pinned, err := pinNonScaling(base, []paramPin{{"block_size", "128"}, {"theta", "0.3"}})
		if err != nil {
			t.Fatalf("pinNonScaling: %v", err)
		}
		for _, sweep := range []bool{true, false} {
			cfgs := generateConfigs(pinned, sweep, nil, nil)
			if len(cfgs) != 2 {
				t.Fatalf("sweep=%v: got %d configs, want 2", sweep, len(cfgs))
			}
			for _, c := range cfgs {
				if c.NSDict["block_size"] != float64(128) || c.NSDict["theta"] != 0.3 {
					t.Fatalf("sweep=%v: config %v not pinned", sweep, c.NSDict)
				}
			}
		}
		if len(base.NonScalingParams[0].Values) != 2 {
			t.Fatal("pinNonScaling modified its input")
		}
	})

	t.Run("rejects unknown, undeclared and measurement params", func(t *testing.T) {
		for _, pin := range []paramPin{{"nope", "1"}, {"block_size", "512"}, {"vector_add_warmup", "false"}} {
			if _, err := pinNonScaling(base, []paramPin{pin}); err == nil {
				t.Fatalf("pinNonScaling(%v) accepted", pin)
			}
		}
	})
}
