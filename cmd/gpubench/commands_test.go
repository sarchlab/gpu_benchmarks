package main

import (
	"encoding/json"
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

func TestParseOutput(t *testing.T) {
	stdout := strings.Join([]string{
		`{"type":"kernel","name":"vector_add","time_ms":1.5,"host_time_ms":2.0,"iter":0}`,
		`{"type":"kernel","name":"vector_add","iter":1}`, // missing time_ms -> 0.0
		`not json, ignored`,
		`{"type":"summary","total_time_ms":3.5,"metrics":[{"name":"GFLOPS","value":42.0}]}`,
		`{"type":"error","message":"boom"}`,
		``,
	}, "\n")

	kernels, summary, errs := parseOutput(stdout)

	if len(kernels) != 2 {
		t.Fatalf("got %d kernels, want 2", len(kernels))
	}
	if kernels[0].Name != "vector_add" || kernels[0].TimeMS != 1.5 {
		t.Fatalf("kernel[0] = %+v", kernels[0])
	}
	if kernels[0].Ordinal != 0 || kernels[1].Ordinal != 1 {
		t.Fatalf("ordinals = %d,%d want 0,1", kernels[0].Ordinal, kernels[1].Ordinal)
	}
	if kernels[1].TimeMS != 0.0 {
		t.Fatalf("kernel[1] missing time_ms should default to 0.0, got %v", kernels[1].TimeMS)
	}
	if summary == nil || summary.TotalTimeMS == nil || *summary.TotalTimeMS != 3.5 {
		t.Fatalf("summary = %+v", summary)
	}
	if len(summary.Metrics) != 1 || summary.Metrics[0].Name != "GFLOPS" || summary.Metrics[0].Value != 42.0 {
		t.Fatalf("summary metrics = %+v", summary.Metrics)
	}
	if len(errs) != 1 || errs[0].Message != "boom" {
		t.Fatalf("errors = %+v", errs)
	}
}

func TestParseOutputProtocolFields(t *testing.T) {
	t.Run("kernel params captured, severity parsed", func(t *testing.T) {
		stdout := strings.Join([]string{
			`{"type":"kernel","name":"gemm","time_ms":1.0,"params":{"matrix_size":256,"block_size":16}}`,
			`{"type":"error","severity":"warning","message":"one value OOM, skipped"}`,
			`{"type":"error","severity":"fatal","message":"driver crash"}`,
		}, "\n")
		kernels, _, errs := parseOutput(stdout)
		if len(kernels) != 1 || kernels[0].Params == "" {
			t.Fatalf("kernel params not captured: %+v", kernels)
		}
		if !strings.Contains(kernels[0].Params, "matrix_size") {
			t.Fatalf("params json = %q", kernels[0].Params)
		}
		if !kernels[0].HasTime {
			t.Fatalf("HasTime should be true when time_ms present")
		}
		if len(errs) != 2 || !errs[0].IsWarning() || errs[1].IsWarning() {
			t.Fatalf("severity parsing wrong: %+v", errs)
		}
		if got := firstFatalMessage(errs); got != "driver crash" {
			t.Fatalf("firstFatalMessage = %q, want driver crash", got)
		}
	})

	t.Run("missing time_ms flagged via HasTime", func(t *testing.T) {
		kernels, _, _ := parseOutput(`{"type":"kernel","name":"k"}`)
		if len(kernels) != 1 || kernels[0].HasTime || kernels[0].TimeMS != 0.0 {
			t.Fatalf("missing time_ms handling wrong: %+v", kernels)
		}
	})
}

func TestCoerceParamValue(t *testing.T) {
	if v := coerceParamValue("256", "int"); v != float64(256) {
		t.Errorf("int coercion = %v (%T), want 256 float64", v, v)
	}
	if v := coerceParamValue("1.5", "float"); v != 1.5 {
		t.Errorf("float coercion = %v, want 1.5", v)
	}
	if v := coerceParamValue("true", "enum"); v != "true" {
		t.Errorf("enum coercion = %v, want string true", v)
	}
	if v := coerceParamValue("abc", "int"); v != "abc" {
		t.Errorf("unparseable int keeps string: got %v", v)
	}
}

func TestGenerateConfigs(t *testing.T) {
	warmup := NonScalingParam{
		Name: "vector_add_warmup", Type: "enum", Default: "true",
		Values: []interface{}{"true", "false"},
	}
	block := NonScalingParam{
		Name: "block_size", Type: "int", Default: float64(256),
		Values: []interface{}{float64(128), float64(256)},
	}
	base := &BenchmarkParams{
		Benchmark:    "b",
		ScalingParam: ScalingParam{Name: "n", Type: "int", Values: []interface{}{float64(64), float64(128)}},
	}

	t.Run("scaling only", func(t *testing.T) {
		cfgs := generateConfigs(base, true, nil, nil)
		if len(cfgs) != 2 {
			t.Fatalf("got %d configs, want 2", len(cfgs))
		}
	})

	t.Run("scaling override replaces axis", func(t *testing.T) {
		cfgs := generateConfigs(base, true, []interface{}{float64(1), float64(2), float64(3)}, nil)
		if len(cfgs) != 3 {
			t.Fatalf("got %d configs, want 3", len(cfgs))
		}
	})

	t.Run("sweep non-scaling cartesian", func(t *testing.T) {
		p := &BenchmarkParams{
			Benchmark:        "b",
			ScalingParam:     base.ScalingParam,
			NonScalingParams: []NonScalingParam{block},
		}
		cfgs := generateConfigs(p, true, nil, nil)
		if len(cfgs) != 2*2 { // 2 scaling × 2 block_size
			t.Fatalf("got %d configs, want 4", len(cfgs))
		}
	})

	t.Run("non-scaling defaults only when not sweeping", func(t *testing.T) {
		p := &BenchmarkParams{
			Benchmark:        "b",
			ScalingParam:     base.ScalingParam,
			NonScalingParams: []NonScalingParam{block},
		}
		cfgs := generateConfigs(p, false, nil, nil)
		if len(cfgs) != 2 { // block_size fixed at default
			t.Fatalf("got %d configs, want 2", len(cfgs))
		}
		if cfgs[0].NSDict["block_size"] != float64(256) {
			t.Fatalf("block_size default = %v, want 256", cfgs[0].NSDict["block_size"])
		}
	})

	t.Run("measurement param excluded from sweep, folded in typed", func(t *testing.T) {
		p := &BenchmarkParams{
			Benchmark:        "b",
			ScalingParam:     base.ScalingParam,
			NonScalingParams: []NonScalingParam{warmup},
		}
		// vector_add_warmup must not multiply the config count (2 scaling only).
		cfgs := generateConfigs(p, true, nil, nil)
		if len(cfgs) != 2 {
			t.Fatalf("measurement param leaked into sweep: got %d, want 2", len(cfgs))
		}
		// Default: stored as declared type (string for enum).
		if cfgs[0].NSDict["vector_add_warmup"] != "true" {
			t.Fatalf("default warmup = %v, want string true", cfgs[0].NSDict["vector_add_warmup"])
		}
		// Override goes through the same enum path -> still the string "false".
		cfgs = generateConfigs(p, true, nil, map[string]string{"vector_add_warmup": "false"})
		if cfgs[0].NSDict["vector_add_warmup"] != "false" {
			t.Fatalf("override warmup = %v (%T), want string false",
				cfgs[0].NSDict["vector_add_warmup"], cfgs[0].NSDict["vector_add_warmup"])
		}
	})
}

func TestBuildEnv(t *testing.T) {
	env := buildEnv("matrix_size", float64(1024), map[string]interface{}{
		"block_size": float64(256),
	})
	if !envHas(env, "BENCH_PARAM_matrix_size=1024") {
		t.Fatalf("scaling param env missing; env tail = %v", env[len(env)-3:])
	}
	if !envHas(env, "BENCH_PARAM_block_size=256") {
		t.Fatalf("non-scaling param env missing; env tail = %v", env[len(env)-3:])
	}
}

func envHas(env []string, kv string) bool {
	for _, e := range env {
		if e == kv {
			return true
		}
	}
	return false
}

func TestValueMatchesType(t *testing.T) {
	cases := []struct {
		v    interface{}
		typ  string
		want bool
	}{
		{float64(256), "int", true},
		{float64(256.5), "int", false},
		{float64(1.5), "float", true},
		{float64(2), "float", true},
		{"naive", "string", true},
		{"naive", "enum", true},
		{float64(1), "string", false},
		{"x", "int", false},
	}
	for _, c := range cases {
		if got := valueMatchesType(c.v, c.typ); got != c.want {
			t.Errorf("valueMatchesType(%v, %q) = %v, want %v", c.v, c.typ, got, c.want)
		}
	}
}

// validateParamsJSON writes params under <tmp>/tier1/<bench>/params.json and
// validates it, returning the errors and warnings.
func validateParamsJSON(t *testing.T, bench string, params map[string]interface{}) (errs, warns []string) {
	t.Helper()
	pp := filepath.Join(t.TempDir(), "tier1", bench, "params.json")
	data, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	writeTestFile(t, pp, string(data))
	var decoded map[string]interface{}
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal params: %v", err)
	}
	return validateParams(decoded, pp)
}

func baseValidParams(bench string) map[string]interface{} {
	return map[string]interface{}{
		"benchmark":   bench,
		"tier":        float64(1),
		"suite":       "microbench",
		"description": "test",
		"scaling_param": map[string]interface{}{
			"name":        "n",
			"description": "size",
			"type":        "int",
			"values":      []interface{}{float64(64), float64(128), float64(256)},
		},
		"non_scaling_params": []interface{}{},
		"kernels":            []interface{}{"my_kernel"},
	}
}

func hasErrContaining(errs []string, sub string) bool {
	for _, e := range errs {
		if strings.Contains(e, sub) {
			return true
		}
	}
	return false
}

func TestValidateParamsChecks(t *testing.T) {
	t.Run("valid params have no errors", func(t *testing.T) {
		errs, _ := validateParamsJSON(t, "mybench", baseValidParams("mybench"))
		if len(errs) != 0 {
			t.Fatalf("unexpected errors: %v", errs)
		}
	})

	t.Run("int scaling param with float value errors", func(t *testing.T) {
		p := baseValidParams("mybench")
		p["scaling_param"].(map[string]interface{})["values"] =
			[]interface{}{float64(64), float64(128.5)}
		errs, _ := validateParamsJSON(t, "mybench", p)
		if !hasErrContaining(errs, "not a valid int") {
			t.Fatalf("expected type error, got: %v", errs)
		}
	})

	t.Run("duplicate scaling values error", func(t *testing.T) {
		p := baseValidParams("mybench")
		p["scaling_param"].(map[string]interface{})["values"] =
			[]interface{}{float64(64), float64(64)}
		errs, _ := validateParamsJSON(t, "mybench", p)
		if !hasErrContaining(errs, "duplicate value") {
			t.Fatalf("expected duplicate value error, got: %v", errs)
		}
	})

	t.Run("bad scaling type string errors", func(t *testing.T) {
		p := baseValidParams("mybench")
		p["scaling_param"].(map[string]interface{})["type"] = "integer"
		errs, _ := validateParamsJSON(t, "mybench", p)
		if !hasErrContaining(errs, "not one of int/float/string/enum") {
			t.Fatalf("expected bad type error, got: %v", errs)
		}
	})

	t.Run("non-string scaling type errors", func(t *testing.T) {
		p := baseValidParams("mybench")
		p["scaling_param"].(map[string]interface{})["type"] = float64(5)
		errs, _ := validateParamsJSON(t, "mybench", p)
		if !hasErrContaining(errs, "not one of int/float/string/enum") {
			t.Fatalf("expected non-string type error, got: %v", errs)
		}
	})

	t.Run("non-string non-scaling type errors", func(t *testing.T) {
		p := baseValidParams("mybench")
		p["non_scaling_params"] = []interface{}{
			map[string]interface{}{
				"name": "algo", "description": "d", "type": float64(1),
				"default": "naive", "values": []interface{}{"naive"},
			},
		}
		errs, _ := validateParamsJSON(t, "mybench", p)
		if !hasErrContaining(errs, "not one of int/float/string/enum") {
			t.Fatalf("expected non-string nsp type error, got: %v", errs)
		}
	})

	t.Run("non-scaling default type mismatch errors", func(t *testing.T) {
		p := baseValidParams("mybench")
		p["non_scaling_params"] = []interface{}{
			map[string]interface{}{
				"name":        "algo",
				"description": "algorithm",
				"type":        "int",
				"default":     "naive", // string default for int type
				"values":      []interface{}{"naive"},
			},
		}
		errs, _ := validateParamsJSON(t, "mybench", p)
		if !hasErrContaining(errs, "default") {
			t.Fatalf("expected default type error, got: %v", errs)
		}
	})

	t.Run("duplicate kernel names error", func(t *testing.T) {
		p := baseValidParams("mybench")
		p["kernels"] = []interface{}{"k1", "k1"}
		errs, _ := validateParamsJSON(t, "mybench", p)
		if !hasErrContaining(errs, "Duplicate kernel name") {
			t.Fatalf("expected duplicate kernel error, got: %v", errs)
		}
	})

	t.Run("non-string kernel errors", func(t *testing.T) {
		p := baseValidParams("mybench")
		p["kernels"] = []interface{}{float64(1)}
		errs, _ := validateParamsJSON(t, "mybench", p)
		if !hasErrContaining(errs, "not a string") {
			t.Fatalf("expected non-string kernel error, got: %v", errs)
		}
	})

	t.Run("empty kernels list is allowed", func(t *testing.T) {
		p := baseValidParams("mybench")
		p["kernels"] = []interface{}{}
		errs, _ := validateParamsJSON(t, "mybench", p)
		if len(errs) != 0 {
			t.Fatalf("empty kernels should be valid, got: %v", errs)
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
