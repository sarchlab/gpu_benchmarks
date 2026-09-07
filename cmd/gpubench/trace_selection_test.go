package main

import (
	"database/sql"
	"os"
	"path/filepath"
	"testing"
)

func traceSelectionFixture(t *testing.T) (*sql.DB, string, int64) {
	t.Helper()
	db, path := newExportTestDB(t)
	platform := mustExec(t, db, "INSERT INTO platforms (hostname,os,gpu_model,platform_type) VALUES ('fixture','fixture','fixture','cuda')")
	session := mustExec(t, db, "INSERT INTO sessions (platform_id,status) VALUES (?,'completed')", platform)
	return db, path, session
}

func addTraceSelectionRun(t *testing.T, db *sql.DB, session int64, tier int, suite, axis, value, params, status string) int64 {
	t.Helper()
	return mustExec(t, db, `INSERT INTO runs
		(session_id,tier,suite,benchmark,scaling_param_name,scaling_param_value,
		 non_scaling_params_json,repetition,status)
		VALUES (?,?,?,'demo',?,?,?,?,?)`, session, tier, suite, axis, value, params, 1, status)
}

func TestSuccessfulTraceConfigurations(t *testing.T) {
	db, path, session := traceSelectionFixture(t)
	good := addTraceSelectionRun(t, db, session, 1, "microbench", "size", "256",
		`{"vector_add_warmup":"true", "block_size": 64.0}`, "ok")
	addTraceSelectionRun(t, db, session, 1, "microbench", "size", "256",
		`{"block_size":256,"vector_add_warmup":"true"}`, "error")
	// Successes with a different axis, tier, suite, or measurement control must
	// not qualify the failed/untested full configuration.
	for _, spec := range []struct {
		tier                int
		suite, axis, params string
	}{
		{1, "microbench", "other_size", `{"block_size":512,"vector_add_warmup":"true"}`},
		{2, "microbench", "size", `{"block_size":512,"vector_add_warmup":"true"}`},
		{1, "other_suite", "size", `{"block_size":512,"vector_add_warmup":"true"}`},
		{1, "microbench", "size", `{"block_size":512,"vector_add_warmup":"false"}`},
	} {
		addTraceSelectionRun(t, db, session, spec.tier, spec.suite, spec.axis, "256", spec.params, "ok")
	}
	// Preserve all-sessions-in-selected-DB scope and retain a deterministic
	// successful source even if a later repetition of that same config fails.
	platform := mustExec(t, db, "INSERT INTO platforms (hostname,os,gpu_model,platform_type) VALUES ('second','fixture','fixture','cuda')")
	second := mustExec(t, db, "INSERT INTO sessions (platform_id) VALUES (?)", platform)
	addTraceSelectionRun(t, db, second, 1, "microbench", "size", "256",
		`{"block_size":64,"vector_add_warmup":"true"}`, "ok")
	addTraceSelectionRun(t, db, second, 1, "microbench", "size", "256",
		`{"block_size":64,"vector_add_warmup":"true"}`, "error")

	guidance, err := loadTraceGuidance(path)
	if err != nil {
		t.Fatal(err)
	}
	bench := &BenchmarkParams{Tier: 1, Suite: "microbench", Benchmark: "demo", ScalingParam: ScalingParam{Name: "size"}}
	for _, block := range []int{64, 256, 512} {
		key, err := requestedTraceConfig(bench, configPair{ScalingValue: 256, NSDict: map[string]interface{}{"block_size": block, "vector_add_warmup": "true"}})
		if err != nil {
			t.Fatal(err)
		}
		source, found := guidance[key]
		if found != (block == 64) {
			t.Fatalf("block_size %d eligible=%v; only successful block_size=64 may pass", block, found)
		}
		if found && (source.RunID != good || source.SessionID != session) {
			t.Fatalf("source=%+v, want run %d session %d", source, good, session)
		}
	}
	_, emptyPath, _ := traceSelectionFixture(t)
	empty, err := loadTraceGuidance(emptyPath)
	if err != nil || empty == nil || len(empty) != 0 {
		t.Fatalf("different selected DB must authorize no configs: %v %v", empty, err)
	}
}

func TestCanonicalTraceParameters(t *testing.T) {
	for _, pair := range [][2]string{
		{`{"a":1,"b":"x"}`, ` { "b" : "x", "a":1.0 } `},
		{`{"a":1e2}`, `{"a":100}`},
		{`{"a":0.125}`, `{"a":1.25e-1}`},
	} {
		a, e1 := canonicalConfigParams(pair[0])
		b, e2 := canonicalConfigParams(pair[1])
		if e1 != nil || e2 != nil || a != b {
			t.Fatalf("equivalent %v -> %q/%q, errors %v/%v", pair, a, b, e1, e2)
		}
	}
	for _, pair := range [][2]string{
		{`{"a":9007199254740992}`, `{"a":9007199254740993}`},
		{`{"a":1}`, `{"a":"1"}`},
		{`{"a":true}`, `{"a":"true"}`},
		{`{"a":1}`, `{"a":1,"b":2}`},
	} {
		a, e1 := canonicalConfigParams(pair[0])
		b, e2 := canonicalConfigParams(pair[1])
		if e1 != nil || e2 != nil || a == b {
			t.Fatalf("different %v -> %q/%q, errors %v/%v", pair, a, b, e1, e2)
		}
	}
	for _, raw := range []string{`null`, `[]`, `42`, `"scalar"`, `{`, `{"a":null}`, `{"a":[]}`, `{"a":{}}`} {
		if _, err := canonicalConfigParams(raw); err == nil {
			t.Errorf("accepted malformed params %s", raw)
		}
	}
}

func TestTraceGuidanceFailsClosed(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "missing.db")
	if got, err := loadTraceGuidance(missing); err != nil || got != nil {
		t.Fatalf("missing DB should retain unfiltered mode: %v %v", got, err)
	}
	if _, err := os.Stat(missing); !os.IsNotExist(err) {
		t.Fatalf("read created DB: %v", err)
	}
	for _, content := range []string{"not sqlite", ""} {
		path := filepath.Join(t.TempDir(), "bad.db")
		writeTestFile(t, path, content)
		if got, err := loadTraceGuidance(path); err == nil || got != nil {
			t.Fatalf("invalid DB must fail closed: %v %v", got, err)
		}
	}
	for _, raw := range []string{`null`, `[]`, `{`, `{"block_size":[]}`} {
		db, path, session := traceSelectionFixture(t)
		addTraceSelectionRun(t, db, session, 1, "microbench", "size", "256", `{"block_size":64}`, "ok")
		addTraceSelectionRun(t, db, session, 1, "microbench", "size", "256", raw, "ok")
		if got, err := loadTraceGuidance(path); err == nil || got != nil {
			t.Fatalf("malformed row %s must not return partial guidance: %v %v", raw, got, err)
		}
	}
	db, path, _ := traceSelectionFixture(t)
	closedPath := filepath.Join(filepath.Dir(path), "results ? #.db")
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := loadSuccessfulConfigs(db); err == nil {
		t.Fatal("closed DB accepted")
	}
	if err := os.Rename(path, closedPath); err != nil {
		t.Fatal(err)
	}
	if got, err := loadTraceGuidance(closedPath); err != nil || got == nil {
		t.Fatalf("literal filename with URI characters: %v %v", got, err)
	}
}
