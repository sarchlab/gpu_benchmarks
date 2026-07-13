package main

import (
	"database/sql"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"testing"
)

// newExportTestDB creates a temporary SQLite database with the embedded schema
// applied and returns an open handle plus the file path.
func newExportTestDB(t *testing.T) (*sql.DB, string) {
	t.Helper()
	dbPath := filepath.Join(t.TempDir(), "results.db")
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	if _, err := db.Exec(SCHEMA_SQL); err != nil {
		t.Fatalf("apply schema: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	return db, dbPath
}

func mustExec(t *testing.T, db *sql.DB, query string, args ...interface{}) int64 {
	t.Helper()
	res, err := db.Exec(query, args...)
	if err != nil {
		t.Fatalf("exec %q: %v", query, err)
	}
	id, err := res.LastInsertId()
	if err != nil {
		t.Fatalf("last insert id: %v", err)
	}
	return id
}

// TestQueryPlatformPerSession verifies each session resolves the platform it
// actually ran on rather than always the first platform row.
func TestQueryPlatformPerSession(t *testing.T) {
	db, _ := newExportTestDB(t)

	p1 := mustExec(t, db,
		"INSERT INTO platforms (hostname, os, gpu_model, platform_type) VALUES (?,?,?,?)",
		"host-one", "Linux", "RTX 4090", "cuda")
	p2 := mustExec(t, db,
		"INSERT INTO platforms (hostname, os, gpu_model, platform_type) VALUES (?,?,?,?)",
		"host-two", "Linux", "MI300X", "rocm")

	mustExec(t, db, "INSERT INTO sessions (platform_id, status) VALUES (?, 'completed')", p1)
	mustExec(t, db, "INSERT INTO sessions (platform_id, status) VALUES (?, 'completed')", p2)

	sessions, err := querySessions(db, "")
	if err != nil {
		t.Fatalf("querySessions: %v", err)
	}
	if len(sessions) != 2 {
		t.Fatalf("got %d sessions, want 2", len(sessions))
	}

	for _, s := range sessions {
		plat, err := queryPlatform(db, s.platformID)
		if err != nil {
			t.Fatalf("queryPlatform(%d): %v", s.platformID, err)
		}
		switch s.platformID {
		case p1:
			if plat.Hostname != "host-one" || plat.PlatformType != "cuda" {
				t.Errorf("session on platform %d = %+v, want host-one/cuda", p1, plat)
			}
		case p2:
			if plat.Hostname != "host-two" || plat.PlatformType != "rocm" {
				t.Errorf("session on platform %d = %+v, want host-two/rocm", p2, plat)
			}
		default:
			t.Errorf("unexpected platformID %d", s.platformID)
		}
	}
}

// TestQueryRunsIncludesParams verifies scaling_param_name, non_scaling_params,
// repetition, and kernel params survive the export.
func TestQueryRunsIncludesParams(t *testing.T) {
	db, _ := newExportTestDB(t)

	pid := mustExec(t, db,
		"INSERT INTO platforms (hostname, os, gpu_model, platform_type) VALUES ('h','Linux','G','cuda')")
	sid := mustExec(t, db, "INSERT INTO sessions (platform_id, status) VALUES (?, 'completed')", pid)

	// Two runs identical except for non-scaling params and repetition.
	runA := mustExec(t, db,
		"INSERT INTO runs (session_id, tier, suite, benchmark, scaling_param_name, "+
			"scaling_param_value, non_scaling_params_json, repetition, status) "+
			"VALUES (?,1,'compute_bound','fp32_throughput','fmas_per_thread','4096',?,1,'ok')",
		sid, `{"threads_per_block":128}`)
	runB := mustExec(t, db,
		"INSERT INTO runs (session_id, tier, suite, benchmark, scaling_param_name, "+
			"scaling_param_value, non_scaling_params_json, repetition, status) "+
			"VALUES (?,1,'compute_bound','fp32_throughput','fmas_per_thread','4096',?,2,'ok')",
		sid, `{"threads_per_block":256}`)

	mustExec(t, db,
		"INSERT INTO kernel_results (run_id, kernel_name, time_ms, ordinal, host_time_ms, iteration, params_json) "+
			"VALUES (?,'fp32_fma_kernel',0.5,0,0.6,0,?)",
		runA, `{"num_blocks":1}`)

	runs, err := queryRuns(db, sid)
	if err != nil {
		t.Fatalf("queryRuns: %v", err)
	}
	if len(runs) != 2 {
		t.Fatalf("got %d runs, want 2", len(runs))
	}

	if runs[0].ScalingParamName != "fmas_per_thread" {
		t.Errorf("scaling_param_name = %q, want fmas_per_thread", runs[0].ScalingParamName)
	}

	// The two runs must remain distinguishable by non-scaling params + repetition.
	if string(runs[0].NonScalingParams) == string(runs[1].NonScalingParams) {
		t.Errorf("non_scaling_params identical across runs: %s", runs[0].NonScalingParams)
	}
	if runs[0].Repetition == runs[1].Repetition {
		t.Errorf("repetition identical across runs: %d", runs[0].Repetition)
	}
	_ = runB

	// Kernel params must be emitted as a nested object.
	if len(runs[0].Kernels) != 1 {
		t.Fatalf("run A got %d kernels, want 1", len(runs[0].Kernels))
	}
	var kp map[string]interface{}
	if err := json.Unmarshal(runs[0].Kernels[0].Params, &kp); err != nil {
		t.Fatalf("kernel params not valid JSON: %v", err)
	}
	if kp["num_blocks"].(float64) != 1 {
		t.Errorf("kernel params = %v, want num_blocks=1", kp)
	}
}

// TestNonScalingParamsFallback ensures empty/malformed stored JSON still yields
// well-formed output.
func TestNonScalingParamsFallback(t *testing.T) {
	if got := string(jsonObjectOrEmpty("")); got != "{}" {
		t.Errorf("empty = %q, want {}", got)
	}
	if got := string(jsonObjectOrEmpty("not json")); got != "{}" {
		t.Errorf("malformed = %q, want {}", got)
	}
	if got := string(jsonObjectOrEmpty(`{"a":1}`)); got != `{"a":1}` {
		t.Errorf("valid = %q, want {\"a\":1}", got)
	}
}

// TestCmdExportInvalidSession verifies invalid session IDs fail loudly rather
// than returning exit 0 with an empty session list.
func TestCmdExportInvalidSession(t *testing.T) {
	db, dbPath := newExportTestDB(t)
	pid := mustExec(t, db,
		"INSERT INTO platforms (hostname, os, gpu_model, platform_type) VALUES ('h','Linux','G','cuda')")
	mustExec(t, db, "INSERT INTO sessions (platform_id, status) VALUES (?, 'completed')", pid)

	t.Run("nonnumeric", func(t *testing.T) {
		if code := runExportCapture(t, dbPath, "json", "abc"); code != ExitUsage {
			t.Errorf("exit code = %d, want %d (ExitUsage)", code, ExitUsage)
		}
	})

	t.Run("nonexistent id", func(t *testing.T) {
		if code := runExportCapture(t, dbPath, "json", "999"); code != ExitDB {
			t.Errorf("exit code = %d, want %d (ExitDB)", code, ExitDB)
		}
	})

	t.Run("existing id", func(t *testing.T) {
		if code := runExportCapture(t, dbPath, "json", "1"); code != ExitOK {
			t.Errorf("exit code = %d, want %d (ExitOK)", code, ExitOK)
		}
	})
}

// TestCmdExportOutputShape runs the full command and checks the emitted JSON
// carries the per-session platform and the documented contract.
func TestCmdExportOutputShape(t *testing.T) {
	db, dbPath := newExportTestDB(t)
	pid := mustExec(t, db,
		"INSERT INTO platforms (hostname, os, gpu_model, platform_type) VALUES ('rig','Linux','RTX 4090','cuda')")
	sid := mustExec(t, db, "INSERT INTO sessions (platform_id, status) VALUES (?, 'completed')", pid)
	mustExec(t, db,
		"INSERT INTO runs (session_id, tier, suite, benchmark, scaling_param_name, "+
			"scaling_param_value, non_scaling_params_json, repetition, status) "+
			"VALUES (?,1,'compute_bound','fp32_throughput','fmas_per_thread','4096','{}',1,'ok')",
		sid)

	out, code := captureStdout(t, func() int { return cmdExport(dbPath, "json", "") })
	if code != ExitOK {
		t.Fatalf("cmdExport exit = %d, want ExitOK", code)
	}

	var parsed ExportOutput
	if err := json.Unmarshal([]byte(out), &parsed); err != nil {
		t.Fatalf("output not valid JSON: %v\n%s", err, out)
	}
	if parsed.ExportVersion != "1.1" {
		t.Errorf("export_version = %q, want 1.1", parsed.ExportVersion)
	}
	if len(parsed.Sessions) != 1 {
		t.Fatalf("got %d sessions, want 1", len(parsed.Sessions))
	}
	if parsed.Sessions[0].Platform.Hostname != "rig" {
		t.Errorf("session platform hostname = %q, want rig", parsed.Sessions[0].Platform.Hostname)
	}
	if len(parsed.Sessions[0].Runs) != 1 {
		t.Fatalf("got %d runs, want 1", len(parsed.Sessions[0].Runs))
	}
}

// runExportCapture runs cmdExport, discarding stdout, and returns the exit code.
func runExportCapture(t *testing.T, dbPath, format, session string) int {
	t.Helper()
	_, code := captureStdout(t, func() int { return cmdExport(dbPath, format, session) })
	return code
}

// captureStdout redirects os.Stdout for the duration of fn and returns what it
// wrote along with fn's return value.
func captureStdout(t *testing.T, fn func() int) (string, int) {
	t.Helper()
	orig := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	os.Stdout = w

	done := make(chan string)
	go func() {
		data, _ := io.ReadAll(r)
		done <- string(data)
	}()

	code := fn()
	w.Close()
	os.Stdout = orig
	return <-done, code
}
