package main

import (
	"database/sql"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
)

func TestPlatformVersionSources(t *testing.T) {
	for _, platform := range []string{"cuda", "rocm"} {
		for _, installed := range []bool{true, false} {
			name := platform + "/missing-tools"
			if installed {
				name = platform + "/installed-tools"
			}
			t.Run(name, func(t *testing.T) {
				bin := t.TempDir()
				scripts := map[string]string{
					"nvcc":  "printf '%s\\n' 'Cuda compilation tools, release 12.9, V12.9.86'\n",
					"hipcc": "printf '%s\\n' 'HIP version: 6.3.42131'\n",
					"nvidia-smi": `case "$1" in
--query-gpu=name) printf '%s\n' 'Test GPU';;
--query-gpu=memory.total) printf '%s\n' '81920';;
--query-gpu=driver_version) printf '%s\n' '580.178.04';;
*) exit 1;;
esac
`,
					"rocm-smi": "printf '%s\\n' 'GPU[0] : Card series: Test AMD GPU'\n",
					// Reproduce the legacy ROCm-release/driver conflation even
					// on hosts without /opt/rocm. Other cat calls remain real.
					"cat": `if [ "$1" = /opt/rocm/.info/version ]; then
printf '%s\n' '6.3.0'
else
exec /bin/cat "$@"
fi
`,
				}
				for command, script := range scripts {
					if !installed {
						script = "exit 1\n"
					}
					if err := os.WriteFile(filepath.Join(bin, command), []byte("#!/bin/sh\n"+script), 0755); err != nil {
						t.Fatal(err)
					}
				}
				t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
				platformCache, platformOnce = nil, sync.Once{}
				t.Cleanup(func() { platformCache, platformOnce = nil, sync.Once{} })
				info := collectPlatformInfo(platform)
				if info.RuntimeVersion != nil {
					t.Fatalf("compiler output mislabeled as runtime: %q", *info.RuntimeVersion)
				}
				if platform == "cuda" && installed {
					if info.DriverVersion == nil || *info.DriverVersion != "580.178.04" {
						t.Fatalf("actual NVIDIA driver query lost: %+v", info.DriverVersion)
					}
				} else if info.DriverVersion != nil {
					t.Fatalf("unobserved driver must be unknown: %q", *info.DriverVersion)
				}
				if installed {
					want := "HIP version: 6.3.42131"
					if platform == "cuda" {
						want = "Cuda compilation tools, release 12.9, V12.9.86"
					}
					if info.Compiler == nil || *info.Compiler != want {
						t.Fatalf("installed compiler = %v, want %q", info.Compiler, want)
					}
				} else if info.Compiler != nil {
					t.Fatalf("missing compiler must be unknown: %q", *info.Compiler)
				}
			})
		}
	}
}

func TestPlatformVersionMigrationPreservesHistory(t *testing.T) {
	for _, platform := range []string{"cuda", "rocm"} {
		t.Run(platform, func(t *testing.T) {
			db, path := newExportTestDB(t)
			// Recreate the actual old index: the table columns are unchanged.
			mustExec(t, db, "DROP INDEX idx_platform_fingerprint_v2")
			mustExec(t, db, `CREATE UNIQUE INDEX idx_platform_fingerprint
ON platforms(hostname,gpu_model,driver_version,os,platform_type)`)
			oldDriver, compiler := "580.178.04", "nvcc release 12.9"
			if platform == "rocm" {
				oldDriver, compiler = "6.3.0", "HIP version: 6.3.42131"
			}
			oldID := mustExec(t, db, `INSERT INTO platforms
(hostname,os,gpu_model,driver_version,runtime_version,compiler,platform_type)
VALUES('host','Linux','GPU',?,?,?,?)`, oldDriver, compiler, compiler, platform)
			oldSession := mustExec(t, db, "INSERT INTO sessions(platform_id) VALUES(?)", oldID)
			if err := db.Close(); err != nil {
				t.Fatal(err)
			}

			db = openDatabase(path)
			info := &PlatformInfo{Hostname: "host", OS: "Linux", GPUModel: "GPU", PlatformType: platform, Compiler: strPtr(compiler)}
			if platform == "cuda" {
				info.DriverVersion = strPtr(oldDriver)
			}
			newID := getOrCreatePlatform(db, info)
			if newID == oldID || getOrCreatePlatform(db, info) != newID {
				t.Fatal("corrected observations must have a new, reusable row")
			}
			newSession := createSession(db, newID, "test", 1, "2", "", "")
			info.Compiler = strPtr("different installed compiler")
			compilerID := getOrCreatePlatform(db, info)
			if compilerID == newID || getOrCreatePlatform(db, info) != compilerID {
				t.Fatal("different compiler observation must have a distinct, reusable row")
			}
			info.RuntimeVersion = strPtr("independently observed runtime fixture")
			runtimeID := getOrCreatePlatform(db, info)
			if runtimeID == compilerID || getOrCreatePlatform(db, info) != runtimeID {
				t.Fatal("different runtime observation must have a distinct, reusable row")
			}
			if err := db.Close(); err != nil {
				t.Fatal(err)
			}

			db = openDatabase(path)
			defer db.Close()
			if getOrCreatePlatform(db, info) != runtimeID {
				t.Fatal("reopening must preserve corrected row identity")
			}
			var oldRuntime, oldCompiler, driver string
			if err := db.QueryRow("SELECT runtime_version,compiler,driver_version FROM platforms WHERE id=?", oldID).Scan(&oldRuntime, &oldCompiler, &driver); err != nil {
				t.Fatal(err)
			}
			if oldRuntime != compiler || oldCompiler != compiler || driver != oldDriver {
				t.Fatal("legacy metadata was rewritten")
			}
			var runtime, currentDriver sql.NullString
			if err := db.QueryRow("SELECT runtime_version,driver_version FROM platforms WHERE id=?", newID).Scan(&runtime, &currentDriver); err != nil {
				t.Fatal(err)
			}
			if runtime.Valid || (platform == "rocm" && currentDriver.Valid) {
				t.Fatal("unobserved versions must be SQL NULL")
			}
			for sid, want := range map[int64]int64{oldSession: oldID, newSession: newID} {
				var actual int64
				if err := db.QueryRow("SELECT platform_id FROM sessions WHERE id=?", sid).Scan(&actual); err != nil || actual != want {
					t.Fatalf("session association changed: %d -> %d, want %d; %v", sid, actual, want, err)
				}
			}
			var oldIndexes int
			if err := db.QueryRow("SELECT count(*) FROM sqlite_master WHERE type='index' AND name='idx_platform_fingerprint'").Scan(&oldIndexes); err != nil || oldIndexes != 0 {
				t.Fatalf("old index survived migration: %d; %v", oldIndexes, err)
			}
		})
	}
}

func TestPlatformFingerprintSchemaAgreement(t *testing.T) {
	document, err := os.ReadFile("../../docs/schema.sql")
	if err != nil {
		t.Fatal(err)
	}
	for name, schema := range map[string]string{"embedded": SCHEMA_SQL, "documented": string(document)} {
		t.Run(name, func(t *testing.T) {
			db, err := sql.Open("sqlite3", ":memory:")
			if err != nil {
				t.Fatal(err)
			}
			defer db.Close()
			if _, err := db.Exec(schema); err != nil {
				t.Fatal(err)
			}
			rows, err := db.Query("PRAGMA index_info(idx_platform_fingerprint_v2)")
			if err != nil {
				t.Fatal(err)
			}
			defer rows.Close()
			var columns []string
			for rows.Next() {
				var sequence, columnID int
				var column string
				if err := rows.Scan(&sequence, &columnID, &column); err != nil {
					t.Fatal(err)
				}
				columns = append(columns, column)
			}
			if err := rows.Err(); err != nil {
				t.Fatal(err)
			}
			want := strings.Fields("hostname gpu_model driver_version os platform_type runtime_version compiler")
			if !reflect.DeepEqual(columns, want) {
				t.Fatalf("fingerprint columns = %v, want %v", columns, want)
			}
		})
	}
}
