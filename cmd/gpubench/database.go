package main

import (
	"database/sql"
	"os"
	"strings"

	_ "github.com/mattn/go-sqlite3"
)

// openDatabase opens (or creates) the SQLite database and applies schema.
func openDatabase(dbPath string) *sql.DB {
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		eprintf("Database error: %v\n", err)
		os.Exit(ExitDB)
	}

	_, err = db.Exec(SCHEMA_SQL)
	if err != nil {
		eprintf("Database error: %v\n", err)
		os.Exit(ExitDB)
	}

	// Migrations for databases created before these columns existed.
	// SQLite has no ADD COLUMN IF NOT EXISTS; a duplicate-column error
	// just means the migration already ran.
	for _, mig := range []string{
		"ALTER TABLE kernel_results ADD COLUMN host_time_ms REAL",
		"ALTER TABLE kernel_results ADD COLUMN iteration INTEGER",
		"ALTER TABLE kernel_results ADD COLUMN params_json TEXT",
	} {
		if _, err := db.Exec(mig); err != nil &&
			!strings.Contains(err.Error(), "duplicate column name") {
			eprintf("Database error: %v\n", err)
			os.Exit(ExitDB)
		}
	}

	_, err = db.Exec("PRAGMA foreign_keys = ON")
	if err != nil {
		eprintf("Database error: %v\n", err)
		os.Exit(ExitDB)
	}

	return db
}

// getOrCreatePlatform returns the platform ID, creating a new row if needed.
func getOrCreatePlatform(db *sql.DB, info *PlatformInfo) int64 {
	var id int64
	err := db.QueryRow(
		"SELECT id FROM platforms WHERE hostname=? AND gpu_model=? "+
			"AND driver_version IS ? AND os=? AND platform_type=?",
		info.Hostname, info.GPUModel, ptrToNullString(info.DriverVersion),
		info.OS, info.PlatformType,
	).Scan(&id)
	if err == nil {
		return id
	}

	res, err := db.Exec(
		"INSERT INTO platforms "+
			"(hostname,os,cpu_model,cpu_cores,ram_gb,gpu_model,gpu_vram_gb,"+
			"gpu_count,driver_version,runtime_version,compiler,platform_type,"+
			"python_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
		info.Hostname, info.OS,
		ptrToNullString(info.CPUModel),
		ptrToNullInt(info.CPUCores),
		ptrToNullFloat(info.RAMGB),
		info.GPUModel,
		ptrToNullFloat(info.GPUVramGB),
		info.GPUCount,
		ptrToNullString(info.DriverVersion),
		ptrToNullString(info.RuntimeVersion),
		ptrToNullString(info.Compiler),
		info.PlatformType,
		info.PythonVersion,
	)
	if err != nil {
		eprintf("Database error: %v\n", err)
		os.Exit(ExitDB)
	}
	id, _ = res.LastInsertId()
	return id
}

// createSession creates a new session row.
func createSession(db *sql.DB, platformID int64, cmdLine string, reps int,
	tier, suite, benchmark string) int64 {
	var filterTier, filterSuite, filterBench sql.NullString
	if tier != "" {
		filterTier = sql.NullString{String: tier, Valid: true}
	}
	if suite != "" {
		filterSuite = sql.NullString{String: suite, Valid: true}
	}
	if benchmark != "" {
		filterBench = sql.NullString{String: benchmark, Valid: true}
	}

	res, err := db.Exec(
		"INSERT INTO sessions (platform_id,command_line,status,repetitions,"+
			"filter_tier,filter_suite,filter_bench) VALUES (?,?,?,?,?,?,?)",
		platformID, cmdLine, "running", reps,
		filterTier, filterSuite, filterBench,
	)
	if err != nil {
		eprintf("Database error: %v\n", err)
		os.Exit(ExitDB)
	}
	id, _ := res.LastInsertId()
	return id
}

// finishSession updates the session status.
func finishSession(db *sql.DB, sessionID int64, status string) {
	_, _ = db.Exec(
		"UPDATE sessions SET finished_at=datetime('now'), status=? WHERE id=?",
		status, sessionID,
	)
}

// helper functions for nullable types
func ptrToNullString(s *string) sql.NullString {
	if s == nil {
		return sql.NullString{}
	}
	return sql.NullString{String: *s, Valid: true}
}

func ptrToNullInt(i *int) sql.NullInt64 {
	if i == nil {
		return sql.NullInt64{}
	}
	return sql.NullInt64{Int64: int64(*i), Valid: true}
}

func ptrToNullFloat(f *float64) sql.NullFloat64 {
	if f == nil {
		return sql.NullFloat64{}
	}
	return sql.NullFloat64{Float64: *f, Valid: true}
}

// getCmdLine reconstructs the command line from os.Args.
func getCmdLine() string {
	return strings.Join(os.Args, " ")
}
