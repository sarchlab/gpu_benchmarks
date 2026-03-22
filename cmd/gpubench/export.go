package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"time"
)

// ── Export JSON types ─────────────────────────────────────────────────────

// ExportOutput is the top-level JSON structure for export.
type ExportOutput struct {
	ExportVersion string          `json:"export_version"`
	ExportedAt    string          `json:"exported_at"`
	Platform      ExportPlatform  `json:"platform"`
	Sessions      []ExportSession `json:"sessions"`
}

// ExportPlatform holds platform information in the export.
type ExportPlatform struct {
	Hostname      string   `json:"hostname"`
	OS            string   `json:"os"`
	GPUModel      string   `json:"gpu_model"`
	GPUVramGB     *float64 `json:"gpu_vram_gb"`
	DriverVersion *string  `json:"driver_version"`
	PlatformType  string   `json:"platform_type"`
}

// ExportSession holds a single session's data.
type ExportSession struct {
	ID          int64       `json:"id"`
	StartedAt   string      `json:"started_at"`
	FinishedAt  *string     `json:"finished_at"`
	Status      string      `json:"status"`
	Repetitions int         `json:"repetitions"`
	Runs        []ExportRun `json:"runs"`
}

// ExportRun holds a single benchmark run.
type ExportRun struct {
	Benchmark         string         `json:"benchmark"`
	Tier              int            `json:"tier"`
	Suite             string         `json:"suite"`
	ScalingParamValue string         `json:"scaling_param_value"`
	Status            string         `json:"status"`
	Kernels           []ExportKernel `json:"kernels"`
	Metrics           []ExportMetric `json:"metrics"`
}

// ExportKernel holds kernel timing data.
type ExportKernel struct {
	Name   string  `json:"name"`
	TimeMS float64 `json:"time_ms"`
}

// ExportMetric holds a named metric value.
type ExportMetric struct {
	Name  string  `json:"name"`
	Value float64 `json:"value"`
}

// ── Command: export ───────────────────────────────────────────────────────

func cmdExport(dbPath, format, session string) int {
	if format != "json" {
		eprintf("Unsupported format: %s (only 'json' is supported)\n", format)
		return ExitUsage
	}

	if _, err := os.Stat(dbPath); os.IsNotExist(err) {
		eprintf("Database file not found: %s\n", dbPath)
		return ExitDB
	}

	db, err := sql.Open("sqlite3", dbPath+"?mode=ro")
	if err != nil {
		eprintf("Database error: %v\n", err)
		return ExitDB
	}
	defer db.Close()

	// Read platform info (use first platform row)
	platform, err := queryPlatform(db)
	if err != nil {
		eprintf("Error reading platform data: %v\n", err)
		return ExitDB
	}

	// Read sessions
	sessions, err := querySessions(db, session)
	if err != nil {
		eprintf("Error reading sessions: %v\n", err)
		return ExitDB
	}

	// Build runs for each session
	for i := range sessions {
		runs, err := queryRuns(db, sessions[i].ID)
		if err != nil {
			eprintf("Error reading runs for session %d: %v\n", sessions[i].ID, err)
			return ExitDB
		}
		sessions[i].Runs = runs
	}

	output := ExportOutput{
		ExportVersion: "1.0",
		ExportedAt:    time.Now().UTC().Format(time.RFC3339),
		Platform:      platform,
		Sessions:      sessions,
	}

	data, err := json.MarshalIndent(output, "", "  ")
	if err != nil {
		eprintf("JSON encoding error: %v\n", err)
		return ExitSomeFailed
	}

	fmt.Println(string(data))
	return ExitOK
}

// ── Database query helpers for export ─────────────────────────────────────

func queryPlatform(db *sql.DB) (ExportPlatform, error) {
	var p ExportPlatform
	var gpuVram sql.NullFloat64
	var driverVersion sql.NullString

	err := db.QueryRow(
		"SELECT hostname, os, gpu_model, gpu_vram_gb, driver_version, platform_type " +
			"FROM platforms ORDER BY id LIMIT 1",
	).Scan(&p.Hostname, &p.OS, &p.GPUModel, &gpuVram, &driverVersion, &p.PlatformType)
	if err != nil {
		return p, fmt.Errorf("no platform data found: %w", err)
	}

	if gpuVram.Valid {
		p.GPUVramGB = &gpuVram.Float64
	}
	if driverVersion.Valid {
		p.DriverVersion = &driverVersion.String
	}

	return p, nil
}

func querySessions(db *sql.DB, sessionFilter string) ([]ExportSession, error) {
	query := "SELECT id, started_at, finished_at, status, repetitions FROM sessions"
	var args []interface{}

	if sessionFilter != "" {
		query += " WHERE id = ?"
		args = append(args, sessionFilter)
	}
	query += " ORDER BY id"

	rows, err := db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var sessions []ExportSession
	for rows.Next() {
		var s ExportSession
		var finishedAt sql.NullString
		err := rows.Scan(&s.ID, &s.StartedAt, &finishedAt, &s.Status, &s.Repetitions)
		if err != nil {
			return nil, err
		}
		if finishedAt.Valid {
			s.FinishedAt = &finishedAt.String
		}
		s.Runs = []ExportRun{} // initialize to empty slice for JSON
		sessions = append(sessions, s)
	}

	if sessions == nil {
		sessions = []ExportSession{}
	}
	return sessions, rows.Err()
}

func queryRuns(db *sql.DB, sessionID int64) ([]ExportRun, error) {
	rows, err := db.Query(
		"SELECT id, benchmark, tier, suite, scaling_param_value, status "+
			"FROM runs WHERE session_id = ? ORDER BY id",
		sessionID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var runs []ExportRun
	for rows.Next() {
		var r ExportRun
		var runID int64
		err := rows.Scan(&runID, &r.Benchmark, &r.Tier, &r.Suite,
			&r.ScalingParamValue, &r.Status)
		if err != nil {
			return nil, err
		}

		// Query kernels for this run
		kernels, err := queryKernels(db, runID)
		if err != nil {
			return nil, err
		}
		r.Kernels = kernels

		// Query metrics for this run
		metrics, err := queryMetrics(db, runID)
		if err != nil {
			return nil, err
		}
		r.Metrics = metrics

		runs = append(runs, r)
	}

	if runs == nil {
		runs = []ExportRun{}
	}
	return runs, rows.Err()
}

func queryKernels(db *sql.DB, runID int64) ([]ExportKernel, error) {
	rows, err := db.Query(
		"SELECT kernel_name, time_ms FROM kernel_results WHERE run_id = ? ORDER BY ordinal",
		runID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var kernels []ExportKernel
	for rows.Next() {
		var k ExportKernel
		if err := rows.Scan(&k.Name, &k.TimeMS); err != nil {
			return nil, err
		}
		kernels = append(kernels, k)
	}

	if kernels == nil {
		kernels = []ExportKernel{}
	}
	return kernels, rows.Err()
}

func queryMetrics(db *sql.DB, runID int64) ([]ExportMetric, error) {
	rows, err := db.Query(
		"SELECT metric_name, metric_value FROM metrics WHERE run_id = ? ORDER BY id",
		runID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var metrics []ExportMetric
	for rows.Next() {
		var m ExportMetric
		if err := rows.Scan(&m.Name, &m.Value); err != nil {
			return nil, err
		}
		metrics = append(metrics, m)
	}

	if metrics == nil {
		metrics = []ExportMetric{}
	}
	return metrics, rows.Err()
}
