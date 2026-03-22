package main

// SCHEMA_SQL is the embedded SQLite schema from docs/schema.sql.
const SCHEMA_SQL = `
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS platforms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hostname TEXT NOT NULL, os TEXT NOT NULL,
    cpu_model TEXT, cpu_cores INTEGER, ram_gb REAL,
    gpu_model TEXT NOT NULL, gpu_vram_gb REAL,
    gpu_count INTEGER DEFAULT 1, driver_version TEXT,
    runtime_version TEXT, compiler TEXT,
    platform_type TEXT NOT NULL, python_version TEXT,
    extra_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_platform_fingerprint
    ON platforms (hostname, gpu_model, driver_version, os, platform_type);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    platform_id INTEGER NOT NULL REFERENCES platforms(id),
    command_line TEXT,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    repetitions INTEGER NOT NULL DEFAULT 7,
    filter_tier TEXT, filter_suite TEXT, filter_bench TEXT,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_session_platform ON sessions(platform_id);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    tier INTEGER NOT NULL, suite TEXT NOT NULL,
    benchmark TEXT NOT NULL,
    scaling_param_name TEXT NOT NULL,
    scaling_param_value TEXT NOT NULL,
    non_scaling_params_json TEXT NOT NULL DEFAULT '{}',
    repetition INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'ok',
    error_message TEXT, total_time_ms REAL,
    exit_code INTEGER,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_run_session ON runs(session_id);
CREATE INDEX IF NOT EXISTS idx_run_benchmark ON runs(tier, suite, benchmark);
CREATE INDEX IF NOT EXISTS idx_run_scaling ON runs(benchmark, scaling_param_name, scaling_param_value);
CREATE TABLE IF NOT EXISTS kernel_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    kernel_name TEXT NOT NULL, time_ms REAL NOT NULL,
    ordinal INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_kernel_run ON kernel_results(run_id);
CREATE INDEX IF NOT EXISTS idx_kernel_name ON kernel_results(kernel_name);
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    metric_name TEXT NOT NULL, metric_value REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metric_run ON metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_metric_name ON metrics(metric_name);
`
