-- =============================================================================
-- GPU Benchmark Suite — SQLite Results Schema
-- =============================================================================
-- This schema stores all benchmark execution results, platform metadata, and
-- derived metrics. Designed for analytical queries across platforms, parameter
-- sweeps, and repetitions.
--
-- Usage:
--   sqlite3 results.db < schema.sql
-- =============================================================================

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- -----------------------------------------------------------------------------
-- platforms — One row per unique hardware/software environment
-- -----------------------------------------------------------------------------
-- Captured once per execution session. If the CLI detects the same platform
-- fingerprint (hostname + gpu_model + driver_version + os), it reuses the row.

CREATE TABLE IF NOT EXISTS platforms (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    hostname        TEXT    NOT NULL,               -- machine hostname
    os              TEXT    NOT NULL,               -- e.g., "Linux 6.5.0-44-generic", "Darwin 24.1.0"
    cpu_model       TEXT,                           -- e.g., "AMD EPYC 7763 64-Core"
    cpu_cores       INTEGER,                        -- logical core count
    ram_gb          REAL,                           -- total system RAM in GiB
    gpu_model       TEXT    NOT NULL,               -- e.g., "NVIDIA A100-SXM4-80GB"
    gpu_vram_gb     REAL,                           -- GPU memory in GiB
    gpu_count       INTEGER DEFAULT 1,              -- number of GPUs (benchmark uses 1)
    driver_version  TEXT,                           -- e.g., "535.183.01", "3.4.1 Metal"
    runtime_version TEXT,                           -- e.g., "CUDA 12.2", "ROCm 6.0", "Metal 3"
    compiler        TEXT,                           -- e.g., "nvcc 12.2.140", "hipcc 6.0.2"
    platform_type   TEXT    NOT NULL,               -- "cuda", "rocm", or "metal"
    python_version  TEXT,                           -- e.g., "3.11.5" (for Tier 4/5)
    extra_json      TEXT,                           -- arbitrary JSON for additional platform info
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Unique constraint: same physical setup = same platform row
CREATE UNIQUE INDEX IF NOT EXISTS idx_platform_fingerprint
    ON platforms (hostname, gpu_model, driver_version, os, platform_type);

-- -----------------------------------------------------------------------------
-- sessions — One row per CLI invocation (gpubench run ...)
-- -----------------------------------------------------------------------------
-- Groups all runs from a single CLI invocation together.

CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    platform_id     INTEGER NOT NULL REFERENCES platforms(id),
    command_line    TEXT,                           -- full CLI command for reproducibility
    started_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    finished_at     TEXT,
    status          TEXT    NOT NULL DEFAULT 'running',  -- "running", "completed", "failed", "interrupted"
    repetitions     INTEGER NOT NULL DEFAULT 7,    -- requested repetition count
    filter_tier     TEXT,                           -- tier filter used, if any
    filter_suite    TEXT,                           -- suite filter used, if any
    filter_bench    TEXT,                           -- benchmark filter used, if any
    notes           TEXT                            -- free-form notes
);

CREATE INDEX IF NOT EXISTS idx_session_platform ON sessions(platform_id);

-- -----------------------------------------------------------------------------
-- runs — One row per (benchmark, parameter_config, repetition)
-- -----------------------------------------------------------------------------
-- A "run" is a single execution of one benchmark with one specific set of
-- parameter values and one repetition number.

CREATE TABLE IF NOT EXISTS runs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id              INTEGER NOT NULL REFERENCES sessions(id),
    tier                    INTEGER NOT NULL,               -- 1–5
    suite                   TEXT    NOT NULL,               -- e.g., "rodinia", "shoc", "ml_training"
    benchmark               TEXT    NOT NULL,               -- e.g., "rodinia_hotspot"
    scaling_param_name      TEXT    NOT NULL,               -- e.g., "grid_size"
    scaling_param_value     TEXT    NOT NULL,               -- stored as TEXT for generality
    non_scaling_params_json TEXT    NOT NULL DEFAULT '{}',  -- JSON object of {name: value}
    repetition              INTEGER NOT NULL,               -- 1-based repetition number
    status                  TEXT    NOT NULL DEFAULT 'ok',  -- "ok", "error", "timeout", "skipped"
    error_message           TEXT,                           -- error details if status != "ok"
    total_time_ms           REAL,                           -- from summary event
    exit_code               INTEGER,                       -- benchmark process exit code
    started_at              TEXT    NOT NULL DEFAULT (datetime('now')),
    finished_at             TEXT
);

CREATE INDEX IF NOT EXISTS idx_run_session    ON runs(session_id);
CREATE INDEX IF NOT EXISTS idx_run_benchmark  ON runs(tier, suite, benchmark);
CREATE INDEX IF NOT EXISTS idx_run_scaling    ON runs(benchmark, scaling_param_name, scaling_param_value);

-- -----------------------------------------------------------------------------
-- kernel_results — One row per kernel invocation within a run
-- -----------------------------------------------------------------------------
-- Stores kernel-level timing from "kernel" events in the output protocol.
-- Benchmarks that loop the measurement in-program (tier1) emit one event per
-- timed iteration, so a run holds num_iters rows per kernel, each with the
-- GPU event time, the matching host wall-clock time, and the iteration index.

CREATE TABLE IF NOT EXISTS kernel_results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    kernel_name  TEXT    NOT NULL,           -- e.g., "gemm_kernel", "forward", "backward"
    time_ms      REAL    NOT NULL,           -- GPU event time in milliseconds
    ordinal      INTEGER NOT NULL DEFAULT 0, -- order of emission (0-based)
    host_time_ms REAL,                       -- host wall-clock for the same iteration (launch + completion)
    iteration    INTEGER                     -- 0-based in-program iteration index, NULL if not emitted
);

CREATE INDEX IF NOT EXISTS idx_kernel_run  ON kernel_results(run_id);
CREATE INDEX IF NOT EXISTS idx_kernel_name ON kernel_results(kernel_name);

-- -----------------------------------------------------------------------------
-- metrics — Derived metrics from benchmark summary events
-- -----------------------------------------------------------------------------
-- Stores computed metrics like GFLOPS, bandwidth, images/sec.

CREATE TABLE IF NOT EXISTS metrics (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    metric_name  TEXT    NOT NULL,          -- e.g., "GFLOPS", "bandwidth_gbps", "images_per_sec"
    metric_value REAL    NOT NULL           -- the computed value
);

CREATE INDEX IF NOT EXISTS idx_metric_run  ON metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_metric_name ON metrics(metric_name);

-- =============================================================================
-- Example Queries
-- =============================================================================

-- 1. Show all kernel times for benchmark X across scaling parameter values
--
--    SELECT r.scaling_param_value, kr.kernel_name, kr.time_ms
--    FROM runs r
--    JOIN kernel_results kr ON kr.run_id = r.id
--    WHERE r.benchmark = 'polybench_gemm'
--      AND r.status = 'ok'
--    ORDER BY CAST(r.scaling_param_value AS INTEGER), kr.ordinal;

-- 2. Compare platform A vs platform B for the same benchmark
--
--    SELECT p.gpu_model, r.scaling_param_value,
--           AVG(kr.time_ms) AS avg_time_ms
--    FROM runs r
--    JOIN sessions s ON s.id = r.session_id
--    JOIN platforms p ON p.id = s.platform_id
--    JOIN kernel_results kr ON kr.run_id = r.id
--    WHERE r.benchmark = 'shoc_gemm'
--      AND r.status = 'ok'
--    GROUP BY p.gpu_model, r.scaling_param_value
--    ORDER BY CAST(r.scaling_param_value AS INTEGER);

-- 3. Get the average execution time across repetitions
--
--    SELECT r.benchmark, r.scaling_param_value,
--           AVG(r.total_time_ms) AS avg_total_ms,
--           MIN(r.total_time_ms) AS min_total_ms,
--           MAX(r.total_time_ms) AS max_total_ms,
--           COUNT(*) AS reps
--    FROM runs r
--    WHERE r.benchmark = 'rodinia_hotspot'
--      AND r.status = 'ok'
--    GROUP BY r.benchmark, r.scaling_param_value
--    ORDER BY CAST(r.scaling_param_value AS INTEGER);

-- 4. Get GFLOPS for a benchmark across scaling params
--
--    SELECT r.scaling_param_value,
--           AVG(m.metric_value) AS avg_gflops
--    FROM runs r
--    JOIN metrics m ON m.run_id = r.id
--    WHERE r.benchmark = 'fp32_throughput'
--      AND m.metric_name = 'GFLOPS'
--      AND r.status = 'ok'
--    GROUP BY r.scaling_param_value
--    ORDER BY CAST(r.scaling_param_value AS INTEGER);

-- 5. List all failed runs with error messages
--
--    SELECT r.tier, r.suite, r.benchmark,
--           r.scaling_param_value, r.error_message
--    FROM runs r
--    WHERE r.status != 'ok'
--    ORDER BY r.tier, r.suite, r.benchmark;

-- 6. Session summary: pass/fail counts per session
--
--    SELECT s.id AS session_id, s.started_at,
--           SUM(CASE WHEN r.status = 'ok' THEN 1 ELSE 0 END) AS passed,
--           SUM(CASE WHEN r.status != 'ok' THEN 1 ELSE 0 END) AS failed
--    FROM sessions s
--    JOIN runs r ON r.session_id = s.id
--    GROUP BY s.id;

-- 7. Full kernel breakdown for one run
--
--    SELECT kr.ordinal, kr.kernel_name, kr.time_ms
--    FROM kernel_results kr
--    WHERE kr.run_id = 42
--    ORDER BY kr.ordinal;
