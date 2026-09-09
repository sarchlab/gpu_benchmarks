-- =============================================================================
-- GPU Benchmark Suite — SQLite Results Schema
-- =============================================================================
-- This schema stores all benchmark execution results, platform metadata, and
-- optional legacy/higher-tier metrics. Tier 1/Tier 2 store raw GPU times and
-- workload descriptors; metrics are derived downstream. Queries span platforms, parameter
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
-- fingerprint (hostname, gpu_model, driver_version, os, platform_type,
-- runtime_version, compiler), it reuses the row. These are observed host metadata,
-- not proof of the selected device or the executable's build/runtime identity.

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
    runtime_version TEXT,                           -- NULL when the benchmark runtime is unobserved
    compiler        TEXT,                           -- installed nvcc/hipcc description, not build provenance
    platform_type   TEXT    NOT NULL,               -- "cuda", "rocm", or "metal"
    python_version  TEXT,                           -- e.g., "3.11.5" (for Tier 4/5)
    extra_json      TEXT,                           -- arbitrary JSON for additional platform info
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Version observations distinguish new sessions from legacy mislabeled rows.
-- Replace only the owned index; preserve historical rows and session links.
CREATE UNIQUE INDEX IF NOT EXISTS idx_platform_fingerprint_v2
    ON platforms (hostname, gpu_model, driver_version, os, platform_type,
                  runtime_version, compiler);
DROP INDEX IF EXISTS idx_platform_fingerprint;

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
-- parameter values and one repetition number. Each repetition starts a fresh
-- process and executes the complete algorithm once.

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
-- kernel_results — One accumulated record per kernel name within a valid run
-- -----------------------------------------------------------------------------
-- Stores kernel-level timing from "kernel" events in the output protocol.
-- Algorithmic loops accumulate all launches of a given kernel into one event.
-- Duplicate names are rejected by the runner for successful runs (the SQL table
-- itself does not enforce this, so partial failed runs may contain duplicates).
-- summary.total_time_ms is the sum of these records, excluding setup/verification.
-- Optional host_time_ms and iter fields are retained for compatibility; they do
-- not define repetitions or permit per-launch duplicate records.
-- Free-form info events are discarded by the parser; there is no info table.

CREATE TABLE IF NOT EXISTS kernel_results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    kernel_name  TEXT    NOT NULL,           -- e.g., "gemm_kernel", "forward", "backward"
    time_ms      REAL    NOT NULL,           -- accumulated GPU event time in milliseconds (Tier 1/Tier 2)
    ordinal      INTEGER NOT NULL DEFAULT 0, -- order of emission (0-based)
    host_time_ms REAL,                       -- optional compatibility timing; excluded from GPU total
    iteration    INTEGER,                    -- optional legacy event "iter" annotation, NULL if absent
    params_json  TEXT                        -- JSON of actual workload parameters; NULL if absent in failed/legacy output
);

CREATE INDEX IF NOT EXISTS idx_kernel_run  ON kernel_results(run_id);
CREATE INDEX IF NOT EXISTS idx_kernel_name ON kernel_results(kernel_name);

-- -----------------------------------------------------------------------------
-- metrics — Optional legacy/higher-tier summary metrics
-- -----------------------------------------------------------------------------
-- Stores metrics supplied by the producer, such as images/sec. The runner does
-- not derive them. Tier 1/Tier 2 omit summary metrics and derive them downstream.

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

-- 2. Compare the minimum total time for identical configurations by session
--    Keep sessions separate so builds, measurement controls, and hardware are
--    not accidentally pooled. Confirm equivalent provenance before comparison.
--
--    SELECT s.id AS session_id, p.gpu_model, r.benchmark,
--           r.scaling_param_name, r.scaling_param_value,
--           r.non_scaling_params_json,
--           MIN(r.total_time_ms) AS min_total_ms, COUNT(*) AS reps
--    FROM runs r
--    JOIN sessions s ON s.id = r.session_id
--    JOIN platforms p ON p.id = s.platform_id
--    WHERE r.benchmark = 'polybench_gemm' AND r.status = 'ok'
--    GROUP BY s.id, p.gpu_model, r.benchmark, r.scaling_param_name,
--             r.scaling_param_value, r.non_scaling_params_json
--    ORDER BY s.id, CAST(r.scaling_param_value AS INTEGER);

-- 3. Select the least-perturbed complete run and its kernel breakdown
--    Rank by the total; do not sum minima from different repetitions. A tie
--    selects the earliest run ID. JSON configuration strings below are those
--    serialized by gpubench; externally written databases may need normalization.
--
--    WITH ranked AS (
--        SELECT r.*, ROW_NUMBER() OVER (
--            PARTITION BY session_id, tier, suite, benchmark,
--                         scaling_param_name, scaling_param_value,
--                         non_scaling_params_json
--            ORDER BY total_time_ms, id
--        ) AS sample_rank
--        FROM runs r WHERE status = 'ok'
--    )
--    SELECT r.id, r.session_id, r.benchmark, r.scaling_param_value,
--           r.non_scaling_params_json, r.total_time_ms,
--           kr.kernel_name, kr.time_ms
--    FROM ranked r JOIN kernel_results kr ON kr.run_id = r.id
--    WHERE r.sample_rank = 1
--    ORDER BY r.id, kr.ordinal;

-- 4. Inspect optional higher-tier metrics without pooling configurations
--
--    SELECT r.id, r.session_id, r.scaling_param_value,
--           r.non_scaling_params_json, r.repetition, m.metric_value
--    FROM runs r JOIN metrics m ON m.run_id = r.id
--    WHERE r.benchmark = 'resnet50_train'
--      AND m.metric_name = 'images_per_sec' AND r.status = 'ok'
--    ORDER BY r.id;

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
