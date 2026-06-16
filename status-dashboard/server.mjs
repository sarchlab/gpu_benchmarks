import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { stat, readFile, readdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");
const dbPath = process.env.GPUBENCH_DB || path.join(repoRoot, "results.db");
const port = Number(process.env.PORT || 4174);
const distDir = path.join(__dirname, "dist");
const apiOnly = process.env.STATUS_DASHBOARD_API_ONLY === "1";

function runSql(query) {
  return new Promise((resolve, reject) => {
    const child = spawn("sqlite3", ["-cmd", ".timeout 5000", "-json", dbPath, query], {
      cwd: repoRoot,
      stdio: ["ignore", "pipe", "pipe"]
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(stderr || `sqlite3 exited with ${code}`));
        return;
      }
      try {
        resolve(stdout.trim() ? JSON.parse(stdout) : []);
      } catch (error) {
        reject(error);
      }
    });
  });
}

function runText(command, args, { allowNoMatch = false } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: repoRoot,
      stdio: ["ignore", "pipe", "pipe"]
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("close", (code) => {
      if (code === 0 || (allowNoMatch && code === 1)) {
        resolve(stdout);
        return;
      }
      reject(new Error(stderr || `${command} exited with ${code}`));
    });
  });
}

async function getGpubenchProcesses() {
  const output = await runText("pgrep", ["-af", "gpubench run"], { allowNoMatch: true });
  return output
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .filter((line) => !line.includes("pgrep -af"));
}

async function fileSize(file) {
  try {
    return (await stat(file)).size;
  } catch {
    return 0;
  }
}

function sqlString(value) {
  return `'${String(value).replaceAll("'", "''")}'`;
}

function sortValues(left, right) {
  const leftNumber = Number(left);
  const rightNumber = Number(right);
  if (Number.isFinite(leftNumber) && Number.isFinite(rightNumber)) {
    return leftNumber - rightNumber;
  }
  return String(left).localeCompare(String(right), undefined, { numeric: true });
}

function uniqueSorted(values) {
  return [...new Set(values.map((value) => String(value)))].sort(sortValues);
}

function shellWords(commandLine) {
  return String(commandLine || "").match(/(?:[^\s"']+|"[^"]*"|'[^']*')+/g)?.map((word) => (
    word.replace(/^(['"])(.*)\1$/, "$2")
  )) || [];
}

function commandFlagValue(commandLine, flag) {
  const words = shellWords(commandLine);
  for (let index = 0; index < words.length; index += 1) {
    const word = words[index];
    if (word === flag) return words[index + 1] || "";
    if (word.startsWith(`${flag}=`)) return word.slice(flag.length + 1);
  }
  return "";
}

function commandHasDefaultNonScalingSweep(commandLine) {
  const explicit = commandFlagValue(commandLine, "--sweep-non-scaling");
  if (explicit) return explicit !== "false";
  return !shellWords(commandLine).includes("--sweep-non-scaling=false");
}

async function loadBenchmarkSpecs() {
  const tierDirs = (await readdir(repoRoot, { withFileTypes: true }))
    .filter((entry) => entry.isDirectory() && /^tier\d+$/.test(entry.name))
    .map((entry) => entry.name)
    .sort(sortValues);
  const specs = [];
  for (const tierDir of tierDirs) {
    const tierPath = path.join(repoRoot, tierDir);
    const benchmarkDirs = await readdir(tierPath, { withFileTypes: true });
    for (const entry of benchmarkDirs) {
      if (!entry.isDirectory()) continue;
      const paramsPath = path.join(tierPath, entry.name, "params.json");
      if (!existsSync(paramsPath)) continue;
      const params = JSON.parse(await readFile(paramsPath, "utf8"));
      specs.push({
        benchmark: params.benchmark || entry.name,
        suite: params.suite || "",
        tier: Number(params.tier ?? tierDir.slice("tier".length)),
        scalingValues: params.scaling_param?.values || [],
        nonScalingParams: params.non_scaling_params || []
      });
    }
  }
  return specs;
}

async function expectedBenchmarksForSession(session) {
  if (!session) {
    return { rows: [], total: null, title: "All Benchmarks" };
  }

  const scalingOverride = commandFlagValue(session.command_line, "--scaling-values")
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  const sweepNonScaling = commandHasDefaultNonScalingSweep(session.command_line);
  const specs = (await loadBenchmarkSpecs()).filter((spec) => (
    (session.filter_tier == null || String(spec.tier) === String(session.filter_tier)) &&
    (session.filter_suite == null || spec.suite === session.filter_suite) &&
    (session.filter_bench == null || spec.benchmark === session.filter_bench)
  ));

  const reps = Number(session.repetitions || 1);
  const rows = specs.map((spec) => {
    const scalingCount = scalingOverride.length || spec.scalingValues.length;
    const nonScalingCount = sweepNonScaling
      ? spec.nonScalingParams.reduce((product, param) => product * Math.max(1, param.values?.length || 0), 1)
      : 1;
    return {
      benchmark: spec.benchmark,
      suite: spec.suite,
      tier: spec.tier,
      expectedRuns: scalingCount * nonScalingCount * reps
    };
  });
  const total = rows.reduce((sum, row) => sum + row.expectedRuns, 0);
  const title = session.filter_bench
    ? session.filter_bench
    : session.filter_suite
      ? `${session.filter_suite} Suite`
      : session.filter_tier
        ? `Tier ${session.filter_tier}`
        : "All Benchmarks";
  return { rows, total, title };
}

async function getStatus() {
  if (!existsSync(dbPath)) {
    return {
      dbPath,
      exists: false,
      generatedAt: new Date().toISOString(),
      sessions: [],
      benchmarks: [],
      totals: {
        runs: 0,
        ok: 0,
        errors: 0,
        timeouts: 0,
        expectedRuns: null,
        percent: 0,
        elapsedSeconds: null,
        runRate: null,
        remainingRuns: null,
        etaSeconds: null,
        title: "All Benchmarks"
      },
      runtime: {
        dbSessionStatus: null,
        gpubenchRunning: false,
        gpubenchProcesses: [],
        health: "idle",
        staleSeconds: null,
        lastFinishedAt: null
      },
      storage: {
        dbBytes: 0,
        walBytes: 0,
        shmBytes: 0,
        totalBytes: 0
      }
    };
  }

  const [sessions, tableRows, gpubenchProcesses] = await Promise.all([
    runSql(`
      SELECT id, started_at, finished_at, status, command_line, repetitions, filter_tier,
             filter_suite, filter_bench
      FROM sessions
      ORDER BY id DESC
      LIMIT 5;
    `),
    runSql(`
      SELECT
        (SELECT COUNT(*) FROM runs) AS run_rows,
        (SELECT COUNT(*) FROM kernel_results) AS kernel_rows,
        (SELECT COUNT(*) FROM metrics) AS metric_rows;
    `),
    getGpubenchProcesses()
  ]);

  const activeSession = sessions.find((session) => session.status === "running") || sessions[0];
  const sessionWhere = activeSession ? `WHERE session_id = ${Number(activeSession.id)}` : "";
  const [totalsRows, observedBenchmarks] = await Promise.all([
    runSql(`
      SELECT
        COUNT(*) AS runs,
        COALESCE(SUM(status='ok'), 0) AS ok,
        COALESCE(SUM(status='error'), 0) AS errors,
        COALESCE(SUM(status='timeout'), 0) AS timeouts,
        MIN(started_at) AS first_started_at,
        MAX(finished_at) AS last_finished_at
      FROM runs
      ${sessionWhere};
    `),
    runSql(`
      SELECT
        tier,
        suite,
        benchmark,
        COUNT(*) AS runs,
        COUNT(DISTINCT scaling_param_value) AS points,
        COALESCE(SUM(status='ok'), 0) AS ok,
        COALESCE(SUM(status='error'), 0) AS errors,
        COALESCE(SUM(status='timeout'), 0) AS timeouts,
        MIN(started_at) AS first_started_at,
        MAX(finished_at) AS last_finished_at
      FROM runs
      ${sessionWhere}
      GROUP BY tier, suite, benchmark
      ORDER BY MIN(id);
    `)
  ]);

  const totals = totalsRows[0] || {};
  const expected = await expectedBenchmarksForSession(activeSession);
  const observedByBenchmark = new Map(observedBenchmarks.map((row) => [row.benchmark, row]));
  const benchmarks = expected.rows.length
    ? expected.rows.map((row) => {
        const observed = observedByBenchmark.get(row.benchmark) || {};
        return {
          ...row,
          runs: Number(observed.runs || 0),
          points: Number(observed.points || 0),
          ok: Number(observed.ok || 0),
          errors: Number(observed.errors || 0),
          timeouts: Number(observed.timeouts || 0),
          first_started_at: observed.first_started_at || null,
          last_finished_at: observed.last_finished_at || null
        };
      })
    : observedBenchmarks.map((row) => ({
        ...row,
        tier: Number(row.tier),
        runs: Number(row.runs || 0),
        points: Number(row.points || 0),
        ok: Number(row.ok || 0),
        errors: Number(row.errors || 0),
        timeouts: Number(row.timeouts || 0),
        expectedRuns: null
      }));
  const expectedRuns = expected.total;
  const runs = Number(totals.runs || 0);
  const firstStart = totals.first_started_at ? Date.parse(`${totals.first_started_at}Z`) : null;
  const elapsedSeconds = firstStart ? Math.max(1, (Date.now() - firstStart) / 1000) : null;
  const runRate = elapsedSeconds ? runs / elapsedSeconds : null;
  const remainingRuns = expectedRuns == null ? null : Math.max(0, expectedRuns - runs);
  const etaSeconds = runRate && remainingRuns != null ? remainingRuns / runRate : null;
  const lastFinish = totals.last_finished_at ? Date.parse(`${totals.last_finished_at}Z`) : null;
  const gpubenchRunning = gpubenchProcesses.length > 0;
  const dbSessionStatus = activeSession?.status ?? null;
  const health = dbSessionStatus === "running"
    ? gpubenchRunning ? "running" : "stale"
    : dbSessionStatus === "completed" || dbSessionStatus === "complete" ? "complete" : "idle";
  const staleSeconds = health === "stale"
    ? Math.max(0, (Date.now() - (lastFinish || firstStart || Date.now())) / 1000)
    : null;
  const dbBytes = await fileSize(dbPath);
  const walBytes = await fileSize(`${dbPath}-wal`);
  const shmBytes = await fileSize(`${dbPath}-shm`);

  return {
    dbPath,
    exists: true,
    generatedAt: new Date().toISOString(),
    sessions,
    benchmarks,
    tableRows: tableRows[0] || { run_rows: 0, kernel_rows: 0, metric_rows: 0 },
    totals: {
      runs,
      ok: Number(totals.ok || 0),
      errors: Number(totals.errors || 0),
      timeouts: Number(totals.timeouts || 0),
      expectedRuns,
      percent: expectedRuns ? (runs / expectedRuns) * 100 : 0,
      elapsedSeconds,
      runRate,
      remainingRuns,
      etaSeconds,
      title: expected.title
    },
    runtime: {
      dbSessionStatus,
      gpubenchRunning,
      gpubenchProcesses,
      health,
      staleSeconds,
      lastFinishedAt: totals.last_finished_at || null
    },
    storage: {
      dbBytes,
      walBytes,
      shmBytes,
      totalBytes: dbBytes + walBytes + shmBytes
    }
  };
}

async function getResultOptions(benchmark) {
  const where = benchmark ? `WHERE benchmark = ${sqlString(benchmark)}` : "";
  const [benchmarks, scalingRows, scalingValueRows, repetitionRows, keyRows, kernelRows] = await Promise.all([
    runSql(`
      SELECT benchmark, suite, tier, COUNT(*) AS runs
      FROM runs
      GROUP BY benchmark, suite, tier
      ORDER BY tier, suite, benchmark;
    `),
    runSql(`
      SELECT DISTINCT scaling_param_name
      FROM runs
      ${where}
      ORDER BY scaling_param_name;
    `),
    runSql(`
      SELECT DISTINCT scaling_param_value
      FROM runs
      ${where}
      ORDER BY scaling_param_value;
    `),
    runSql(`
      SELECT DISTINCT repetition
      FROM runs
      ${where}
      ORDER BY repetition;
    `),
    runSql(`
      SELECT DISTINCT non_scaling_params_json
      FROM runs
      ${where}
      LIMIT 5000;
    `),
    runSql(`
      SELECT DISTINCT k.kernel_name
      FROM kernel_results k
      JOIN runs r ON r.id = k.run_id
      ${benchmark ? `WHERE r.benchmark = ${sqlString(benchmark)}` : ""}
      ORDER BY k.kernel_name;
    `)
  ]);

  const keys = new Set();
  for (const row of keyRows) {
    try {
      const parsed = JSON.parse(row.non_scaling_params_json || "{}");
      for (const key of Object.keys(parsed)) {
        keys.add(key);
      }
    } catch {
      // Ignore malformed legacy rows; the chart can still use base fields.
    }
  }

  const nonScalingFields = [...keys].sort().map((key) => ({
    id: `param:${key}`,
    label: key
  }));
  const filterValues = {};
  for (const key of keys) {
    filterValues[key] = new Set();
  }
  for (const row of keyRows) {
    try {
      const parsed = JSON.parse(row.non_scaling_params_json || "{}");
      for (const key of keys) {
        if (parsed[key] != null) {
          filterValues[key].add(String(parsed[key]));
        }
      }
    } catch {
      // Ignore malformed legacy rows.
    }
  }
  const filters = [...keys].sort().map((key) => ({
    id: `param:${key}`,
    key,
    label: key,
    values: [...filterValues[key]].sort(sortValues)
  }));
  const scalingNames = scalingRows.map((row) => row.scaling_param_name).filter(Boolean);
  const scalingLabel = scalingNames.length === 1
    ? `${scalingNames[0]} (scaling)`
    : "Scaling parameter value";
  const scalingField = { id: "scaling_param_value", label: scalingLabel };
  const xFields = [{ id: "scaling_param_value", label: scalingLabel }, ...nonScalingFields];
  const seriesFields = [...nonScalingFields];
  if (kernelRows.length) {
    seriesFields.push({ id: "kernel_name", label: "Kernel name" });
  }
  const properties = [
    {
      ...scalingField,
      kind: "scaling",
      values: uniqueSorted(scalingValueRows.map((row) => row.scaling_param_value)),
      defaultChannel: "x",
      allowedChannels: ["x", "color", "aggregate"]
    },
    ...filters.map((filter) => ({
      id: filter.id,
      label: filter.label,
      kind: "parameter",
      values: filter.values,
      defaultChannel: "color",
      allowedChannels: ["x", "color", "aggregate"]
    })),
    {
      id: "repetition",
      label: "repetition",
      kind: "sample",
      values: uniqueSorted(repetitionRows.map((row) => row.repetition)),
      defaultChannel: "aggregate",
      allowedChannels: ["color", "aggregate"]
    }
  ];
  if (kernelRows.length) {
    properties.push({
      id: "kernel_name",
      label: "Kernel name",
      kind: "kernel",
      values: kernelRows.map((row) => row.kernel_name),
      defaultChannel: "aggregate",
      allowedChannels: ["color", "aggregate"]
    });
  }
  properties.push({
    id: "time_ms",
    label: "execution time",
    kind: "metric",
    values: [],
    defaultChannel: "y",
    allowedChannels: ["y"]
  });

  return {
    benchmarks,
    xFields,
    seriesFields,
    fields: [...xFields, ...seriesFields.filter((field) => !xFields.some((xField) => xField.id === field.id))],
    filters,
    properties,
    kernels: kernelRows.map((row) => row.kernel_name)
  };
}

function valueForField(row, field) {
  if (!field || field === "none") return "all";
  if (field.startsWith("param:")) {
    const key = field.slice("param:".length);
    return row.params[key] == null ? "(unset)" : String(row.params[key]);
  }
  return row[field] == null || row[field] === "" ? "(unset)" : String(row[field]);
}

async function getChartData(queryParams) {
  const benchmark = queryParams.get("benchmark");
  if (!benchmark) {
    return { benchmark: null, xField: null, colorFields: [], yMode: "total", filters: {}, points: [], series: [] };
  }

  const channelEntries = [...queryParams.entries()]
    .filter(([key]) => key.startsWith("channel:"))
    .map(([key, value]) => [key.slice("channel:".length), value]);
  const xField = channelEntries.find(([, channel]) => channel === "x")?.[0] || "scaling_param_value";
  const colorFields = channelEntries
    .filter(([, channel]) => channel === "color")
    .map(([field]) => field)
    .filter((field) => field !== xField && field !== "time_ms");
  const filters = {};
  for (const key of new Set([...queryParams.keys()].filter((param) => param.startsWith("filter:")))) {
    const field = key.slice("filter:".length);
    const values = queryParams.getAll(key).filter(Boolean);
    if (values.length) {
      filters[field] = values;
    }
  }
  const usesKernelName = xField === "kernel_name" || colorFields.includes("kernel_name") || Boolean(filters.kernel_name?.length);
  const yMode = usesKernelName ? "kernel" : "total";
  const includeKernelRows = usesKernelName;
  const rows = await runSql(`
    WITH kernel_totals AS (
      SELECT run_id, SUM(time_ms) AS total_kernel_time_ms
      FROM kernel_results
      GROUP BY run_id
    )
    SELECT
      r.id,
      r.scaling_param_name,
      r.scaling_param_value,
      r.non_scaling_params_json,
      r.repetition,
      r.status,
      COALESCE(kt.total_kernel_time_ms, r.total_time_ms) AS total_time_ms,
      ${includeKernelRows ? "k.kernel_name" : "NULL AS kernel_name"},
      ${includeKernelRows ? "k.time_ms" : "NULL"} AS kernel_time_ms
    FROM runs r
    LEFT JOIN kernel_totals kt ON kt.run_id = r.id
    ${includeKernelRows ? "LEFT JOIN kernel_results k ON k.run_id = r.id" : ""}
    WHERE r.benchmark = ${sqlString(benchmark)}
      AND r.status = 'ok'
      AND COALESCE(kt.total_kernel_time_ms, r.total_time_ms) IS NOT NULL
    ORDER BY r.id${includeKernelRows ? ", k.ordinal" : ""};
  `);

  const grouped = new Map();
  for (const row of rows) {
    if (yMode === "kernel" && row.kernel_time_ms == null) continue;

    let params = {};
    try {
      params = JSON.parse(row.non_scaling_params_json || "{}");
    } catch {
      params = {};
    }

    const flat = {
      ...row,
      params,
      time_ms: yMode === "kernel" ? Number(row.kernel_time_ms) : Number(row.total_time_ms)
    };
    const filterMatches = Object.entries(filters).every(([field, values]) => (
      values.includes(valueForField(flat, field))
    ));
    if (!filterMatches) continue;

    const x = valueForField(flat, xField);
    const series = colorFields.length
      ? colorFields.map((field) => `${field.startsWith("param:") ? field.slice("param:".length) : field}=${valueForField(flat, field)}`).join(", ")
      : "all";
    const key = `${x}\u0000${series}`;
    const bucket = grouped.get(key) || { x, series, values: [] };
    bucket.values.push(flat.time_ms);
    grouped.set(key, bucket);
  }

  const points = [...grouped.values()].map((bucket) => {
    const values = bucket.values;
    const avgMs = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance = values.length > 1
      ? values.reduce((sum, value) => sum + (value - avgMs) ** 2, 0) / (values.length - 1)
      : 0;
    const stddevMs = Math.sqrt(variance);
    const stderrMs = values.length > 0 ? stddevMs / Math.sqrt(values.length) : 0;
    const ci95Ms = values.length > 1 ? 1.96 * stderrMs : 0;
    return {
      x: bucket.x,
      series: bucket.series,
      avgMs,
      minMs: Math.min(...values),
      maxMs: Math.max(...values),
      stddevMs,
      stderrMs,
      ci95Ms,
      ciLowMs: Math.max(0, avgMs - ci95Ms),
      ciHighMs: avgMs + ci95Ms,
      count: values.length
    };
  }).sort((left, right) => {
    const xCompare = sortValues(left.x, right.x);
    return xCompare || sortValues(left.series, right.series);
  });

  return {
    benchmark,
    xField,
    colorFields,
    yMode,
    filters,
    points,
    series: [...new Set(points.map((point) => point.series))].sort(sortValues)
  };
}

function sendJson(res, code, body) {
  res.writeHead(code, {
    "content-type": "application/json",
    "cache-control": "no-store"
  });
  res.end(JSON.stringify(body));
}

async function serveStatic(req, res) {
  if (apiOnly) {
    res.writeHead(404);
    res.end("Not found");
    return;
  }

  const requestPath = new URL(req.url || "/", "http://localhost").pathname;
  const safePath = requestPath === "/" ? "index.html" : requestPath.slice(1);
  const filePath = path.normalize(path.join(distDir, safePath));
  if (!filePath.startsWith(distDir)) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }
  const target = existsSync(filePath) ? filePath : path.join(distDir, "index.html");
  try {
    const data = await readFile(target);
    const ext = path.extname(target);
    const contentType = {
      ".html": "text/html",
      ".js": "text/javascript",
      ".css": "text/css",
      ".svg": "image/svg+xml"
    }[ext] || "application/octet-stream";
    res.writeHead(200, { "content-type": contentType });
    res.end(data);
  } catch {
    res.writeHead(404);
    res.end("Not found");
  }
}

createServer(async (req, res) => {
  const url = new URL(req.url || "/", "http://localhost");
  if (url.pathname === "/api/status") {
    try {
      sendJson(res, 200, await getStatus());
    } catch (error) {
      sendJson(res, 500, { error: error instanceof Error ? error.message : String(error) });
    }
    return;
  }
  if (url.pathname === "/api/results/options") {
    try {
      sendJson(res, 200, await getResultOptions(url.searchParams.get("benchmark")));
    } catch (error) {
      sendJson(res, 500, { error: error instanceof Error ? error.message : String(error) });
    }
    return;
  }
  if (url.pathname === "/api/results/chart") {
    try {
      sendJson(res, 200, await getChartData(url.searchParams));
    } catch (error) {
      sendJson(res, 500, { error: error instanceof Error ? error.message : String(error) });
    }
    return;
  }
  if (url.pathname === "/api/download") {
    try {
      if (!existsSync(dbPath)) {
        res.writeHead(404, { "content-type": "text/plain" });
        res.end("results.db not found");
        return;
      }
      const data = await readFile(dbPath);
      res.writeHead(200, {
        "content-type": "application/x-sqlite3",
        "content-disposition": 'attachment; filename="results.db"',
        "content-length": data.length,
        "cache-control": "no-store"
      });
      res.end(data);
    } catch (error) {
      sendJson(res, 500, { error: error instanceof Error ? error.message : String(error) });
    }
    return;
  }
  await serveStatic(req, res);
}).listen(port, "127.0.0.1", () => {
  const mode = apiOnly ? "API server" : "API/static server";
  console.log(`status dashboard ${mode}: http://127.0.0.1:${port}`);
});
