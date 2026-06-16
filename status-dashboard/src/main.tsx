import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  ChevronDown,
  Clock,
  Database,
  Download,
  Gauge,
  RefreshCw,
  TimerReset
} from "lucide-react";
import "./styles.css";

type Session = {
  id: number;
  started_at: string | null;
  finished_at: string | null;
  status: string;
  command_line: string | null;
  repetitions: number;
  filter_tier: string | number | null;
  filter_suite: string | null;
  filter_bench: string | null;
};

type Benchmark = {
  tier: number;
  suite: string;
  benchmark: string;
  expectedRuns: number | null;
  runs: number;
  points: number;
  ok: number;
  errors: number;
  timeouts: number;
  first_started_at: string | null;
  last_finished_at: string | null;
};

type Status = {
  dbPath: string;
  exists: boolean;
  generatedAt: string;
  sessions: Session[];
  benchmarks: Benchmark[];
  tableRows?: {
    run_rows: number;
    kernel_rows: number;
    metric_rows: number;
  };
  totals: {
    runs: number;
    ok: number;
    errors: number;
    timeouts: number;
    expectedRuns: number | null;
    percent: number;
    elapsedSeconds: number | null;
    runRate: number | null;
    remainingRuns: number | null;
    etaSeconds: number | null;
    title: string;
  };
  runtime: {
    dbSessionStatus: string | null;
    gpubenchRunning: boolean;
    gpubenchProcesses: string[];
    health: "running" | "stale" | "idle" | "complete";
    staleSeconds: number | null;
    lastFinishedAt: string | null;
  };
  storage: {
    dbBytes: number;
    walBytes: number;
    shmBytes: number;
    totalBytes: number;
  };
};

type FieldOption = {
  id: string;
  label: string;
};

type ResultOptions = {
  benchmarks: Array<{
    benchmark: string;
    suite: string;
    tier: number;
    runs: number;
  }>;
  fields: FieldOption[];
  xFields: FieldOption[];
  seriesFields: FieldOption[];
  filters: Array<{
    id: string;
    key: string;
    label: string;
    values: string[];
  }>;
  properties: Array<{
    id: string;
    label: string;
    kind: string;
    values: string[];
    defaultChannel: Channel;
    allowedChannels: Channel[];
  }>;
  kernels: string[];
};

type Channel = "x" | "y" | "color" | "aggregate";

type ChartPoint = {
  x: string;
  series: string;
  avgMs: number;
  minMs: number;
  maxMs: number;
  stddevMs: number;
  stderrMs: number;
  ci95Ms: number;
  ciLowMs: number;
  ciHighMs: number;
  count: number;
};

type ChartData = {
  benchmark: string | null;
  xField: string | null;
  colorFields: string[];
  yMode: "total" | "kernel";
  filters: Record<string, string[]>;
  points: ChartPoint[];
  series: string[];
};

type GlobalTab = "session" | "explorer";

const emptyResultOptions: ResultOptions = {
  benchmarks: [],
  fields: [],
  xFields: [],
  seriesFields: [],
  filters: [],
  properties: [],
  kernels: []
};

const validChannels: Channel[] = ["x", "y", "color", "aggregate"];

function isChannel(value: unknown): value is Channel {
  return typeof value === "string" && validChannels.includes(value as Channel);
}

function normalizeResultOptions(value: Partial<ResultOptions> | null | undefined): ResultOptions {
  return {
    ...emptyResultOptions,
    benchmarks: Array.isArray(value?.benchmarks) ? value.benchmarks : [],
    fields: Array.isArray(value?.fields) ? value.fields : [],
    xFields: Array.isArray(value?.xFields) ? value.xFields : [],
    seriesFields: Array.isArray(value?.seriesFields) ? value.seriesFields : [],
    filters: Array.isArray(value?.filters) ? value.filters : [],
    properties: Array.isArray(value?.properties)
      ? value.properties.map((property) => {
          const allowedChannels = Array.isArray(property.allowedChannels)
            ? property.allowedChannels.filter(isChannel)
            : [];
          const defaultChannel = isChannel(property.defaultChannel)
            ? property.defaultChannel
            : allowedChannels[0] || "aggregate";
          return {
            ...property,
            values: Array.isArray(property.values) ? property.values : [],
            defaultChannel,
            allowedChannels: allowedChannels.length ? allowedChannels : [defaultChannel]
          };
        })
      : [],
    kernels: Array.isArray(value?.kernels) ? value.kernels : []
  };
}

function cn(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(" ");
}

function FilterDropdown({
  values,
  selected,
  disabled,
  onChange
}: {
  values: string[];
  selected: string[];
  disabled: boolean;
  onChange: (selected: string[]) => void;
}) {
  const filterValues = values;
  const selectedValues = selected;
  const effectiveSelected = selectedValues.length ? selectedValues : filterValues;
  const selectedSet = useMemo(() => new Set(effectiveSelected), [effectiveSelected]);
  const allSelected = filterValues.length > 0 && effectiveSelected.length === filterValues.length;
  const summary = !filterValues.length ? "No values" : allSelected ? "All values" : `${effectiveSelected.length} selected`;

  function toggleValue(value: string) {
    const nextSelected = selectedSet.has(value)
      ? effectiveSelected.filter((selectedValue) => selectedValue !== value)
      : [...effectiveSelected, value];
    const nextUniqueSelected = filterValues.filter((filterValue) => nextSelected.includes(filterValue));
    if (nextUniqueSelected.length === filterValues.length) {
      onChange([]);
      return;
    }
    onChange(nextUniqueSelected);
  }

  function selectAll() {
    onChange([]);
  }

  if (disabled) {
    return (
      <div className="filter-dropdown filter-dropdown-disabled">
        <span>{summary}</span>
        <ChevronDown size={16} />
      </div>
    );
  }

  return (
    <details className="filter-dropdown">
      <summary>
        <span>{summary}</span>
        <ChevronDown size={16} />
      </summary>
      <div className="filter-menu">
        {!allSelected && (
          <button className="filter-clear" type="button" onClick={selectAll}>
            Select all
          </button>
        )}
        <div className="filter-options">
          {filterValues.map((value) => (
            <label className="filter-option" key={value}>
              <input type="checkbox" checked={selectedSet.has(value)} onChange={() => toggleValue(value)} />
              <span>{value}</span>
            </label>
          ))}
        </div>
      </div>
    </details>
  );
}

function formatNumber(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return "n/a";
  return new Intl.NumberFormat("en-US").format(Math.round(value));
}

function formatPercent(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return "0.0%";
  return `${value.toFixed(1)}%`;
}

function formatBytes(bytes: number) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function formatDuration(seconds: number | null | undefined) {
  if (seconds == null || !Number.isFinite(seconds)) return "n/a";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${Math.round(seconds % 60)}s`;
  const hours = Math.floor(minutes / 60);
  if (hours < 48) return `${hours}h ${minutes % 60}m`;
  const days = Math.floor(hours / 24);
  return `${days}d ${hours % 24}h`;
}

function formatDate(value: string | null | undefined) {
  if (!value) return "n/a";
  const date = new Date(value.endsWith("Z") ? value : `${value}Z`);
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  });
}

function StatCard({
  label,
  value,
  detail,
  icon
}: {
  label: string;
  value: string;
  detail: string;
  icon: React.ReactNode;
}) {
  return (
    <div className="card stat-card">
      <div className="stat-icon">{icon}</div>
      <div>
        <p className="muted">{label}</p>
        <div className="stat-value">{value}</div>
        <p className="fine-print">{detail}</p>
      </div>
    </div>
  );
}

function ProgressBar({ value }: { value: number }) {
  return (
    <div className="progress">
      <div className="progress-fill" style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
    </div>
  );
}

const colors = ["#0f766e", "#2563eb", "#b45309", "#7c3aed", "#dc2626", "#0891b2", "#4d7c0f", "#be185d"];

function LineChart({ data }: { data: ChartData | null }) {
  const points = data?.points || [];
  const seriesNames = data?.series || [];
  const xValues = [...new Set(points.map((point) => point.x))];
  const width = 760;
  const height = 320;
  const padding = { top: 24, right: 28, bottom: 58, left: 66 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const maxY = Math.max(...points.map((point) => point.ciHighMs ?? point.avgMs), 0);
  const yTop = maxY > 0 ? maxY * 1.08 : 1;
  const xPosition = (x: string) => {
    if (xValues.length <= 1) return padding.left + plotWidth / 2;
    return padding.left + (xValues.indexOf(x) / (xValues.length - 1)) * plotWidth;
  };
  const yPosition = (value: number) => padding.top + plotHeight - (value / yTop) * plotHeight;

  if (!points.length) {
    return (
      <div className="empty-chart">
        <BarChart3 size={22} />
        <span>No chart data for this selection yet.</span>
      </div>
    );
  }

  return (
    <div className="chart-wrap">
      <svg className="line-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Execution time line chart">
        <line x1={padding.left} y1={padding.top} x2={padding.left} y2={padding.top + plotHeight} className="axis" />
        <line
          x1={padding.left}
          y1={padding.top + plotHeight}
          x2={padding.left + plotWidth}
          y2={padding.top + plotHeight}
          className="axis"
        />
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
          const y = padding.top + plotHeight - tick * plotHeight;
          return (
            <g key={tick}>
              <line x1={padding.left} y1={y} x2={padding.left + plotWidth} y2={y} className="grid-line" />
              <text x={padding.left - 10} y={y + 4} textAnchor="end" className="axis-label">
                {(yTop * tick).toFixed(yTop >= 10 ? 0 : 3)}
              </text>
            </g>
          );
        })}
        {xValues.map((x, index) => {
          const shouldShow = xValues.length <= 10 || index % Math.ceil(xValues.length / 10) === 0;
          if (!shouldShow) return null;
          return (
            <text key={x} x={xPosition(x)} y={height - 24} textAnchor="middle" className="axis-label">
              {x.length > 12 ? `${x.slice(0, 11)}...` : x}
            </text>
          );
        })}
        {seriesNames.map((seriesName, seriesIndex) => {
          const seriesPoints = points.filter((point) => point.series === seriesName);
          const areaPoints = seriesPoints.filter((point) => point.count > 1 && point.ci95Ms > 0);
          const areaPath = areaPoints.length
            ? [
                ...areaPoints.map((point, index) => `${index === 0 ? "M" : "L"} ${xPosition(point.x)} ${yPosition(point.ciHighMs)}`),
                ...[...areaPoints]
                  .reverse()
                  .map((point) => `L ${xPosition(point.x)} ${yPosition(point.ciLowMs)}`),
                "Z"
              ].join(" ")
            : "";
          const path = seriesPoints
            .map((point, index) => `${index === 0 ? "M" : "L"} ${xPosition(point.x)} ${yPosition(point.avgMs)}`)
            .join(" ");
          const color = colors[seriesIndex % colors.length];
          return (
            <g key={seriesName}>
              {areaPath && <path d={areaPath} fill={color} className="ci-band" />}
              <path d={path} fill="none" stroke={color} strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />
              {seriesPoints.map((point) => (
                <g key={`${point.x}-${seriesName}`}>
                  {point.count > 1 && point.ci95Ms > 0 && (
                    <>
                      <line
                        x1={xPosition(point.x)}
                        y1={yPosition(point.ciLowMs)}
                        x2={xPosition(point.x)}
                        y2={yPosition(point.ciHighMs)}
                        stroke={color}
                        className="ci-line"
                      />
                      <line
                        x1={xPosition(point.x) - 4}
                        y1={yPosition(point.ciLowMs)}
                        x2={xPosition(point.x) + 4}
                        y2={yPosition(point.ciLowMs)}
                        stroke={color}
                        className="ci-line"
                      />
                      <line
                        x1={xPosition(point.x) - 4}
                        y1={yPosition(point.ciHighMs)}
                        x2={xPosition(point.x) + 4}
                        y2={yPosition(point.ciHighMs)}
                        stroke={color}
                        className="ci-line"
                      />
                    </>
                  )}
                  <circle cx={xPosition(point.x)} cy={yPosition(point.avgMs)} r="3.5" fill={color}>
                    <title>
                      {`${seriesName} @ ${point.x}: mean ${point.avgMs.toFixed(6)} ms, 95% CI +/- ${point.ci95Ms.toFixed(6)} ms (${point.count} samples)`}
                    </title>
                  </circle>
                </g>
              ))}
            </g>
          );
        })}
      </svg>
      <div className="legend">
        {seriesNames.map((seriesName, index) => (
          <span className="legend-item" key={seriesName}>
            <span className="legend-swatch" style={{ background: colors[index % colors.length] }} />
            {seriesName}
          </span>
        ))}
      </div>
    </div>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState<GlobalTab>("session");
  const [status, setStatus] = useState<Status | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [options, setOptions] = useState<ResultOptions | null>(null);
  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [selectedBenchmark, setSelectedBenchmark] = useState("");
  const [channels, setChannels] = useState<Record<string, Channel>>({});
  const [filters, setFilters] = useState<Record<string, string[]>>({});

  async function refresh() {
    setLoading(true);
    try {
      const response = await fetch("/api/status", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setStatus(await response.json());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
    const id = window.setInterval(refresh, 5000);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    async function loadOptions() {
      const url = selectedBenchmark
        ? `/api/results/options?benchmark=${encodeURIComponent(selectedBenchmark)}`
        : "/api/results/options";
      const response = await fetch(url, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const nextOptions = normalizeResultOptions(await response.json());
      setOptions(nextOptions);
      if (!selectedBenchmark && nextOptions.benchmarks.length) {
        setSelectedBenchmark(nextOptions.benchmarks[0].benchmark);
      }
      setChannels((current) => {
        if (Object.keys(current).length) return current;
        return Object.fromEntries(
          nextOptions.properties.map((property: ResultOptions["properties"][number]) => [
            property.id,
            property.defaultChannel
          ])
        );
      });
    }
    loadOptions().catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [selectedBenchmark]);

  useEffect(() => {
    setFilters({});
    setChannels({});
  }, [selectedBenchmark]);

  useEffect(() => {
    if (!selectedBenchmark) return;
    async function loadChart() {
      const params = new URLSearchParams({ benchmark: selectedBenchmark });
      for (const [property, channel] of Object.entries(channels)) {
        params.set(`channel:${property}`, channel);
      }
      for (const [property, values] of Object.entries(filters)) {
        for (const value of values) {
          params.append(`filter:${property}`, value);
        }
      }
      const response = await fetch(`/api/results/chart?${params.toString()}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setChartData(await response.json());
    }
    loadChart().catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [selectedBenchmark, channels, filters, status?.totals.runs]);

  const activeSession = status?.sessions.find((session) => session.status === "running") || status?.sessions[0];

  const benchmarkRows = useMemo(() => {
    return (status?.benchmarks || []).map((bench) => {
      const expected = bench.expectedRuns;
      const runs = bench.runs || 0;
      return {
        name: bench.benchmark,
        expected,
        runs,
        points: bench.points || 0,
        ok: bench.ok || 0,
        errors: bench.errors || 0,
        timeouts: bench.timeouts || 0,
        percent: expected ? (runs / expected) * 100 : 0
      };
    });
  }, [status]);

  const currentBenchmark =
    benchmarkRows.find((row) => row.expected && row.runs > 0 && row.runs < row.expected)?.name ||
    [...benchmarkRows].reverse().find((row) => row.runs > 0)?.name ||
    "waiting";
  const statusTitle = status?.totals.title || "Benchmarks";
  const health = status?.runtime?.health || activeSession?.status || "offline";
  const sessionDetail = activeSession
    ? `Started ${formatDate(activeSession.started_at)} · DB ${status?.runtime?.dbSessionStatus || activeSession.status} · process ${
        status?.runtime?.gpubenchRunning ? "found" : "not found"
      }`
    : "Waiting for results.db";
  const xChannel = Object.entries(channels).find(([, channel]) => channel === "x")?.[0] || "scaling_param_value";

  function setPropertyChannel(property: ResultOptions["properties"][number], channel: Channel) {
    setChannels((current) => {
      const next = { ...current, [property.id]: channel };
      if (channel === "x") {
        for (const key of Object.keys(next)) {
          if (key !== property.id && next[key] === "x") {
            next[key] = "aggregate";
          }
        }
      }
      if (!Object.values(next).includes("x")) {
        next.scaling_param_value = "x";
      }
      return next;
    });
  }

  function updateFilter(propertyId: string, selected: string[]) {
    setFilters((current) => {
      const next = { ...current };
      if (selected.length) {
        next[propertyId] = selected;
      } else {
        delete next[propertyId];
      }
      return next;
    });
  }

  function selectTab(tab: GlobalTab) {
    setActiveTab(tab);
  }

  return (
    <main className="page">
      <section className="topbar">
        <div>
          <p className="eyebrow">GPU Benchmarks</p>
          <h1>Status Dashboard</h1>
        </div>
        <div className="topbar-actions">
          <a className="button" href="/api/download" download title="Download results.db">
            <Download size={16} />
            Download DB
          </a>
          <button className="button" onClick={refresh} disabled={loading} title="Refresh now">
            <RefreshCw size={16} className={cn(loading && "spin")} />
            Refresh
          </button>
        </div>
      </section>

      <nav className="global-tabs" role="tablist" aria-label="Dashboard sections">
        <button
          className={cn("global-tab", activeTab === "session" && "global-tab-active")}
          type="button"
          role="tab"
          aria-selected={activeTab === "session"}
          onClick={() => selectTab("session")}
        >
          <Activity size={16} />
          Current Session
        </button>
        <button
          className={cn("global-tab", activeTab === "explorer" && "global-tab-active")}
          type="button"
          role="tab"
          aria-selected={activeTab === "explorer"}
          onClick={() => selectTab("explorer")}
        >
          <BarChart3 size={16} />
          Data Explorer
        </button>
      </nav>

      {error && (
        <div className="alert">
          <AlertTriangle size={18} />
          <span>{error}</span>
        </div>
      )}

      {activeTab === "session" && (
        <div role="tabpanel" aria-label="Current session">
          <section className="summary">
            <div className="card hero-card">
              <div className="hero-row">
                <div>
                  <p className="muted">Current run</p>
                  <h2>{activeSession ? `Session ${activeSession.id}` : "No session"}</h2>
                  <p className="fine-print">{sessionDetail}</p>
                  {status?.runtime?.health === "stale" && (
                    <p className="fine-print warning-text">
                      No `gpubench run` process seen for {formatDuration(status.runtime.staleSeconds)}.
                    </p>
                  )}
                </div>
                <span className={cn("badge", health === "running" && "badge-live", health === "stale" && "badge-stale")}>
                  {health}
                </span>
              </div>
              <div className="headline-progress">
                <div>
                  <div className="big-number">{formatPercent(status?.totals.percent)}</div>
                  <p className="muted">Current session progress</p>
                </div>
                <div className="progress-wrap">
                  <ProgressBar value={status?.totals.percent || 0} />
                  <p className="fine-print">
                    {formatNumber(status?.totals.runs)} / {formatNumber(status?.totals.expectedRuns)} runs
                  </p>
                </div>
              </div>
            </div>

            <div className="stat-grid">
              <StatCard
                label="Current benchmark"
                value={currentBenchmark}
                detail={`${formatNumber(status?.totals.ok)} successful rows`}
                icon={<Activity size={18} />}
              />
              <StatCard
                label="ETA"
                value={formatDuration(status?.totals.etaSeconds)}
                detail={`${(status?.totals.runRate || 0).toFixed(2)} runs/sec`}
                icon={<Clock size={18} />}
              />
              <StatCard
                label="Elapsed"
                value={formatDuration(status?.totals.elapsedSeconds)}
                detail={`${formatNumber(status?.totals.remainingRuns)} runs remaining`}
                icon={<TimerReset size={18} />}
              />
              <StatCard
                label="Dataset size"
                value={formatBytes(status?.storage.totalBytes || 0)}
                detail={`DB ${formatBytes(status?.storage.dbBytes || 0)} · WAL ${formatBytes(status?.storage.walBytes || 0)}`}
                icon={<Database size={18} />}
              />
            </div>
          </section>

          <section className="grid-two">
            <div className="card">
              <div className="section-head">
                <div>
                  <p className="muted">Benchmarks</p>
                  <h2>{statusTitle} Sweep</h2>
                </div>
                <Gauge size={20} />
              </div>
              <div className="bench-list">
                {benchmarkRows.map((row) => (
                  <div className="bench-row" key={row.name}>
                    <div className="bench-top">
                      <span>{row.name}</span>
                      <span className="fine-print">
                        {formatNumber(row.runs)} / {formatNumber(row.expected)}
                      </span>
                    </div>
                    <ProgressBar value={row.percent} />
                    <div className="bench-meta">
                      <span>{formatPercent(row.percent)}</span>
                      <span>{row.points} scaling points</span>
                      <span>{row.errors + row.timeouts} failed</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="card">
              <div className="section-head">
                <div>
                  <p className="muted">Storage</p>
                  <h2>Recorded Tables</h2>
                </div>
                <BarChart3 size={20} />
              </div>
              <div className="table-card">
                <div className="table-row">
                  <span>runs</span>
                  <strong>{formatNumber(status?.tableRows?.run_rows)}</strong>
                </div>
                <div className="table-row">
                  <span>kernel_results</span>
                  <strong>{formatNumber(status?.tableRows?.kernel_rows)}</strong>
                </div>
                <div className="table-row">
                  <span>metrics</span>
                  <strong>{formatNumber(status?.tableRows?.metric_rows)}</strong>
                </div>
              </div>
              <div className="note">
                <p className="fine-print">Auto-refreshes every 5 seconds from {status?.dbPath || "results.db"}.</p>
                <p className="fine-print">ETA is based on the current observed insert rate, so it will move as benchmarks change.</p>
              </div>
            </div>
          </section>
        </div>
      )}

      {activeTab === "explorer" && (
        <section className="card results-card" role="tabpanel" aria-label="Data explorer">
          <div className="section-head">
            <div>
              <p className="muted">Collected Results</p>
              <h2>Execution Time Explorer</h2>
            </div>
            <BarChart3 size={20} />
          </div>
          <div className="control-grid">
            <label>
              <span>Benchmark</span>
              <select value={selectedBenchmark} onChange={(event) => setSelectedBenchmark(event.target.value)}>
                {options?.benchmarks.map((bench) => (
                  <option key={bench.benchmark} value={bench.benchmark}>
                    {bench.benchmark} ({formatNumber(bench.runs)})
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="mapping-panel">
            <div className="mapping-head">
              <span>Property</span>
              <span>Filter</span>
              <span>Channel</span>
            </div>
            {options?.properties.map((property) => (
              <div className="mapping-row" key={property.id}>
                <div>
                  <strong>{property.label}</strong>
                  <p className="fine-print">{property.kind}</p>
                </div>
                <FilterDropdown
                  values={property.values}
                  selected={filters[property.id] || []}
                  disabled={!property.values.length || property.id === xChannel || channels[property.id] === "y"}
                  onChange={(selected) => updateFilter(property.id, selected)}
                />
                <select
                  value={channels[property.id] || property.defaultChannel}
                  onChange={(event) => setPropertyChannel(property, event.target.value as Channel)}
                >
                  {property.allowedChannels.map((channel) => (
                    <option key={channel} value={channel}>
                      {channel}
                    </option>
                  ))}
                </select>
              </div>
            ))}
            <div className="mapping-actions">
              <button className="button secondary-button" onClick={() => setFilters({})} type="button">
                Clear filters
              </button>
              <p className="fine-print">Empty filters include all values. Multiple selected values are included.</p>
            </div>
          </div>
          <LineChart data={chartData} />
          <p className="fine-print chart-note">
            Y is mean execution time in milliseconds. Shaded bands and error bars show the 95% confidence interval when a point has multiple samples.
          </p>
        </section>
      )}
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
