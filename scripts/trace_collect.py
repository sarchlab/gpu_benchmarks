#!/usr/bin/env python3
"""Collect Accel-Sim 2.0 (.tracez) NVBit traces for tier1 + tier2 (CUDA).

Every trace goes through `gpubench run --collect-trace ... --tracer v2.0`, one
gpubench call per configuration, so this script only chooses configurations,
measures sizes, files the result and keeps the books.

Pipeline per configuration (Accel-Sim 2.0, pinned commit below):
  NVBit tracer -> kernel-*.trace.xz      (in-tracer address compression + xz -1)
  post-traces-processing -> *.tracez     (per-warp zstd chunks, trained dictionary)
Sizes recorded per configuration (summed over its kernels):
  S0  uncompressed text trace, read from the .trace.xz headers (never written out)
  S1  .trace.xz          S1_ratio = S1 / S0
  S2  .tracez            S2_ratio = S2 / S0
Only .tracez, kernelslist.g, stats and meta.json are kept; .trace.xz is deleted
after its size is recorded.

Configurations: every configuration with a status='ok' run in the guidance DB
(the ground-truth timing DB), vector_add_warmup excluded (gpubench forces it off
while tracing). Per benchmark and non-scaling combination, sizes run smallest
first; once one times out, the larger sizes of that combination are skipped.

Layout under the trace root (default /scratch/data/enze/gpu_benchmarks_trace):
  index.db          table `traces`: one row per trace_id (config, status, sizes)
  data/<trace_id>/  meta.json + traces/{kernelslist.g, *.tracez, stats*}
  staging/          gpubench's working dir for the configuration in progress
trace_id = <benchmark>-<first 12 hex of sha256(canonical config JSON)>, so the
same configuration always maps to the same directory; re-tracing replaces it.

Working files (all git-ignored) stay under scripts/:
  logs/<stamp>.csv  one row per configuration, written as it finishes (tail -f it)
  logs/<stamp>.log  driver log, ending with the size/ratio table
  logs/<stamp>.gpubench.log  raw gpubench output
  tmp/              demo trace root (wiped at the start of every demo)
  logs/report/      output of `report` (wiped and rewritten every time)
  .build/           Accel-Sim checkout, NVBit and the built tools

Demo: one tiny configuration of every benchmark (or of --benchmarks): its
smallest ok size and, among the non-scaling combinations at that size, the one
with the smallest median GPU time in the guidance DB (least work / fewest
launches).

Timeout: 600 s per configuration (trace + post-processing each), for run and demo.

Usage (from the repo root on the server):
  python3 scripts/trace_collect.py setup     # build tracer, post-processor, gpubench
  python3 scripts/trace_collect.py demo      # 1 tiny config per benchmark into scripts/tmp/
  python3 scripts/trace_collect.py run       # everything, into the trace root (tmux)
  python3 scripts/trace_collect.py stats     # print the table from index.db
  python3 scripts/trace_collect.py report    # md/csv/html report from index.db only,
                                             # into scripts/logs/report/ (recreated each time)
A configuration whose program launches no kernel (e.g. memory_bandwidth and
pcie_bandwidth only call cudaMemcpy) has nothing to trace; it is recorded as
no_kernels, with no data directory.
Configurations skipped after a timeout are recorded in index.db as
skipped_after_timeout (skipped_bench_timeouts with --max-timeouts-per-bench), metadata
only; such a row never replaces a real result and is re-decided on every run.
Rerunning `run` resumes: configurations already ok, no_kernels or timed out are skipped
(--force retraces them); errors are retried. --retry-timeouts retraces only the
timed-out ones (e.g. with a larger --timeout); once a size passes, the larger
sizes of that combination that were skipped behind it are traced too.
--max-timeouts-per-bench N (default 0 = no limit): after N timeouts in one
benchmark (any combinations), its remaining configurations are skipped.

Python 3.6+, stdlib only. Needs nvcc, nvidia-smi, xz, git, wget, make, bc,
libzstd-dev and Go (gpubench).
"""
import argparse
import csv
import datetime as dt
import decimal
import hashlib
import html
import json
import os
import re
import shutil
import signal
import sqlite3
import statistics
import subprocess
import sys
import time

ACCEL_SIM_URL = 'https://github.com/accel-sim/accel-sim-framework.git'
ACCEL_SIM_COMMIT = '7ba54ded60c4ba0083410c37d22d719166acd6dc'   # dev, 2026-10-27
DEFAULT_TRACE_ROOT = '/scratch/data/enze/gpu_benchmarks_trace'
DEFAULT_GUIDANCE_DB = os.path.join('ground_truth_h100_all', 'ground_truth_all.db')
MEASUREMENT_PARAMS = {'vector_add_warmup'}
PLATFORM = 'cuda'
TRACER = 'v2.0'

CSV_COLS = [
    'progress', 'timestring_start', 'timestring_end', 'time_span_sec',
    'tier', 'benchmark', 'trace_id', 'scaling_param', 'scaling_value',
    'non_scaling_params', 'status', 'n_kernels', 'size_s0', 'size_s1', 'size_s2',
    's1_ratio', 's2_ratio', 'free_gb', 'error_message',
]
SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id                TEXT PRIMARY KEY,
    tier                    INTEGER NOT NULL,
    benchmark               TEXT NOT NULL,
    platform                TEXT NOT NULL,
    scaling_param           TEXT NOT NULL,
    scaling_value           TEXT NOT NULL,
    non_scaling_params_json TEXT NOT NULL,
    config_json             TEXT NOT NULL,
    config_dir              TEXT,
    status                  TEXT NOT NULL,   -- ok | no_kernels | timeout | error | skipped_*
    n_kernels               INTEGER,
    size_s0                 INTEGER,         -- uncompressed .trace bytes
    size_s1                 INTEGER,         -- .trace.xz bytes
    size_s2                 INTEGER,         -- .tracez bytes
    s1_ratio                REAL,
    s2_ratio                REAL,
    wall_sec                REAL,
    error_message           TEXT,
    data_path               TEXT,            -- data/<trace_id>, ok rows only
    tracer                  TEXT,
    accel_sim_commit        TEXT,
    gpubench_commit         TEXT,
    gpu                     TEXT,
    collected_at            TEXT,
    updated_at              TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS traces_benchmark ON traces(benchmark, status);
"""


def now_str():
    return dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def one_line(s, limit=300):
    s = ' | '.join(x.strip() for x in (s or '').splitlines() if x.strip())
    return s if len(s) <= limit else s[:limit] + '...'


def canon(v):
    """Numbers as int when integral (DB stores scaling values as text)."""
    if isinstance(v, str):
        try:
            v = float(v)
        except ValueError:
            return v
    if isinstance(v, bool):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def fmt_value(v):
    """Spelled like gpubench's formatValue (Go 'f' format, no exponent)."""
    v = canon(v)
    return format(decimal.Decimal(repr(v)), 'f') if isinstance(v, float) else str(v)


def human(n):
    if n is None:
        return '-'
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if abs(n) < 1024 or unit == 'TiB':
            return '%.2f %s' % (n, unit) if unit != 'B' else '%d B' % n
        n /= 1024.0


class Config:
    def __init__(self, tier, bench, sp_name, sp_value, ns):
        self.tier = int(tier)
        self.bench = bench
        self.sp_name = sp_name
        self.sp_value = canon(sp_value)
        self.ns = {k: canon(v) for k, v in sorted(ns.items()) if k not in MEASUREMENT_PARAMS}
        self.ns_json = json.dumps(self.ns, sort_keys=True, separators=(',', ':'))
        self.config_json = json.dumps(
            {'tier': self.tier, 'benchmark': bench, 'platform': PLATFORM,
             'scaling': {sp_name: self.sp_value}, 'params': self.ns},
            sort_keys=True, separators=(',', ':'))
        self.trace_id = '%s-%s' % (
            bench, hashlib.sha256(self.config_json.encode()).hexdigest()[:12])

        self.size_rank = 0        # position in params.json's scaling values

    @property
    def group(self):
        return (self.tier, self.bench, self.ns_json)

    def sort_key(self):
        # Declared order, not value order: e.g. npb_mg classes run S, W, A, B, C.
        return (self.tier, self.bench, self.ns_json, self.size_rank, fmt_value(self.sp_value))


class Collector:
    def __init__(self, a, root, mode):
        self.a = a
        self.mode = mode
        self.repo = a.repo
        self.scripts = os.path.join(self.repo, 'scripts')
        self.build = os.path.join(self.scripts, '.build')
        self.root = root
        self.data = os.path.join(root, 'data')
        self.staging = os.path.join(root, 'staging')
        self.index_path = os.path.join(root, 'index.db')
        self.gpubench = os.path.join(self.repo, 'gpubench')
        tn = os.path.join(self.build, 'accel-sim', 'util', 'tracer_nvbit', 'tracer_tool')
        self.tracer_so = os.path.join(tn, 'tracer_tool.so')
        self.post_process = os.path.join(tn, 'traces-processing', 'post-traces-processing')
        stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        logs = os.path.join(self.scripts, 'logs')
        os.makedirs(logs, exist_ok=True)
        self.tag = '%s_%s' % (stamp, mode)
        self.log_path = os.path.join(logs, self.tag + '.log')
        self.csv_path = os.path.join(logs, self.tag + '.csv')
        self.gb_log_path = os.path.join(logs, self.tag + '.gpubench.log')
        self.log_f = open(self.log_path, 'a')
        self.child = None
        self.stopped = ''         # set when the disk guard stops a run
        self.done = 0
        self.total = 0

    # ── output ────────────────────────────────────────────────────────────
    def say(self, msg):
        line = '[%s] %s' % (now_str(), msg)
        print(line, flush=True)
        self.log_f.write(line + '\n')
        self.log_f.flush()

    def csv_write(self, row):
        new = not os.path.exists(self.csv_path)
        with open(self.csv_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLS)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, '') for k in CSV_COLS})
        latest = os.path.join(os.path.dirname(self.csv_path), 'latest.csv')
        try:
            if os.path.islink(latest) or os.path.exists(latest):
                os.remove(latest)
            os.symlink(os.path.basename(self.csv_path), latest)
        except OSError:
            pass

    def sh(self, cmd, cwd=None, timeout=None):
        p = subprocess.run(cmd, cwd=cwd or self.repo, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True,
                           timeout=timeout)
        return p.returncode, p.stdout

    # ── environment ───────────────────────────────────────────────────────
    def check_tools(self):
        missing = [p for p in (self.tracer_so, self.post_process, self.gpubench)
                   if not os.path.isfile(p)]
        if missing:
            sys.exit('missing %s\nrun: python3 scripts/trace_collect.py setup' % ', '.join(missing))
        _, out = self.sh([self.gpubench, 'run', '--help'])
        if '--tracer' not in out:
            sys.exit('%s has no --tracer flag (built from an older checkout)\n'
                     'run: python3 scripts/trace_collect.py setup' % self.gpubench)
        if not shutil.which('xz'):
            sys.exit('xz not found (needed to read .trace.xz sizes)')

    def provenance(self):
        info = {'tracer': TRACER, 'accel_sim_commit': ACCEL_SIM_COMMIT}
        try:
            _, out = self.sh(['nvidia-smi', '--query-gpu=name,uuid,driver_version',
                              '--format=csv,noheader'], timeout=30)
            info['gpu'] = out.strip().splitlines()[0] if out.strip() else ''
        except (OSError, subprocess.TimeoutExpired):
            info['gpu'] = ''
        _, head = self.sh(['git', 'rev-parse', 'HEAD'])
        _, dirty = self.sh(['git', 'status', '--porcelain', '--', 'cmd'])
        info['gpubench_commit'] = head.strip() + ('-dirty' if dirty.strip() else '')
        return info

    # ── index.db ──────────────────────────────────────────────────────────
    def db(self):
        con = sqlite3.connect(self.index_path, timeout=30)
        con.executescript(SCHEMA)
        return con

    def index_status(self):
        con = self.db()
        rows = dict(con.execute('SELECT trace_id, status FROM traces').fetchall())
        con.close()
        return rows

    def upsert(self, cfg, fields):
        row = {
            'trace_id': cfg.trace_id, 'tier': cfg.tier, 'benchmark': cfg.bench,
            'platform': PLATFORM, 'scaling_param': cfg.sp_name,
            'scaling_value': fmt_value(cfg.sp_value),
            'non_scaling_params_json': cfg.ns_json, 'config_json': cfg.config_json,
            'updated_at': now_str(),
        }
        row.update(fields)
        con = self.db()
        cols = ','.join(row)
        con.execute('INSERT OR REPLACE INTO traces (%s) VALUES (%s)'
                    % (cols, ','.join('?' * len(row))), list(row.values()))
        con.commit()
        con.close()

    def record_skip(self, cfg, status, row):
        """Note a skipped configuration in index.db (metadata only, no data dir).

        Only a configuration with no row yet, or an earlier skipped_* row, is
        written: a real result (ok, no_kernels, timeout, error) is never
        replaced. skipped_* rows are not in any resume skip list, so the next
        run decides afresh whether to trace them.
        """
        prev = status.get(cfg.trace_id)
        if prev is not None and not prev.startswith('skipped_'):
            return
        self.upsert(cfg, {'status': row['status'], 'error_message': row['error_message']})
        status[cfg.trace_id] = row['status']

    # ── configuration list ────────────────────────────────────────────────
    def configs(self):
        path = self.a.guidance_db
        if not os.path.isfile(path):
            sys.exit('guidance DB not found: %s (use --guidance-db)' % path)
        con = sqlite3.connect('file:%s?mode=ro' % path, uri=True)
        rows = con.execute(
            "SELECT DISTINCT tier, benchmark, scaling_param_name, scaling_param_value,"
            " non_scaling_params_json FROM runs WHERE status = 'ok'").fetchall()
        con.close()
        tiers = set(int(t) for t in self.a.tiers.split(','))
        seen = {}
        ranks = {}
        for tier, bench, spn, spv, nsp in rows:
            if int(tier) not in tiers:
                continue
            if self.a.benchmarks and bench not in self.a.benchmarks:
                continue
            c = Config(tier, bench, spn, spv, json.loads(nsp or '{}'))
            if bench not in ranks:
                p = os.path.join(self.repo, 'tier%d' % c.tier, bench, PLATFORM, 'params.json')
                with open(p) as f:
                    vals = json.load(f)['scaling_param']['values']
                ranks[bench] = {fmt_value(v): i for i, v in enumerate(vals)}
            c.size_rank = ranks[bench].get(fmt_value(c.sp_value), len(ranks[bench]))
            seen[c.trace_id] = c
        return sorted(seen.values(), key=Config.sort_key)

    def demo_configs(self, allc):
        """Per benchmark: smallest size, then the combination with least GPU time."""
        con = sqlite3.connect('file:%s?mode=ro' % self.a.guidance_db, uri=True)
        rows = con.execute(
            "SELECT tier, benchmark, scaling_param_name, scaling_param_value,"
            " non_scaling_params_json, total_time_ms FROM runs"
            " WHERE status = 'ok' AND total_time_ms IS NOT NULL").fetchall()
        con.close()
        times = {}
        for tier, bench, spn, spv, nsp, t in rows:
            times.setdefault(Config(tier, bench, spn, spv, json.loads(nsp or '{}')).trace_id,
                             []).append(t)

        def median(v):
            v = sorted(v)
            return v[len(v) // 2]
        best = {}
        for c in allc:
            key = (c.size_rank, median(times[c.trace_id]) if c.trace_id in times else float('inf'),
                   c.sort_key())
            if c.bench not in best or key < best[c.bench][0]:
                best[c.bench] = (key, c)
        return sorted((c for _, c in best.values()), key=Config.sort_key)

    # ── one configuration ─────────────────────────────────────────────────
    def compile(self, tier, bench):
        self.say('compile tier%d/%s' % (tier, bench))
        try:
            rc, out = self.sh([self.gpubench, 'compile', '--tier', str(tier), '--benchmark',
                               bench, '--platform', PLATFORM], timeout=900)
        except subprocess.TimeoutExpired:
            rc, out = -1, 'compile timed out after 900s'
        with open(self.gb_log_path, 'a') as f:
            f.write('\n===== %s compile tier%d/%s (exit %d)\n%s' % (now_str(), tier, bench, rc, out))
        return rc, out

    def gpubench_args(self, cfg):
        args = [self.gpubench, 'run', '--tier', str(cfg.tier), '--benchmark', cfg.bench,
                '--platform', PLATFORM, '--reps', '1', '--continue-on-error',
                '--collect-trace', self.tracer_so, '--tracer', TRACER,
                '--post-process', self.post_process, '--trace-output', self.staging,
                '--timeout', str(self.a.timeout), '--db', self.a.guidance_db,
                '--scaling-values', fmt_value(cfg.sp_value)]
        for k, v in cfg.ns.items():
            args += ['--param', '%s=%s' % (k, fmt_value(v))]
        return args

    def run_gpubench(self, args):
        """Returns gpubench's exit code, or None if interrupted."""
        with open(self.gb_log_path, 'a') as lf:
            lf.write('\n===== %s %s\n' % (now_str(), ' '.join(args)))
            lf.flush()
            # Own process group: Ctrl-C reaches only this script, which forwards
            # it to gpubench (second Ctrl-C kills the group).
            proc = subprocess.Popen(args, cwd=self.repo, stdout=lf, stderr=subprocess.STDOUT,
                                    preexec_fn=os.setpgrp)
            self.child = proc
            try:
                proc.wait()
            except KeyboardInterrupt:
                self.say('Ctrl-C: stopping after the current trace (Ctrl-C again to kill it)')
                try:
                    proc.send_signal(signal.SIGINT)
                    proc.wait()
                except KeyboardInterrupt:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                return None
            finally:
                self.child = None
        return proc.returncode

    @staticmethod
    def xz_uncompressed(files):
        if not files:
            return 0
        p = subprocess.run(['xz', '--robot', '--list'] + files, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True)
        if p.returncode != 0:
            raise RuntimeError('xz --list failed: ' + one_line(p.stderr))
        want = 'totals' if len(files) > 1 else 'file'
        for line in p.stdout.splitlines():
            f = line.split('\t')
            if f[0] == want:
                return int(f[4])
        raise RuntimeError('xz --list: no %s line' % want)

    @staticmethod
    def tracez_refs(traces):
        refs = []
        for name in sorted(os.listdir(traces)):
            if name.startswith('kernelslist') and name.endswith('.g'):
                with open(os.path.join(traces, name)) as f:
                    refs += [os.path.basename(x.strip()) for x in f
                             if x.strip().endswith('.tracez')]
        return refs

    def collect(self, cfg, info):
        """Trace one configuration; returns (status, fields, message)."""
        shutil.rmtree(self.staging, ignore_errors=True)
        os.makedirs(self.staging)
        t0 = time.time()
        rc = self.run_gpubench(self.gpubench_args(cfg))
        wall = time.time() - t0
        if rc is None:
            return 'interrupted', {'wall_sec': wall}, 'interrupted'
        cfg_dirs = [os.path.join(self.staging, cfg.bench, d)
                    for d in os.listdir(os.path.join(self.staging, cfg.bench))] \
            if os.path.isdir(os.path.join(self.staging, cfg.bench)) else []
        if len(cfg_dirs) != 1:
            return 'error', {'wall_sec': wall}, \
                'gpubench exit %s, %d config dirs (see %s)' % (rc, len(cfg_dirs), self.gb_log_path)
        cfg_dir = cfg_dirs[0]
        traces = os.path.join(cfg_dir, 'traces')
        if os.path.exists(os.path.join(traces, 'trace_timeout.json')):
            return 'timeout', {'wall_sec': wall, 'config_dir': os.path.basename(cfg_dir)}, \
                'timeout after %ds' % self.a.timeout
        refs = self.tracez_refs(traces) if os.path.isdir(traces) else []
        names = set(os.listdir(traces)) if os.path.isdir(traces) else set()
        xz = sorted(n for n in names if n.endswith('.trace.xz'))
        tz = sorted(n for n in names if n.endswith('.tracez'))
        lists = [n for n in names if n.startswith('kernelslist') and n.endswith('.g')]
        if lists and not refs and not xz and not tz:
            # Post-processed fine, but the program launched no kernel (only
            # memcpys), so gpubench reports it incomplete. Nothing to keep.
            return 'no_kernels', {'wall_sec': round(wall, 1), 'n_kernels': 0,
                                  'config_dir': os.path.basename(cfg_dir)}, \
                'program launched no kernel (kernelslist.g has only memcpys)'
        if rc != 0 or not refs or any(r not in names for r in refs) or len(xz) != len(tz):
            return 'error', {'wall_sec': wall, 'config_dir': os.path.basename(cfg_dir)}, \
                'gpubench exit %s, %d listed / %d .tracez / %d .trace.xz (see %s)' % (
                    rc, len(refs), len(tz), len(xz), self.gb_log_path)
        try:
            s0 = self.xz_uncompressed([os.path.join(traces, n) for n in xz])
        except RuntimeError as e:
            return 'error', {'wall_sec': wall}, str(e)
        s1 = sum(os.path.getsize(os.path.join(traces, n)) for n in xz)
        s2 = sum(os.path.getsize(os.path.join(traces, n)) for n in tz)
        for n in xz:
            os.remove(os.path.join(traces, n))
        fields = {
            'config_dir': os.path.basename(cfg_dir), 'n_kernels': len(refs),
            'size_s0': s0, 'size_s1': s1, 'size_s2': s2,
            's1_ratio': s1 / s0 if s0 else None, 's2_ratio': s2 / s0 if s0 else None,
            'wall_sec': round(wall, 1), 'data_path': os.path.join('data', cfg.trace_id),
            'collected_at': now_str(),
        }
        fields.update(info)
        meta = {'trace_id': cfg.trace_id, 'config': json.loads(cfg.config_json)}
        meta.update(fields)
        with open(os.path.join(cfg_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        # Replace any previous trace of this configuration via a temp name, so
        # data/<trace_id> is never half-written.
        os.makedirs(self.data, exist_ok=True)
        dest = os.path.join(self.data, cfg.trace_id)
        tmp = dest + '.tmp'
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.move(cfg_dir, tmp)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        os.rename(tmp, dest)
        return 'ok', fields, ''

    # ── main loop ─────────────────────────────────────────────────────────
    def run(self, configs, force=False):
        """Trace configs in order; returns False if interrupted."""
        if not self.a.dry_run:
            os.makedirs(self.root, exist_ok=True)
            self.check_tools()
        info = self.provenance()
        status = self.index_status() if os.path.isfile(self.index_path) else {}
        self.say('trace root %s, timeout %ds, min free %d GB, %s' % (
            self.root, self.a.timeout, self.a.min_free_gb, info.get('gpu', '')))
        self.say('csv: %s' % self.csv_path)
        skip = ('ok', 'no_kernels') if self.a.retry_timeouts else ('ok', 'no_kernels', 'timeout')
        todo = [c for c in configs if force or status.get(c.trace_id) not in skip]
        self.done = 0
        # Combinations with a timed-out size: their larger sizes are skipped.
        # With --retry-timeouts the timed-out sizes are retried first (sizes run
        # smallest first), so the block is rebuilt from this run's results.
        fresh = force or self.a.retry_timeouts
        timed_out = set() if fresh else \
            set(c.group for c in configs if status.get(c.trace_id) == 'timeout')
        # Timeouts per benchmark, for --max-timeouts-per-bench.
        bench_timeouts = {}
        if not fresh:
            for c in configs:
                if status.get(c.trace_id) == 'timeout':
                    bench_timeouts[(c.tier, c.bench)] = bench_timeouts.get((c.tier, c.bench), 0) + 1
        max_to = self.a.max_timeouts_per_bench
        self.total = len(todo)
        self.say('%d configurations, %d to trace (%d already done or timed out in index.db)' % (
            len(configs), len(todo), len(configs) - len(todo)))
        if self.a.dry_run:
            for c in todo:
                self.say('  plan %s %s=%s %s' % (c.trace_id, c.sp_name, fmt_value(c.sp_value), c.ns_json))
            return True
        compiled = {}
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        for c in todo:
            self.done += 1
            row = {'progress': '%d/%d' % (self.done, self.total), 'tier': c.tier,
                   'benchmark': c.bench, 'trace_id': c.trace_id, 'scaling_param': c.sp_name,
                   'scaling_value': fmt_value(c.sp_value), 'non_scaling_params': c.ns_json,
                   'timestring_start': now_str()}
            free_gb = shutil.disk_usage(self.root).free / 1e9
            row['free_gb'] = '%.1f' % free_gb
            if free_gb < self.a.min_free_gb:
                row.update(status='stopped_low_disk', timestring_end=now_str(), time_span_sec=0,
                           error_message='%.1f GB free < --min-free-gb %d' % (free_gb, self.a.min_free_gb))
                self.csv_write(row)
                self.stopped = 'only %.1f GB free on %s (< --min-free-gb %d)' % (
                    free_gb, self.root, self.a.min_free_gb)
                self.say('STOP: ' + self.stopped)
                return True
            if max_to and bench_timeouts.get((c.tier, c.bench), 0) >= max_to:
                row.update(status='skipped_bench_timeouts', timestring_end=now_str(), time_span_sec=0,
                           error_message='%d configurations of this benchmark timed out' % max_to)
                self.csv_write(row)
                self.record_skip(c, status, row)
                self.say('[%s] skip %s (%d timeouts in %s)' % (row['progress'], c.trace_id,
                                                               max_to, c.bench))
                continue
            if c.group in timed_out:
                row.update(status='skipped_after_timeout', timestring_end=now_str(), time_span_sec=0,
                           error_message='a smaller size of this combination timed out')
                self.csv_write(row)
                self.record_skip(c, status, row)
                self.say('[%s] skip %s (smaller size timed out)' % (row['progress'], c.trace_id))
                continue
            key = (c.tier, c.bench)
            if key not in compiled and not self.a.skip_compile:
                compiled[key] = self.compile(c.tier, c.bench)
            crc, cout = compiled.get(key, (0, ''))
            t0 = time.time()
            if crc != 0:
                st, fields, msg = 'error', {}, 'compile failed: ' + one_line(cout[-300:])
            else:
                st, fields, msg = self.collect(c, info)
            if st == 'interrupted':
                shutil.rmtree(self.staging, ignore_errors=True)
                self.say('interrupted; rerun the same command to resume')
                return False
            if st == 'timeout':
                timed_out.add(c.group)
                bench_timeouts[(c.tier, c.bench)] = bench_timeouts.get((c.tier, c.bench), 0) + 1
            if st == 'ok' or status.get(c.trace_id) != 'ok':
                # A failed retrace never overwrites an existing ok trace.
                db_fields = dict(fields, status=st, error_message=msg or None)
                if st != 'ok':
                    db_fields.update({k: info[k] for k in info})
                self.upsert(c, db_fields)
                status[c.trace_id] = st
            row.update(status=st, timestring_end=now_str(), time_span_sec=int(time.time() - t0),
                       error_message=one_line(msg))
            for k in ('n_kernels', 'size_s0', 'size_s1', 'size_s2'):
                row[k] = fields.get(k, '')
            for k in ('s1_ratio', 's2_ratio'):
                row[k] = '' if fields.get(k) is None else '%.6f' % fields[k]
            self.csv_write(row)
            extra = ''
            if st == 'ok':
                extra = '  S0 %s  S1 %s (%.4f)  S2 %s (%.4f)' % (
                    human(fields['size_s0']), human(fields['size_s1']), fields['s1_ratio'] or 0,
                    human(fields['size_s2']), fields['s2_ratio'] or 0)
            self.say('[%s] %s %s %s=%s %s %s%s' % (
                row['progress'], st, c.trace_id, c.sp_name, fmt_value(c.sp_value),
                c.ns_json, '' if st == 'ok' else one_line(msg, 160), extra))
        shutil.rmtree(self.staging, ignore_errors=True)
        return True

    # ── statistics ────────────────────────────────────────────────────────
    def stats(self):
        if not os.path.isfile(self.index_path):
            self.say('no index.db under %s' % self.root)
            return
        con = self.db()
        rows = con.execute(
            "SELECT benchmark, MIN(tier), SUM(status='ok'), SUM(status='timeout'),"
            " SUM(status='error'), SUM(status='no_kernels'),"
            " SUM(CASE WHEN status='ok' THEN size_s0 END),"
            " SUM(CASE WHEN status='ok' THEN size_s1 END),"
            " SUM(CASE WHEN status='ok' THEN size_s2 END)"
            " FROM traces GROUP BY benchmark ORDER BY MIN(tier), benchmark").fetchall()
        con.close()
        head = ('benchmark', '#configs', 'S0_raw', 'S1_trace.xz', 'S2_tracez',
                'S1_ratio', 'S2_ratio', '#timeout', '#error', '#no_kernels')
        lines = []
        tot = [0, 0, 0, 0, 0, 0, 0]
        for b, _, ok, to, er, nk, s0, s1, s2 in rows:
            s0, s1, s2 = s0 or 0, s1 or 0, s2 or 0
            lines.append((b, ok, human(s0), human(s1), human(s2),
                          '%.6f' % (s1 / s0) if s0 else '-', '%.6f' % (s2 / s0) if s0 else '-',
                          to, er, nk))
            for i, v in enumerate((ok, to, er, nk, s0, s1, s2)):
                tot[i] += v or 0
        ok, to, er, nk, s0, s1, s2 = tot
        lines.append(('TOTAL', ok, human(s0), human(s1), human(s2),
                      '%.6f' % (s1 / s0) if s0 else '-', '%.6f' % (s2 / s0) if s0 else '-',
                      to, er, nk))
        widths = [max(len(str(r[i])) for r in [head] + lines) for i in range(len(head))]
        fmt = '  '.join(('%%-%ds' if i == 0 else '%%%ds') % w for i, w in enumerate(widths))
        table = [fmt % head] + [fmt % tuple(r) for r in lines]
        self.say('trace statistics for %s (sizes summed over ok configurations;'
                 ' ratio = S/S0):\n%s' % (self.root, '\n'.join(table)))


def demo(a):
    tmp = os.path.join(a.repo, 'scripts', 'tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    root = os.path.join(tmp, 'trace_root')
    col = Collector(a, root, 'demo')
    col.say('demo: cleared %s' % tmp)
    picks = col.demo_configs(col.configs())
    col.say('demo: %d benchmarks, one tiny configuration each' % len(picks))
    for c in picks:
        col.say('demo pick %s %s=%s %s' % (c.trace_id, c.sp_name, fmt_value(c.sp_value), c.ns_json))
    if not col.run(picks):
        return 130
    if a.dry_run:
        return 0
    if col.stopped:
        col.say('demo aborted: %s' % col.stopped)
        return 1
    # Replacement check: retrace the first pick; its trace_id and directory
    # must stay the same and index.db must still hold one row for it.
    if picks:
        first = picks[0]
        col.say('demo: retracing %s to check replacement' % first.trace_id)
        if not col.run([first], force=True):
            return 130
    con = col.db()
    rows = con.execute('SELECT trace_id, status, n_kernels FROM traces').fetchall()
    con.close()
    ok = [r for r in rows if r[1] in ('ok', 'no_kernels')]
    problems = []
    for tid, st, nk in rows:
        d = os.path.join(col.data, tid, 'traces')
        if st == 'no_kernels':
            col.say('  %s  no kernels (memcpy only), nothing stored' % tid)
            continue
        if st != 'ok':
            problems.append('%s: status %s' % (tid, st))
            continue
        tz = [n for n in os.listdir(d) if n.endswith('.tracez')] if os.path.isdir(d) else []
        leftovers = [n for n in os.listdir(d) if n.endswith('.trace.xz')] if os.path.isdir(d) else []
        if not os.path.isfile(os.path.join(col.data, tid, 'meta.json')):
            problems.append('%s: no meta.json' % tid)
        if len(tz) != nk or nk == 0:
            problems.append('%s: %d .tracez files, index says %d kernels' % (tid, len(tz), nk))
        if leftovers:
            problems.append('%s: %d .trace.xz left behind' % (tid, len(leftovers)))
        col.say('  %s  %d kernels  %d files' % (tid, nk, len(os.listdir(d)) if os.path.isdir(d) else 0))
    dirs = sorted(x for x in os.listdir(col.data)) if os.path.isdir(col.data) else []
    stored = [r for r in rows if r[1] == 'ok']
    if len(dirs) != len(stored):
        problems.append('data/ has %d entries, index.db has %d ok rows' % (len(dirs), len(stored)))
    col.stats()
    col.say('demo: %d/%d picked configurations traced ok, data/ entries %d, problems: %s' % (
        len(ok), len(picks), len(dirs), '; '.join(problems) or 'none'))
    col.say('log: %s' % col.log_path)
    return 0 if len(ok) == len(picks) and not problems else 1


# ── report (index.db only) ────────────────────────────────────────────────
REPORT_BUCKETS = [(0, 1 << 20, '< 1 MiB'), (1 << 20, 10 << 20, '1-10 MiB'),
                  (10 << 20, 100 << 20, '10-100 MiB'), (100 << 20, 1 << 30, '100 MiB-1 GiB'),
                  (1 << 30, 1 << 62, '>= 1 GiB')]


def hms(sec):
    sec = int(round(sec or 0))
    if sec < 60:
        return '%ds' % sec
    if sec < 3600:
        return '%dm%02ds' % (sec // 60, sec % 60)
    return '%dh%02dm' % (sec // 3600, sec % 3600 // 60)


def fmt_ratio(x):
    return '-' if x is None else '%.4f' % x


def scaling_ranks(repo, tier, bench):
    """Position of each scaling value in params.json (the declared size order)."""
    try:
        with open(os.path.join(repo, 'tier%d' % tier, bench, PLATFORM, 'params.json')) as f:
            vals = json.load(f)['scaling_param']['values']
    except (OSError, KeyError, ValueError):
        return {}
    return {fmt_value(canon(v)): i for i, v in enumerate(vals)}


def report_summary(repo, rows):
    benches = {}
    for r in rows:
        benches.setdefault((r['tier'], r['benchmark']), []).append(r)
    out = []
    for (tier, bench), rs in sorted(benches.items()):
        ranks = scaling_ranks(repo, tier, bench)
        rank = lambda r: ranks.get(r['scaling_value'], -1)
        ok = [r for r in rs if r['status'] == 'ok']
        cnt = lambda pred: sum(1 for r in rs if pred(r['status']))
        s0 = sum(r['size_s0'] or 0 for r in ok)
        s1 = sum(r['size_s1'] or 0 for r in ok)
        s2 = sum(r['size_s2'] or 0 for r in ok)
        ratios = [r['s2_ratio'] for r in ok if r['s2_ratio'] is not None]
        s = {
            'tier': tier, 'benchmark': bench, 'scaling_param': rs[0]['scaling_param'],
            'planned': len(rs), 'ok': len(ok), 'timeout': cnt(lambda x: x == 'timeout'),
            'skipped': cnt(lambda x: x.startswith('skipped_')),
            'no_kernels': cnt(lambda x: x == 'no_kernels'), 'error': cnt(lambda x: x == 'error'),
            's0': s0, 's1': s1, 's2': s2,
            's1_ratio': s1 / s0 if s0 else None, 's2_ratio': s2 / s0 if s0 else None,
            's2_ratio_median': statistics.median(ratios) if ratios else None,
            'wall_sec': sum(r['wall_sec'] or 0 for r in rs), 'max_ok': None, 'min_timeout': None,
        }
        if ok:
            b = max(ok, key=lambda r: (rank(r), r['size_s0'] or 0))
            s['max_ok'] = {'value': b['scaling_value'], 'params': b['non_scaling_params_json'],
                           's0': b['size_s0'] or 0, 's2': b['size_s2'] or 0,
                           'n_kernels': b['n_kernels'] or 0, 'sec': b['wall_sec'] or 0,
                           'trace_id': b['trace_id']}
        tos = [r for r in rs if r['status'] == 'timeout']
        if tos:
            f = min(tos, key=rank)
            s['min_timeout'] = {'value': f['scaling_value'], 'params': f['non_scaling_params_json']}
        out.append(s)
    return out


def report_buckets(rows):
    ok = [r for r in rows if r['status'] == 'ok' and r['size_s0']]
    res = []
    for lo, hi, name in REPORT_BUCKETS:
        b = [r for r in ok if lo <= r['size_s0'] < hi]
        if not b:
            continue
        s0 = sum(r['size_s0'] for r in b)
        s1 = sum(r['size_s1'] for r in b)
        s2 = sum(r['size_s2'] for r in b)
        res.append({'bucket': name, 'n': len(b), 's0': s0, 's1': s1, 's2': s2,
                    's1_ratio': s1 / s0, 's2_ratio': s2 / s0,
                    's2_lt_s1': sum(1 for r in b if r['size_s2'] < r['size_s1'])})
    return res


def report_totals(rows, summary):
    tot = {k: sum(s[k] for s in summary) for k in
           ('planned', 'ok', 'timeout', 'skipped', 'no_kernels', 'error', 's0', 's1', 's2', 'wall_sec')}
    ok = [r for r in rows if r['status'] == 'ok']
    real = [r for r in rows if not r['status'].startswith('skipped_')]
    stamps = sorted(r['updated_at'] for r in real if r['updated_at'])
    tot['first'] = stamps[0] if stamps else ''
    tot['last'] = stamps[-1] if stamps else ''
    tot['timeout_sec'] = sum(r['wall_sec'] or 0 for r in rows if r['status'] == 'timeout')
    tot['s2_lt_s1'] = sum(1 for r in ok if (r['size_s2'] or 0) < (r['size_s1'] or 0))
    gpus = [r['gpu'] for r in ok if r['gpu']]
    tot['gpu'] = gpus[0].split(',')[0].strip() if gpus else 'GPU'
    limits = set()
    for r in rows:
        m = re.match(r'timeout after (\d+)s', r['error_message'] or '')
        if r['status'] == 'timeout' and m:
            limits.add(int(m.group(1)))
    tot['timeout_limit'] = '/'.join('%d s' % t for t in sorted(limits)) or 'n/a'
    return tot


def report_md(path, root, tot, summary, buckets):
    L = ['# Accel-Sim 2.0 trace collection report (%s, CUDA)\n' % tot['gpu'],
         'Source: `%s/index.db`  ' % root,
         'Collected %s -> %s, traced wall time %s (timeouts: %s)  ' % (
             tot['first'], tot['last'], hms(tot['wall_sec']), hms(tot['timeout_sec'])),
         'S0 = uncompressed text trace, S1 = .trace.xz, S2 = .tracez (kept); '
         'ratios are size-weighted, sum(S) / sum(S0).\n',
         '## Overall\n',
         '| configs | traced ok | timeout (%s) | skipped after timeout | no kernels | error |'
         % tot['timeout_limit'],
         '|---:|---:|---:|---:|---:|---:|',
         '| %d | %d | %d | %d | %d | %d |\n' % (
             tot['planned'], tot['ok'], tot['timeout'], tot['skipped'], tot['no_kernels'], tot['error']),
         '| S0 total | S1 total | S2 total | S1 ratio | S2 ratio | configs with S2 < S1 |',
         '|---:|---:|---:|---:|---:|---:|',
         '| %s | %s | %s | %s | %s | %d / %d |\n' % (
             human(tot['s0']), human(tot['s1']), human(tot['s2']),
             fmt_ratio(tot['s1'] / tot['s0'] if tot['s0'] else None),
             fmt_ratio(tot['s2'] / tot['s0'] if tot['s0'] else None), tot['s2_lt_s1'], tot['ok']),
         '## Compression by trace size (S0)\n',
         '| S0 bucket | configs | S0 | S1 ratio | S2 ratio | S2 < S1 |',
         '|---|---:|---:|---:|---:|---:|']
    for b in buckets:
        L.append('| %s | %d | %s | %s | %s | %d |' % (
            b['bucket'], b['n'], human(b['s0']), fmt_ratio(b['s1_ratio']),
            fmt_ratio(b['s2_ratio']), b['s2_lt_s1']))
    L += ['', '## Per benchmark\n',
          '| tier | benchmark | ok / configs | timeout | skipped | S0 | S2 | S1 ratio | S2 ratio '
          '| largest ok size | its S0 | its time | smallest timeout size | wall time |',
          '|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|']
    for s in summary:
        m, t = s['max_ok'], s['min_timeout']
        note = ' (no kernels: %d)' % s['no_kernels'] if s['no_kernels'] else ''
        L.append('| %d | %s | %d / %d%s | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            s['tier'], s['benchmark'], s['ok'], s['planned'], note, s['timeout'], s['skipped'],
            human(s['s0']) if s['ok'] else '-', human(s['s2']) if s['ok'] else '-',
            fmt_ratio(s['s1_ratio']), fmt_ratio(s['s2_ratio']),
            ('%s=%s %s' % (s['scaling_param'], m['value'], m['params'])) if m else '-',
            human(m['s0']) if m else '-', hms(m['sec']) if m else '-',
            ('%s=%s' % (s['scaling_param'], t['value'])) if t else '-', hms(s['wall_sec'])))
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')


def report_csv(path, summary):
    cols = ['tier', 'benchmark', 'scaling_param', 'planned', 'ok', 'timeout', 'skipped',
            'no_kernels', 'error', 's0', 's1', 's2', 's1_ratio', 's2_ratio', 's2_ratio_median',
            'max_ok_value', 'max_ok_params', 'max_ok_s0', 'max_ok_s2', 'max_ok_n_kernels',
            'max_ok_sec', 'max_ok_trace_id', 'min_timeout_value', 'min_timeout_params', 'wall_sec']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s in summary:
            m, t = s['max_ok'] or {}, s['min_timeout'] or {}
            r = {k: s[k] for k in cols if k in s}
            for k in ('s1_ratio', 's2_ratio', 's2_ratio_median'):
                r[k] = '' if s[k] is None else '%.6f' % s[k]
            r.update(max_ok_value=m.get('value', ''), max_ok_params=m.get('params', ''),
                     max_ok_s0=m.get('s0', ''), max_ok_s2=m.get('s2', ''),
                     max_ok_n_kernels=m.get('n_kernels', ''), max_ok_sec=m.get('sec', ''),
                     max_ok_trace_id=m.get('trace_id', ''), min_timeout_value=t.get('value', ''),
                     min_timeout_params=t.get('params', ''), wall_sec=int(s['wall_sec']))
            w.writerow(r)


def report_chart(buckets):
    W, H, padl, padb, top = 640, 260, 52, 56, 16
    ymax = max([0.12] + [max(b['s1_ratio'], b['s2_ratio']) * 1.15 for b in buckets])
    step = ymax / 4
    plot = H - top - padb
    bw = (W - padl - 20) / max(len(buckets), 1)
    svg = ['<svg viewBox="0 0 %d %d" role="img" aria-label="Compression ratio by trace size">' % (W, H)]
    for i in range(5):
        t = step * i
        y = top + plot * (1 - t / ymax)
        svg.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" class="grid"/>' % (padl, W - 10, y, y))
        svg.append('<text x="%d" y="%.1f" class="ax" text-anchor="end">%.2f</text>' % (padl - 6, y + 4, t))
    for i, b in enumerate(buckets):
        x0, w = padl + i * bw + bw * 0.18, bw * 0.3
        for j, (k, cls) in enumerate((('s1_ratio', 'b1'), ('s2_ratio', 'b2'))):
            v = b[k]
            y = top + plot * (1 - v / ymax)
            x = x0 + j * (w + 4)
            svg.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" class="%s" rx="2"/>'
                       % (x, y, w, top + plot - y, cls))
            svg.append('<text x="%.1f" y="%.1f" class="val" text-anchor="middle">%.3f</text>'
                       % (x + w / 2, y - 4, v))
        cx = padl + i * bw + bw / 2
        svg.append('<text x="%.1f" y="%d" class="ax" text-anchor="middle">%s</text>'
                   % (cx, H - padb + 18, html.escape(b['bucket'])))
        svg.append('<text x="%.1f" y="%d" class="ax sub" text-anchor="middle">%d configs</text>'
                   % (cx, H - padb + 34, b['n']))
    svg.append('</svg>')
    return '\n'.join(svg)


def report_html(path, tot, summary, buckets):
    e = html.escape
    trs = []
    for s in summary:
        m, cov = s['max_ok'], (s['ok'] / s['planned'] if s['planned'] else 0)
        if s['no_kernels'] == s['planned']:
            chip = '<span class="chip nk">memcpy only</span>'
        elif s['ok'] == 0:
            chip = '<span class="chip bad">none traced</span>'
        elif s['ok'] == s['planned']:
            chip = '<span class="chip good">all</span>'
        else:
            chip = ''
        if m:
            p = json.loads(m['params']) if m['params'] else {}
            largest = '<code>%s=%s</code><div class="muted">%s</div>' % (
                e(s['scaling_param']), e(m['value']), e(', '.join('%s=%s' % kv for kv in p.items())))
        else:
            largest = '<span class="muted">-</span>'
        first_to = ('%s=%s' % (s['scaling_param'], s['min_timeout']['value'])) if s['min_timeout'] else '-'
        num = lambda x: -1 if x is None else x
        trs.append(
            '<tr data-tier="%d"><td class="n">%d</td><td><b>%s</b> %s</td>'
            '<td class="n" data-v="%f">%d / %d<div class="bar"><span style="width:%.1f%%"></span></div></td>'
            '<td class="n" data-v="%s">%s</td><td class="n" data-v="%s">%s</td>'
            '<td class="n" data-v="%s">%s</td><td class="n" data-v="%d">%s</td><td>%s</td>'
            '<td class="n" data-v="%d">%s</td><td class="n" data-v="%d">%s</td><td><code>%s</code></td></tr>' % (
                s['tier'], s['tier'], e(s['benchmark']), chip, cov, s['ok'], s['planned'], 100 * cov,
                num(s['s2_ratio']), fmt_ratio(s['s2_ratio']), num(s['s1_ratio']), fmt_ratio(s['s1_ratio']),
                num(s['s2_ratio_median']), fmt_ratio(s['s2_ratio_median']),
                s['s2'], human(s['s2']) if s['ok'] else '-', largest,
                m['s0'] if m else -1, human(m['s0']) if m else '-',
                m['sec'] if m else -1, hms(m['sec']) if m else '-', e(first_to)))
    none = [s['benchmark'] for s in summary if s['ok'] == 0 and s['no_kernels'] < s['planned']]
    nk = [s['benchmark'] for s in summary if s['planned'] and s['no_kernels'] == s['planned']]
    heavy = [s['benchmark'] for s in summary if s['s2_ratio'] is not None and s['s2_ratio'] > 0.5]
    notes = ['Timeout %s per configuration (tracing and post-processing are timed separately). '
             'Sizes run smallest first. Once one size times out, the larger sizes with the same '
             'other parameters are skipped (%d configurations skipped, %d timeouts).'
             % (tot['timeout_limit'], tot['skipped'], tot['timeout'])]
    if none:
        notes.append('Nothing traced in %d benchmark%s: even the smallest configuration ran out '
                     'of time (%s).' % (len(none), '' if len(none) == 1 else 's', e(', '.join(none))))
    if nk:
        notes.append('%s only call <code>cudaMemcpy</code> and launch no kernels, so there is '
                     'nothing to trace (%d configurations).' % (e(', '.join(nk)), tot['no_kernels']))
    if heavy:
        notes.append('S2 ratio above 0.5 in %s: the warps execute only a few instructions each, '
                     'so the per-block and per-warp index of .tracez outweighs the content.'
                     % e(', '.join(heavy)))
    big = buckets[-1] if buckets else None
    big_text = ('For traces of %s, .tracez is %.3f of raw against %.3f for xz (%d configurations). '
                % (big['bucket'], big['s2_ratio'], big['s1_ratio'], big['n'])) if big else ''
    vals = {
        'gpu': e(tot['gpu']), 'first': e(tot['first'][:10]), 'last': e(tot['last'][:10]),
        'ok': str(tot['ok']), 'planned': str(tot['planned']), 'errors': str(tot['error']),
        's0': human(tot['s0']), 's1': human(tot['s1']), 's2': human(tot['s2']),
        'r1': fmt_ratio(tot['s1'] / tot['s0'] if tot['s0'] else None),
        'r2': fmt_ratio(tot['s2'] / tot['s0'] if tot['s0'] else None),
        'x1': '%.1f' % (tot['s0'] / tot['s1']) if tot['s1'] else '-',
        'x2': '%.1f' % (tot['s0'] / tot['s2']) if tot['s2'] else '-',
        'wall': '%.1f' % (tot['wall_sec'] / 3600), 'to_h': '%.1f' % (tot['timeout_sec'] / 3600),
        'timeout_limit': e(tot['timeout_limit']), 'chart': report_chart(buckets),
        'big_text': big_text, 'lt': str(tot['s2_lt_s1']),
        'notes': '\n'.join('<li>%s</li>' % n for n in notes),
        'nb': str(len(summary)), 'nbok': str(sum(1 for s in summary if s['ok'])),
        'rows': '\n'.join(trs),
    }
    page = REPORT_HTML
    for k, v in vals.items():
        page = page.replace('{{%s}}' % k, v)
    with open(path, 'w') as f:
        f.write(page)


def report(a):
    root = os.path.abspath(a.root)
    index = os.path.join(root, 'index.db')
    if not os.path.isfile(index):
        print('no index.db under %s' % root, file=sys.stderr)
        return 1
    con = sqlite3.connect('file:%s?mode=ro' % index, uri=True, timeout=30)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute('SELECT * FROM traces')]
    con.close()
    if a.benchmarks:
        rows = [r for r in rows if r['benchmark'] in a.benchmarks]
    if not rows:
        print('index.db has no rows to report', file=sys.stderr)
        return 1
    out = os.path.join(a.repo, 'scripts', 'logs', 'report')
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    summary = report_summary(a.repo, rows)
    buckets = report_buckets(rows)
    tot = report_totals(rows, summary)
    report_md(os.path.join(out, 'report.md'), root, tot, summary, buckets)
    report_csv(os.path.join(out, 'report.csv'), summary)
    report_html(os.path.join(out, 'report.html'), tot, summary, buckets)
    print('report from %s (%d configurations, %d ok, %d skipped rows)' % (
        index, tot['planned'], tot['ok'], tot['skipped']))
    if not tot['skipped'] and tot['timeout']:
        print('note: index.db has timeouts but no skipped_* rows; run `run` once to record them')
    for n in ('report.html', 'report.md', 'report.csv'):
        print('  %s' % os.path.join(out, n))
    return 0


REPORT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trace Collection Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
/* Layout: one reading column for summary + method, full-width table below. Engineering-report feel. */
:root {
  --bg: #f6f7f9;
  --surface: #ffffff;
  --ink: #1b2330;
  --muted: #5d6878;
  --line: #dde2e9;
  --accent: #1f5fae;      /* S2 / .tracez */
  --accent2: #b8894a;     /* S1 / .trace.xz */
  --good: #2f7d4f;
  --bad: #b3412f;
  --font-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #12161d; --surface: #1a2029; --ink: #e5e9ef; --muted: #98a3b3; --line: #2c3440;
    --accent: #6ea4ea; --accent2: #d7a865; --good: #6cc08f; --bad: #e98a76; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #12161d; --surface: #1a2029; --ink: #e5e9ef; --muted: #98a3b3; --line: #2c3440;
  --accent: #6ea4ea; --accent2: #d7a865; --good: #6cc08f; --bad: #e98a76; color-scheme: dark;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.55 var(--font-body); }
main { max-width: 1240px; margin: 0 auto; padding: 32px 20px 64px; display: grid; gap: 28px; }
h1, h2 { text-wrap: balance; margin: 0; font-weight: 600; }
h1 { font-size: 28px; letter-spacing: -0.01em; }
h2 { font-size: 18px; }
p { margin: 0; max-width: 72ch; }
.muted { color: var(--muted); font-size: 13px; }
code { font-family: var(--font-mono); font-size: 12.5px; }
header { display: grid; gap: 6px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }
.tile { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; display: grid; gap: 2px; }
.tile .k { color: var(--muted); font-size: 13px; }
.tile .v { font-size: 26px; font-weight: 600; font-variant-numeric: tabular-nums; }
.tile .s { color: var(--muted); font-size: 13px; }
.tile.hl { border-color: var(--accent); }
.tile.hl .v { color: var(--accent); }
section { display: grid; gap: 12px; }
.two { display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr); gap: 20px; align-items: start; }
@media (max-width: 860px) { .two { grid-template-columns: 1fr; } }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 16px 18px; display: grid; gap: 10px; }
.card ul { margin: 0; padding-left: 18px; display: grid; gap: 6px; }
dl { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 6px 14px; }
dt { font-family: var(--font-mono); font-weight: 500; }
dd { margin: 0; }
svg { width: 100%; height: auto; display: block; }
svg .grid { stroke: var(--line); stroke-width: 1; }
svg .ax { fill: var(--muted); font: 12px var(--font-body); }
svg .ax.sub { font-size: 11px; }
svg .val { fill: var(--ink); font: 11px var(--font-mono); }
svg .b1 { fill: var(--accent2); }
svg .b2 { fill: var(--accent); }
.legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 13px; color: var(--muted); }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
.tablewrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; min-width: 1080px; font-size: 13.5px; }
th, td { padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
th { position: sticky; top: 0; background: var(--surface); font-weight: 600; font-size: 12.5px; color: var(--muted); white-space: nowrap; }
th button { all: unset; cursor: pointer; }
th button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
th button[aria-sort]::after { content: " ↕"; opacity: .45; }
th button[aria-sort="ascending"]::after { content: " ↑"; opacity: 1; }
th button[aria-sort="descending"]::after { content: " ↓"; opacity: 1; }
td.n { font-variant-numeric: tabular-nums; white-space: nowrap; text-align: right; }
td .muted { font-size: 12px; margin-top: 2px; }
tr:last-child td { border-bottom: 0; }
.bar { height: 4px; background: var(--line); border-radius: 2px; margin-top: 5px; overflow: hidden; }
.bar span { display: block; height: 100%; background: var(--accent); }
.chip { display: inline-block; font-size: 11px; font-weight: 500; padding: 1px 7px; border-radius: 999px; margin-left: 6px; border: 1px solid currentColor; }
.chip.good { color: var(--good); }
.chip.bad { color: var(--bad); }
.chip.nk { color: var(--muted); }
.filters { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.filters button { font: inherit; font-size: 13px; padding: 4px 12px; border-radius: 999px; border: 1px solid var(--line); background: var(--surface); color: var(--ink); cursor: pointer; }
.filters button[aria-pressed="true"] { border-color: var(--accent); color: var(--accent); }
.filters button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
</style>
</head>
<body>
<main>
  <header>
    <h1>NVBit trace collection with Accel-Sim 2.0 compression</h1>
    <p class="muted">tier1 + tier2 CUDA benchmarks, {{gpu}} · {{first}} to {{last}} · every configuration that ran successfully in the ground-truth timing sweep, traced once</p>
  </header>

  <div class="tiles">
    <div class="tile"><span class="k">Configurations traced</span><span class="v">{{ok}}</span><span class="s">of {{planned}} planned, {{errors}} errors</span></div>
    <div class="tile"><span class="k">Raw trace text (S0)</span><span class="v">{{s0}}</span><span class="s">never written to disk</span></div>
    <div class="tile hl"><span class="k">Stored as .tracez (S2)</span><span class="v">{{s2}}</span><span class="s">S2/S0 = {{r2}} ({{x2}}× smaller)</span></div>
    <div class="tile"><span class="k">.trace.xz for comparison (S1)</span><span class="v">{{s1}}</span><span class="s">S1/S0 = {{r1}} ({{x1}}× smaller)</span></div>
    <div class="tile"><span class="k">Collection time</span><span class="v">{{wall}} h</span><span class="s">{{to_h}} h spent on {{timeout_limit}} timeouts</span></div>
  </div>

  <div class="two">
    <section class="card">
      <h2>Compression ratio by trace size</h2>
      <div class="legend"><span><i class="sw" style="background:var(--accent2)"></i>S1 .trace.xz / S0</span><span><i class="sw" style="background:var(--accent)"></i>S2 .tracez / S0</span><span>lower is better</span></div>
      {{chart}}
      <p class="muted">Configurations grouped by raw trace size S0; ratios are size-weighted (sum S / sum S0). {{big_text}}For small traces, the fixed per-file, per-block and per-warp overhead of .tracez outweighs that advantage. Overall, .tracez is smaller than .trace.xz for {{lt}} of {{ok}} configurations.</p>
    </section>

    <section class="card">
      <h2>What was measured</h2>
      <dl>
        <dt>S0</dt><dd>The tracer's text trace: one line per executed warp instruction, with Accel-Sim's address compression already applied. Measured from the .trace.xz headers.</dd>
        <dt>S1</dt><dd><code>.trace.xz</code>: S0 piped through <code>xz -1</code> by the tracer. Deleted once measured.</dd>
        <dt>S2</dt><dd><code>.tracez</code>: Accel-Sim 2.0 post-processing regroups the trace per block and warp and compresses it with zstd in independent chunks plus a trained dictionary, so the simulator can read any warp without decompressing the whole kernel. This is the format we keep.</dd>
      </dl>
      <ul class="muted">
{{notes}}
      </ul>
    </section>
  </div>

  <section>
    <h2>Per benchmark</h2>
    <div class="filters" role="group" aria-label="Filter by tier">
      <span class="muted">Show</span>
      <button type="button" id="f-all" data-tier="" aria-pressed="true">All {{nb}}</button>
      <button type="button" id="f-1" data-tier="1" aria-pressed="false">Tier 1</button>
      <button type="button" id="f-2" data-tier="2" aria-pressed="false">Tier 2</button>
      <span class="muted">{{nbok}} of {{nb}} benchmarks have at least one trace · click a column header to sort</span>
    </div>
    <div class="tablewrap">
      <table id="t">
        <thead><tr>
          <th><button type="button" data-col="0" data-num="1" aria-sort="none">Tier</button></th>
          <th><button type="button" data-col="1" aria-sort="none">Benchmark</button></th>
          <th style="text-align:right"><button type="button" data-col="2" data-num="1" aria-sort="none">Traced / planned</button></th>
          <th style="text-align:right"><button type="button" data-col="3" data-num="1" aria-sort="none">S2 ratio</button></th>
          <th style="text-align:right"><button type="button" data-col="4" data-num="1" aria-sort="none">S1 ratio</button></th>
          <th style="text-align:right"><button type="button" data-col="5" data-num="1" aria-sort="none">Median S2 ratio</button></th>
          <th style="text-align:right"><button type="button" data-col="6" data-num="1" aria-sort="none">Stored (S2)</button></th>
          <th>Largest configuration traced</th>
          <th style="text-align:right"><button type="button" data-col="8" data-num="1" aria-sort="none">Its S0</button></th>
          <th style="text-align:right"><button type="button" data-col="9" data-num="1" aria-sort="none">Its time</button></th>
          <th>Smallest timeout</th>
        </tr></thead>
        <tbody>
{{rows}}
        </tbody>
      </table>
    </div>
    <p class="muted">S1 and S2 ratios are size-weighted over each benchmark's traced configurations. "Largest configuration traced" is the largest scaling value that finished in time; its time includes tracing and post-processing. "Smallest timeout" is the smallest scaling value that timed out, across all combinations of the other parameters.</p>
  </section>
</main>
<script>
(function () {
  var tbody = document.querySelector('#t tbody');
  var heads = document.querySelectorAll('#t th button');
  heads.forEach(function (b) {
    b.addEventListener('click', function () {
      var col = +b.dataset.col, num = !!b.dataset.num;
      var dir = b.getAttribute('aria-sort') === 'ascending' ? 'descending' : 'ascending';
      heads.forEach(function (h) { h.setAttribute('aria-sort', 'none'); });
      b.setAttribute('aria-sort', dir);
      var rows = Array.prototype.slice.call(tbody.rows);
      rows.sort(function (a, c) {
        var x = a.cells[col], y = c.cells[col], r;
        if (num) {
          r = parseFloat(x.dataset.v !== undefined ? x.dataset.v : x.textContent) -
              parseFloat(y.dataset.v !== undefined ? y.dataset.v : y.textContent);
        } else { r = x.textContent.localeCompare(y.textContent); }
        return dir === 'ascending' ? r : -r;
      });
      rows.forEach(function (r) { tbody.appendChild(r); });
    });
  });
  var fb = document.querySelectorAll('.filters button');
  fb.forEach(function (b) {
    b.addEventListener('click', function () {
      fb.forEach(function (o) { o.setAttribute('aria-pressed', o === b ? 'true' : 'false'); });
      Array.prototype.forEach.call(tbody.rows, function (r) {
        r.hidden = !!b.dataset.tier && r.dataset.tier !== b.dataset.tier;
      });
    });
  });
})();
</script>
</body>
</html>
"""


def setup(a):
    build = os.path.join(a.repo, 'scripts', '.build')
    os.makedirs(build, exist_ok=True)
    log = Collector(a, os.path.join(build, 'unused'), 'setup')
    missing = [t for t in ('git', 'wget', 'make', 'bc', 'nvcc', 'g++', 'xz', 'go')
               if not shutil.which(t)]
    if missing:
        sys.exit('missing tools: %s' % ' '.join(missing))
    if not any(os.path.isfile(os.path.join(d, 'zstd.h'))
               for d in ('/usr/include', '/usr/local/include')):
        log.say('warning: zstd.h not found in /usr/include; the build needs libzstd-dev')

    def step(cmd, cwd):
        log.say('$ (cd %s && %s)' % (cwd, ' '.join(cmd)))
        rc, out = log.sh(cmd, cwd=cwd)
        log.log_f.write(out)
        if rc != 0:
            print(out[-3000:])
            sys.exit('setup failed (exit %d); full output in %s' % (rc, log.log_path))

    src = os.path.join(build, 'accel-sim')
    rc, head = (1, '')
    if os.path.isdir(os.path.join(src, '.git')):
        rc, head = log.sh(['git', 'rev-parse', 'HEAD'], cwd=src)
    if rc != 0 or head.strip() != ACCEL_SIM_COMMIT:
        shutil.rmtree(src, ignore_errors=True)
        os.makedirs(src)
        step(['git', 'init', '-q'], src)
        step(['git', 'fetch', '-q', '--depth', '1', ACCEL_SIM_URL, ACCEL_SIM_COMMIT], src)
        step(['git', 'checkout', '-q', 'FETCH_HEAD'], src)
    tn = os.path.join(src, 'util', 'tracer_nvbit')
    if not os.path.isdir(os.path.join(tn, 'nvbit_release', 'core')):
        step(['bash', 'install_nvbit.sh'], tn)
    step(['make', '-C', 'tracer_tool', 'ARCH=%s' % a.arch], tn)
    step(['make', '-C', 'tracer_tool/traces-processing'], tn)
    step(['go', 'build', '-o', 'gpubench', './cmd/gpubench'], a.repo)
    log.say('setup done: accel-sim %s, tracer %s/tracer_tool/tracer_tool.so' % (ACCEL_SIM_COMMIT[:7], tn))
    return 0


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split('\n', 2)[2])
    ap.add_argument('mode', choices=['setup', 'demo', 'run', 'stats', 'report'])
    ap.add_argument('--root', default=DEFAULT_TRACE_ROOT,
                    help='trace root for run/stats/report (default: %(default)s)')
    ap.add_argument('--guidance-db', default=None,
                    help='timing DB whose ok configurations are traced '
                         '(default: <repo>/%s)' % DEFAULT_GUIDANCE_DB)
    ap.add_argument('--tiers', default='1,2')
    ap.add_argument('--benchmarks', default='', help='comma-separated subset (default: all)')
    ap.add_argument('--timeout', type=int, default=None,
                    help='per trace (and per post-processing), seconds (default: 600)')
    ap.add_argument('--min-free-gb', type=int, default=None,
                    help='stop when the trace root has less free space '
                         '(default: 200 for run on /scratch, 10 for demo in scripts/tmp)')
    ap.add_argument('--force', action='store_true', help='retrace configurations already ok/timed out')
    ap.add_argument('--retry-timeouts', action='store_true',
                    help='retrace only configurations that timed out (keep ok ones)')
    ap.add_argument('--max-timeouts-per-bench', type=int, default=0,
                    help='skip the rest of a benchmark after this many timeouts (0 = no limit; '
                         'default: %(default)s)')
    ap.add_argument('--skip-compile', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='print the plan only')
    ap.add_argument('--arch', default='sm_90', help='setup: tracer build arch (H100 = sm_90)')
    a = ap.parse_args()
    a.repo = os.path.dirname(here)
    if a.timeout is None:
        a.timeout = 600
    if a.min_free_gb is None:
        a.min_free_gb = 10 if a.mode == 'demo' else 200
    a.guidance_db = os.path.abspath(a.guidance_db or os.path.join(a.repo, DEFAULT_GUIDANCE_DB))
    a.benchmarks = set(x.strip() for x in a.benchmarks.split(',') if x.strip())
    try:
        if a.mode == 'setup':
            sys.exit(setup(a))
        if a.mode == 'demo':
            sys.exit(demo(a))
        if a.mode == 'report':
            sys.exit(report(a))
        col = Collector(a, os.path.abspath(a.root), a.mode)
        if a.mode == 'stats':
            col.stats()
            sys.exit(0)
        finished = col.run(col.configs(), force=a.force)
        col.stats()
        col.say('log: %s' % col.log_path)
        sys.exit(0 if finished else 130)
    except KeyboardInterrupt:
        sys.exit('interrupted; rerun the same command to resume')


if __name__ == '__main__':
    main()
