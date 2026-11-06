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
  .build/           Accel-Sim checkout, NVBit and the built tools

Usage (from the repo root on the server):
  python3 scripts/trace_collect.py setup     # build tracer, post-processor, gpubench
  python3 scripts/trace_collect.py demo      # 6 small configs into scripts/tmp/, verified
  python3 scripts/trace_collect.py run       # everything, into the trace root (tmux)
  python3 scripts/trace_collect.py stats     # print the table from index.db
Rerunning `run` resumes: configurations already ok or timed out are skipped
(--force retraces them); errors are retried.

Python 3.6+, stdlib only. Needs nvcc, nvidia-smi, xz, git, wget, make, bc,
libzstd-dev and Go (gpubench).
"""
import argparse
import csv
import datetime as dt
import decimal
import hashlib
import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time

ACCEL_SIM_URL = 'https://github.com/accel-sim/accel-sim-framework.git'
ACCEL_SIM_COMMIT = '7ba54ded60c4ba0083410c37d22d719166acd6dc'   # dev, 2026-10-27
DEFAULT_TRACE_ROOT = '/scratch/data/enze/gpu_benchmarks_trace'
DEFAULT_GUIDANCE_DB = os.path.join('ground_truth_h100_all', 'ground_truth_all.db')
DEMO_BENCHMARKS = ['fp32_throughput', 'memory_bandwidth', 'polybench_gemm',
                   'rodinia_hotspot', 'cuda_nbody', 'shoc_spmv']
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
    status                  TEXT NOT NULL,   -- ok | timeout | error
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

    @property
    def group(self):
        return (self.tier, self.bench, self.ns_json)

    def sort_key(self):
        v = self.sp_value
        return (self.tier, self.bench, self.ns_json,
                (0, v, '') if isinstance(v, (int, float)) else (1, 0, str(v)))


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
        for tier, bench, spn, spv, nsp in rows:
            if int(tier) not in tiers:
                continue
            if self.a.benchmarks and bench not in self.a.benchmarks:
                continue
            c = Config(tier, bench, spn, spv, json.loads(nsp or '{}'))
            seen[c.trace_id] = c
        return sorted(seen.values(), key=Config.sort_key)

    def demo_configs(self, allc):
        """Smallest size at the default non-scaling values of each demo benchmark."""
        out = []
        for bench in self.a.demo_benchmarks:
            cands = [c for c in allc if c.bench == bench]
            if not cands:
                self.say('demo: %s has no ok configuration in the guidance DB, skipped' % bench)
                continue
            p = os.path.join(self.repo, 'tier%d' % cands[0].tier, bench, PLATFORM, 'params.json')
            with open(p) as f:
                defaults = {q['name']: canon(q['default'])
                            for q in json.load(f).get('non_scaling_params', [])
                            if q['name'] not in MEASUREMENT_PARAMS}
            dflt = [c for c in cands if c.ns == defaults] or cands
            out.append(min(dflt, key=Config.sort_key))
        return out

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
        todo = [c for c in configs if force or status.get(c.trace_id) not in ('ok', 'timeout')]
        self.done = 0
        timed_out = set(c.group for c in configs if status.get(c.trace_id) == 'timeout')
        self.total = len(todo)
        self.say('%d configurations, %d to trace (%d already ok/timeout in index.db)' % (
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
            if c.group in timed_out:
                row.update(status='skipped_after_timeout', timestring_end=now_str(), time_span_sec=0,
                           error_message='a smaller size of this combination timed out')
                self.csv_write(row)
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
            " SUM(status='error'), SUM(CASE WHEN status='ok' THEN size_s0 END),"
            " SUM(CASE WHEN status='ok' THEN size_s1 END),"
            " SUM(CASE WHEN status='ok' THEN size_s2 END)"
            " FROM traces GROUP BY benchmark ORDER BY MIN(tier), benchmark").fetchall()
        con.close()
        head = ('benchmark', '#configs', 'S0_raw', 'S1_trace.xz', 'S2_tracez',
                'S1_ratio', 'S2_ratio', '#timeout', '#error')
        lines = []
        tot = [0, 0, 0, 0, 0, 0]
        for b, _, ok, to, er, s0, s1, s2 in rows:
            s0, s1, s2 = s0 or 0, s1 or 0, s2 or 0
            lines.append((b, ok, human(s0), human(s1), human(s2),
                          '%.6f' % (s1 / s0) if s0 else '-', '%.6f' % (s2 / s0) if s0 else '-',
                          to, er))
            for i, v in enumerate((ok, to, er, s0, s1, s2)):
                tot[i] += v or 0
        ok, to, er, s0, s1, s2 = tot
        lines.append(('TOTAL', ok, human(s0), human(s1), human(s2),
                      '%.6f' % (s1 / s0) if s0 else '-', '%.6f' % (s2 / s0) if s0 else '-', to, er))
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
    ok = [r for r in rows if r[1] == 'ok']
    problems = []
    for tid, st, nk in rows:
        d = os.path.join(col.data, tid, 'traces')
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
    if len(dirs) != len(rows):
        problems.append('data/ has %d entries, index.db has %d rows' % (len(dirs), len(rows)))
    col.stats()
    col.say('demo: %d/%d picked configurations traced ok, data/ entries %d, problems: %s' % (
        len(ok), len(picks), len(dirs), '; '.join(problems) or 'none'))
    col.say('log: %s' % col.log_path)
    return 0 if len(ok) == len(picks) and not problems else 1


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
    ap.add_argument('mode', choices=['setup', 'demo', 'run', 'stats'])
    ap.add_argument('--root', default=DEFAULT_TRACE_ROOT,
                    help='trace root for run/stats (default: %(default)s)')
    ap.add_argument('--guidance-db', default=None,
                    help='timing DB whose ok configurations are traced '
                         '(default: <repo>/%s)' % DEFAULT_GUIDANCE_DB)
    ap.add_argument('--tiers', default='1,2')
    ap.add_argument('--benchmarks', default='', help='comma-separated subset (default: all)')
    ap.add_argument('--demo-benchmarks', default=','.join(DEMO_BENCHMARKS))
    ap.add_argument('--timeout', type=int, default=3600, help='per trace (and per post-processing), seconds')
    ap.add_argument('--min-free-gb', type=int, default=None,
                    help='stop when the trace root has less free space '
                         '(default: 200 for run on /scratch, 10 for demo in scripts/tmp)')
    ap.add_argument('--force', action='store_true', help='retrace configurations already ok/timed out')
    ap.add_argument('--skip-compile', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='print the plan only')
    ap.add_argument('--arch', default='sm_90', help='setup: tracer build arch (H100 = sm_90)')
    a = ap.parse_args()
    a.repo = os.path.dirname(here)
    if a.min_free_gb is None:
        a.min_free_gb = 10 if a.mode == 'demo' else 200
    a.guidance_db = os.path.abspath(a.guidance_db or os.path.join(a.repo, DEFAULT_GUIDANCE_DB))
    a.benchmarks = set(x.strip() for x in a.benchmarks.split(',') if x.strip())
    a.demo_benchmarks = [x.strip() for x in a.demo_benchmarks.split(',') if x.strip()]
    try:
        if a.mode == 'setup':
            sys.exit(setup(a))
        if a.mode == 'demo':
            sys.exit(demo(a))
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
