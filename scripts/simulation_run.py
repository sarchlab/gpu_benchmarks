#!/usr/bin/env python3
"""Simulate the collected Accel-Sim 2.0 traces with MGPUSim NVIDIA (H100).

Reads the trace index written by trace_collect.py (index.db, status='ok' rows),
and for every traced configuration:
  1. symlinks its .tracez files into a scratch directory and expands each one
     to the .traceg text the simulator reads, with Accel-Sim's traceDsm (same
     pinned Accel-Sim checkout as the tracer);
  2. writes a kernelslist.g that points at the .traceg files;
  3. runs the simulator (`go build ./nvidia` of the mgpusim checkout, default
     device H100 @ 1755 MHz) and records its simulated cycles and time;
  4. deletes the scratch directory.
The kernel names of every launch (from the .tracez headers, demangled with
c++filt) are stored too, so a report can line them up with the kernels the
hardware ground truth timed.

Parallel: --jobs simulations at once (each is single-threaded). Only the main
process writes the database, the CSV and the logs, so nothing interleaves;
workers only run the subprocesses. Within one benchmark + non-scaling
combination the sizes run one after another, smallest first; once one times
out the larger sizes of that combination are skipped (as in trace_collect.py).
Different combinations run in parallel, heaviest first. New work is held back
while available memory or free disk is below --min-free-mem-gb / --min-free-gb.

Results: scripts/logs_simulation/simulation.db, table `sims`, one row per
(trace_id, sim_version). sim_version is the mgpusim commit (+ "-dirty" with
uncommitted changes to tracked files), so a new simulator build starts afresh
while rerunning the same build resumes: ok and timeout rows are kept, errors are
retried, skipped rows are decided again; --retry-timeouts also reruns timeouts. At the end of every run (also after Ctrl-C) the rows of
this sim_version are exported to scripts/logs_simulation/<stamp>.db; that
snapshot is what report.py reads.

Working files (git-ignored) under scripts/logs_simulation/:
  <stamp>_<mode>.log      driver log, ending with the summary table
  <stamp>_<mode>.csv      one row per configuration as it finishes (tail -f it)
  <stamp>_<mode>.sim.log  raw simulator / traceDsm output, one block per config
  latest.csv              symlink to the newest CSV
  simulation.db           all results so far (resume state)
  <stamp>.db              snapshot of this run's simulator version
  demo/simulation.db      the demo's own database (wiped by every demo)
Scratch space: <trace root>/sim_staging_<mode>/ (wiped at start).

Usage (from the repo root on the server):
  python3 scripts/simulation_run.py setup   # build traceDsm and the simulator
  python3 scripts/simulation_run.py demo    # smallest trace of every benchmark
  python3 scripts/simulation_run.py run     # everything (in tmux)
  python3 scripts/simulation_run.py stats   # summary of simulation.db
run and demo rebuild the simulator first (go build is incremental).

Python 3.6+, stdlib only. Needs Go, g++, libzstd-dev, c++filt, and the
Accel-Sim checkout that `trace_collect.py setup` made (scripts/.build/accel-sim).
"""
import argparse
import csv
import datetime as dt
import json
import os
import queue
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import trace_collect as tc  # noqa: E402  (shared helpers: fmt_value, human, paths)

DEFAULT_MGPUSIM = os.path.expanduser('~/workspace/mgpusim')
MGPUSIM_BRANCH = 'nvidia_ready_for_merge'
DEFAULT_GT_DB = os.path.join('ground_truth_h100_all', 'ground_truth_all.db')
DEVICE = 'H100'

CSV_COLS = [
    'progress', 'timestring_start', 'timestring_end', 'time_span_sec', 'tier', 'benchmark',
    'trace_id', 'scaling_param', 'scaling_value', 'non_scaling_params', 'status', 'n_kernels',
    'sim_cycles', 'sim_time_us', 'num_insts', 'traceg_bytes', 'convert_sec', 'sim_sec',
    'peak_rss_mb', 'running', 'free_mem_gb', 'free_disk_gb', 'error_message',
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS sims (
    trace_id                TEXT NOT NULL,
    sim_version             TEXT NOT NULL,   -- mgpusim commit (+ -dirty)
    tier                    INTEGER NOT NULL,
    benchmark               TEXT NOT NULL,
    scaling_param           TEXT NOT NULL,
    scaling_value           TEXT NOT NULL,
    non_scaling_params_json TEXT NOT NULL,
    config_json             TEXT NOT NULL,
    status                  TEXT NOT NULL,   -- ok | timeout | error | skipped_after_timeout
    device                  TEXT,
    sim_freq_mhz            REAL,
    sim_cycles              INTEGER,
    sim_time_us             REAL,
    num_kernels             INTEGER,         -- launches the simulator ran
    num_warps               INTEGER,
    num_insts               INTEGER,
    kernels_json            TEXT,            -- [{"name","mangled","launches"}], first-launch order
    trace_s0                INTEGER,         -- from index.db, for reference
    trace_s2                INTEGER,
    traceg_bytes            INTEGER,         -- expanded text the simulator read
    convert_sec             REAL,            -- traceDsm wall time
    sim_sec                 REAL,            -- simulator wall time
    wall_sec                REAL,
    peak_rss_mb             REAL,            -- simulator peak resident memory
    error_message           TEXT,
    stderr_tail             TEXT,
    mgpusim_branch          TEXT,
    gpu_benchmarks_commit   TEXT,
    host                    TEXT,
    started_at              TEXT,
    finished_at             TEXT,
    updated_at              TEXT NOT NULL,
    PRIMARY KEY (trace_id, sim_version)
);
CREATE INDEX IF NOT EXISTS sims_version ON sims(sim_version, benchmark, status);
CREATE TABLE IF NOT EXISTS runs (
    stamp        TEXT PRIMARY KEY,
    mode         TEXT,
    sim_version  TEXT,
    command_line TEXT,
    jobs         INTEGER,
    timeout_sec  INTEGER,
    trace_root   TEXT,
    started_at   TEXT,
    finished_at  TEXT,
    outcome      TEXT                -- finished | interrupted | stopped: <reason>
);
"""


def now_str():
    return dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def one_line(s, limit=300):
    return tc.one_line(s, limit)


def mem_available_gb():
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float('inf')     # not Linux: no guard


def wait_child(proc, deadline, stop_event):
    """Wait for proc with os.wait4; returns (status, rc, peak_rss_mb).

    status: 'done' | 'timeout' | 'interrupted'. The child runs in its own
    process group, which is killed on timeout or interruption.
    """
    while True:
        pid, st, ru = os.wait4(proc.pid, os.WNOHANG)
        if pid == proc.pid:
            rc = os.WEXITSTATUS(st) if os.WIFEXITED(st) else -os.WTERMSIG(st)
            proc.returncode = rc
            rss = ru.ru_maxrss / (1024.0 if sys.platform.startswith('linux') else 1024.0 ** 2)
            return 'done', rc, rss
        why = None
        if stop_event.is_set():
            why = 'interrupted'
        elif deadline and time.time() > deadline:
            why = 'timeout'
        if why:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            _, st, ru = os.wait4(proc.pid, 0)
            proc.returncode = -9
            return why, -9, ru.ru_maxrss / (1024.0 if sys.platform.startswith('linux') else 1024.0 ** 2)
        time.sleep(0.5)


def tracez_header(path):
    """The null-terminated text header at the start of a .tracez file."""
    out = bytearray()
    with open(path, 'rb') as f:
        while len(out) < (1 << 20):
            chunk = f.read(4096)
            if not chunk:
                break
            i = chunk.find(b'\0')
            if i >= 0:
                out += chunk[:i]
                break
            out += chunk
    return out.decode('utf-8', 'replace')


def demangle(names):
    """Mangled -> plain function name (no namespace, template args or params)."""
    uniq = sorted(set(names))
    plain = {}
    try:
        p = subprocess.run(['c++filt'], input='\n'.join(uniq) + '\n', stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, universal_newlines=True, timeout=60)
        dem = p.stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        dem = uniq
    if len(dem) != len(uniq):
        dem = uniq
    for m, d in zip(uniq, dem):
        s = d.split('(', 1)[0]
        depth, base = 0, ''
        for ch in s:            # drop template arguments
            if ch == '<':
                depth += 1
            elif ch == '>':
                depth -= 1
            elif depth == 0:
                base += ch
        base = base.strip().split('::')[-1].split()[-1] if base.strip() else m
        plain[m] = base
    return plain


class Job:
    """One traced configuration from index.db."""

    def __init__(self, row, rank):
        self.trace_id = row['trace_id']
        self.tier = row['tier']
        self.bench = row['benchmark']
        self.sp_name = row['scaling_param']
        self.sp_value = row['scaling_value']
        self.ns_json = row['non_scaling_params_json']
        self.config_json = row['config_json']
        self.data_path = row['data_path'] or os.path.join('data', row['trace_id'])
        self.s0 = row['size_s0'] or 0
        self.s2 = row['size_s2'] or 0
        self.n_kernels = row['n_kernels'] or 0
        self.rank = rank
        self.group = (self.tier, self.bench, self.ns_json)


class Simulator:
    def __init__(self, a, mode):
        self.a = a
        self.mode = mode
        self.repo = a.repo
        self.logs = os.path.join(self.repo, 'scripts', 'logs_simulation')
        os.makedirs(self.logs, exist_ok=True)
        self.build = os.path.join(self.repo, 'scripts', '.build')
        self.sim_bin = os.path.join(self.build, 'mgpusim_nvidia')
        self.dsm_dir = os.path.join(self.build, 'accel-sim', 'util', 'tracer_nvbit', 'others',
                                    'traceDsm_tool')
        self.dsm_bin = os.path.join(self.dsm_dir, 'traceDsm')
        self.trace_root = os.path.abspath(a.trace_root)
        self.index_path = os.path.join(self.trace_root, 'index.db')
        if mode == 'demo':
            self.db_path = os.path.join(self.logs, 'demo', 'simulation.db')
        else:
            self.db_path = os.path.join(self.logs, 'simulation.db')
        self.staging = os.path.join(self.trace_root, 'sim_staging_%s' % mode)
        self.stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        tag = '%s_%s' % (self.stamp, mode)
        self.log_path = os.path.join(self.logs, tag + '.log')
        self.csv_path = os.path.join(self.logs, tag + '.csv')
        self.simlog_path = os.path.join(self.logs, tag + '.sim.log')
        self.snapshot_path = os.path.join(self.logs, self.stamp + ('_demo' if mode == 'demo' else '') + '.db')
        self.log_f = open(self.log_path, 'a')
        self.stop_event = threading.Event()
        self.version = None
        self.host = socket.gethostname()

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
        latest = os.path.join(self.logs, 'latest.csv')
        try:
            if os.path.islink(latest) or os.path.exists(latest):
                os.remove(latest)
            os.symlink(os.path.basename(self.csv_path), latest)
        except OSError:
            pass

    def sh(self, cmd, cwd=None, timeout=None):
        p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=timeout)
        return p.returncode, p.stdout

    # ── build ─────────────────────────────────────────────────────────────
    def mgpusim_version(self):
        m = self.a.mgpusim
        rc, branch = self.sh(['git', '-C', m, 'branch', '--show-current'])
        rc2, head = self.sh(['git', '-C', m, 'rev-parse', 'HEAD'])
        _, dirty = self.sh(['git', '-C', m, 'status', '--porcelain', '--untracked-files=no'])
        if rc2 != 0:
            raise SystemExit('cannot read the mgpusim checkout at %s' % m)
        return branch.strip(), head.strip()[:12] + ('-dirty' if dirty.strip() else '')

    def build_all(self):
        branch, version = self.mgpusim_version()
        if branch != MGPUSIM_BRANCH and not self.a.any_branch:
            raise SystemExit('mgpusim is on branch %r, expected %s (pass --any-branch to override)'
                             % (branch, MGPUSIM_BRANCH))
        self.branch, self.version = branch, version
        if not self.a.skip_build:
            self.say('building simulator: go build -o %s ./nvidia (in %s, %s @ %s)'
                     % (self.sim_bin, self.a.mgpusim, branch, version))
            os.makedirs(self.build, exist_ok=True)
            rc, out = self.sh(['go', 'build', '-o', self.sim_bin, './nvidia'], cwd=self.a.mgpusim,
                              timeout=1800)
            if rc != 0:
                raise SystemExit('simulator build failed:\n' + out[-3000:])
        if not os.path.isfile(self.sim_bin):
            raise SystemExit('no simulator binary at %s (run without --skip-build)' % self.sim_bin)
        if not os.path.isfile(self.dsm_bin):
            if not os.path.isdir(self.dsm_dir):
                raise SystemExit('Accel-Sim checkout missing (%s); run `python3 scripts/'
                                 'trace_collect.py setup` first' % self.dsm_dir)
            self.say('building traceDsm in %s' % self.dsm_dir)
            rc, out = self.sh(['make', '-C', self.dsm_dir], timeout=600)
            if rc != 0:
                raise SystemExit('traceDsm build failed:\n' + out[-3000:])
        if shutil.which('c++filt') is None:
            self.say('WARNING: c++filt not found; kernel names stay mangled')
        _, head = self.sh(['git', '-C', self.repo, 'rev-parse', 'HEAD'])
        self.gb_commit = head.strip()[:12]

    # ── databases ─────────────────────────────────────────────────────────
    def db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        con = sqlite3.connect(self.db_path, timeout=60)
        con.executescript(SCHEMA)
        return con

    def jobs_from_index(self):
        if not os.path.isfile(self.index_path):
            raise SystemExit('no index.db under %s' % self.trace_root)
        con = sqlite3.connect('file:%s?mode=ro' % self.index_path, uri=True, timeout=60)
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute("SELECT * FROM traces WHERE status='ok'")]
        con.close()
        tiers = set(int(t) for t in self.a.tiers.split(',') if t.strip())
        ranks = {}
        jobs = []
        for r in rows:
            if r['tier'] not in tiers:
                continue
            if self.a.benchmarks and r['benchmark'] not in self.a.benchmarks:
                continue
            key = (r['tier'], r['benchmark'])
            if key not in ranks:
                ranks[key] = tc.scaling_ranks(self.repo, r['tier'], r['benchmark'])
            jobs.append(Job(r, ranks[key].get(r['scaling_value'], 1 << 30)))
        jobs.sort(key=lambda j: (j.tier, j.bench, j.ns_json, j.rank, j.sp_value))
        return jobs

    def done_status(self):
        if not os.path.isfile(self.db_path):
            return {}
        con = self.db()
        rows = dict(con.execute('SELECT trace_id, status FROM sims WHERE sim_version=?',
                                (self.version,)).fetchall())
        con.close()
        return rows

    def upsert(self, job, fields):
        row = {
            'trace_id': job.trace_id, 'sim_version': self.version, 'tier': job.tier,
            'benchmark': job.bench, 'scaling_param': job.sp_name, 'scaling_value': job.sp_value,
            'non_scaling_params_json': job.ns_json, 'config_json': job.config_json,
            'trace_s0': job.s0, 'trace_s2': job.s2, 'mgpusim_branch': self.branch,
            'gpu_benchmarks_commit': self.gb_commit, 'host': self.host, 'updated_at': now_str(),
        }
        row.update(fields)
        con = self.db()
        con.execute('INSERT OR REPLACE INTO sims (%s) VALUES (%s)'
                    % (','.join(row), ','.join('?' * len(row))), list(row.values()))
        con.commit()
        con.close()

    def record_run(self, outcome=None):
        con = self.db()
        if outcome is None:
            con.execute('INSERT OR REPLACE INTO runs (stamp, mode, sim_version, command_line, jobs,'
                        ' timeout_sec, trace_root, started_at) VALUES (?,?,?,?,?,?,?,?)',
                        (self.stamp, self.mode, self.version, ' '.join(sys.argv), self.a.jobs,
                         self.a.timeout, self.trace_root, now_str()))
        else:
            con.execute('UPDATE runs SET finished_at=?, outcome=? WHERE stamp=?',
                        (now_str(), outcome, self.stamp))
        con.commit()
        con.close()

    def snapshot(self):
        """Export this sim_version's rows (and the runs that made them)."""
        if os.path.exists(self.snapshot_path):
            os.remove(self.snapshot_path)
        con = self.db()
        con.execute('ATTACH DATABASE ? AS snap', (self.snapshot_path,))
        for stmt in SCHEMA.replace('CREATE TABLE IF NOT EXISTS ', 'CREATE TABLE IF NOT EXISTS snap.') \
                          .replace('CREATE INDEX IF NOT EXISTS sims_version ON sims',
                                   'CREATE INDEX IF NOT EXISTS snap.sims_version ON sims').split(';'):
            if stmt.strip():
                con.execute(stmt)
        con.execute('INSERT INTO snap.sims SELECT * FROM main.sims WHERE sim_version=?',
                    (self.version,))
        con.execute('INSERT INTO snap.runs SELECT * FROM main.runs WHERE sim_version=?',
                    (self.version,))
        con.commit()
        n = con.execute('SELECT COUNT(*) FROM snap.sims').fetchone()[0]
        con.execute('DETACH DATABASE snap')
        con.close()
        self.say('snapshot: %s (%d rows, sim_version %s)' % (self.snapshot_path, n, self.version))

    # ── one configuration (runs in a worker thread) ───────────────────────
    def simulate(self, job):
        """Returns (status, fields, message, raw_output). No shared state touched."""
        t0 = time.time()
        deadline = t0 + self.a.timeout
        src = os.path.join(self.trace_root, job.data_path, 'traces')
        work = os.path.join(self.staging, job.trace_id)
        raw = []
        shutil.rmtree(work, ignore_errors=True)
        try:
            os.makedirs(work)
            klist_src = os.path.join(src, 'kernelslist.g')
            if not os.path.isfile(klist_src):
                return 'error', {}, 'no kernelslist.g in %s' % src, ''
            with open(klist_src) as f:
                lines = [x.rstrip('\n') for x in f]
            tz = [x.strip() for x in lines if x.strip().endswith('.tracez')]
            launches = []
            for name in tz:
                hdr = tracez_header(os.path.join(src, name))
                m = re.search(r'^-kernel name = (.+)$', hdr, re.M)
                launches.append(m.group(1).strip() if m else name)
            # expand .tracez -> .traceg (traceDsm writes next to its input path)
            for name in tz:
                os.symlink(os.path.join(src, name), os.path.join(work, name))
                out_path = os.path.join(work, 'traceDsm.out')
                with open(out_path, 'w') as of:
                    p = subprocess.Popen([self.dsm_bin, os.path.join(work, name)], cwd=work,
                                         stdout=of, stderr=subprocess.STDOUT, preexec_fn=os.setpgrp)
                    st, rc, _ = wait_child(p, deadline, self.stop_event)
                with open(out_path, errors='replace') as of:
                    o = of.read()
                if st != 'done':
                    return st, {'convert_sec': round(time.time() - t0, 1)}, \
                        ('timeout after %ds (while expanding .tracez)' % self.a.timeout
                         if st == 'timeout' else st), o[-2000:]
                if rc != 0:
                    return 'error', {'convert_sec': round(time.time() - t0, 1)}, \
                        'traceDsm exit %d on %s: %s' % (rc, name, one_line(o[-300:])), o[-4000:]
                os.remove(os.path.join(work, name))
            traceg = sum(os.path.getsize(os.path.join(work, n[:-len('.tracez')] + '.traceg'))
                         for n in tz)
            with open(os.path.join(work, 'kernelslist.g'), 'w') as f:
                for x in lines:
                    s = x.strip()
                    f.write((s[:-len('.tracez')] + '.traceg' if s.endswith('.tracez') else x) + '\n')
            t1 = time.time()
            convert = round(t1 - t0, 1)
            # simulate
            out_path = os.path.join(work, 'sim.out')
            with open(out_path, 'w') as of:
                p = subprocess.Popen([self.sim_bin, '-trace-dir', work, '-device', DEVICE],
                                     cwd=work, stdout=of, stderr=subprocess.STDOUT,
                                     preexec_fn=os.setpgrp)
                st, rc, rss = wait_child(p, deadline, self.stop_event)
            sim_sec = round(time.time() - t1, 1)
            with open(out_path, errors='replace') as of:
                out = of.read()
            raw.append(out if len(out) < 200000 else out[:100000] + '\n...\n' + out[-100000:])
            fields = {'convert_sec': convert, 'sim_sec': sim_sec, 'traceg_bytes': traceg,
                      'peak_rss_mb': round(rss, 1), 'stderr_tail': out[-2000:]}
            if st != 'done':
                return st, fields, ('timeout after %ds' % self.a.timeout if st == 'timeout'
                                    else st), '\n'.join(raw)
            if rc != 0:
                return 'error', fields, 'simulator exit %d: %s' % (rc, one_line(out[-400:])), \
                    '\n'.join(raw)

            def grab(pat, cast):
                m = re.search(pat, out, re.M)
                return cast(m.group(1)) if m else None
            fields.update(
                device=grab(r'^device:\s+(\S+)', str),
                sim_freq_mhz=grab(r'^device:\s+\S+ @ ([\d.]+) MHz', float),
                num_kernels=grab(r'^kernels:\s+(\d+)', int),
                num_warps=grab(r'^warps:\s+(\d+)', int),
                num_insts=grab(r'^instructions:\s+(\d+)', int),
                sim_time_us=grab(r'^simulated time:\s+([\d.]+) us', float),
                sim_cycles=grab(r'^simulated cycles:\s+(\d+)', int),
            )
            if fields['sim_cycles'] is None:
                return 'error', fields, 'no "simulated cycles" in simulator output', '\n'.join(raw)
            fields['launch_names'] = launches
            return 'ok', fields, '', '\n'.join(raw)
        except Exception as e:  # keep the run going; the row says what broke
            return 'error', {}, '%s: %s' % (type(e).__name__, e), '\n'.join(raw)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    # ── main loop ─────────────────────────────────────────────────────────
    def run(self, jobs, force=False):
        self.say('mgpusim %s @ %s, simulator %s, device %s' % (self.branch, self.version,
                                                              self.sim_bin, DEVICE))
        self.say('trace root %s, db %s, jobs %d, timeout %ds, min free mem %d GB, min free disk %d GB'
                 % (self.trace_root, self.db_path, self.a.jobs, self.a.timeout,
                    self.a.min_free_mem_gb, self.a.min_free_gb))
        self.say('csv: %s' % self.csv_path)
        status = self.done_status()
        # skipped_after_timeout rows are decided again on every run
        skip = ('ok',) if self.a.retry_timeouts else ('ok', 'timeout')
        todo = [j for j in jobs if force or status.get(j.trace_id) not in skip]
        fresh = force or self.a.retry_timeouts
        timed_out = set() if fresh else \
            set(j.group for j in jobs if status.get(j.trace_id) == 'timeout')
        self.say('%d traced configurations, %d to simulate (%d already done for this sim_version)'
                 % (len(jobs), len(todo), len(jobs) - len(todo)))
        if self.a.dry_run:
            for j in todo:
                self.say('  plan %s %s=%s %s (S0 %s)' % (j.trace_id, j.sp_name, j.sp_value,
                                                         j.ns_json, tc.human(j.s0)))
            return True
        if not todo:
            return True
        shutil.rmtree(self.staging, ignore_errors=True)
        os.makedirs(self.staging)
        self.record_run()
        # group chains, heaviest first
        chains = {}
        for j in todo:
            chains.setdefault(j.group, []).append(j)
        order = sorted(chains, key=lambda g: -sum(j.s0 for j in chains[g]))
        pending = [chains[g] for g in order]          # list of lists, consumed in order
        results = queue.Queue()
        running = {}                                   # thread -> (job, start str, start time)
        busy_groups = set()
        total, done = len(todo), 0
        outcome = 'finished'
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)

        def worker(job):
            results.put((job, self.simulate(job)))

        def next_job():
            for chain in pending:
                if chain and chain[0].group not in busy_groups:
                    return chain
            return None

        def write_row(job, st, fields, msg, started, t_start):
            row = {'progress': '%d/%d' % (done, total), 'timestring_start': started,
                   'timestring_end': now_str(), 'time_span_sec': int(time.time() - t_start),
                   'tier': job.tier, 'benchmark': job.bench, 'trace_id': job.trace_id,
                   'scaling_param': job.sp_name, 'scaling_value': job.sp_value,
                   'non_scaling_params': job.ns_json, 'status': st, 'n_kernels': job.n_kernels,
                   'running': len(running), 'free_mem_gb': '%.0f' % min(mem_available_gb(), 1e6),
                   'free_disk_gb': '%.0f' % (shutil.disk_usage(self.trace_root).free / 1e9),
                   'error_message': one_line(msg)}
            for k in ('sim_cycles', 'sim_time_us', 'num_insts', 'traceg_bytes', 'convert_sec',
                      'sim_sec', 'peak_rss_mb'):
                row[k] = fields.get(k, '')
            self.csv_write(row)

        try:
            while True:
                # dispatch while there is room and resources
                while len(running) < self.a.jobs and not self.stop_event.is_set():
                    chain = next_job()
                    if chain is None:
                        break
                    job = chain[0]
                    if job.group in timed_out:
                        chain.pop(0)
                        done += 1
                        if status.get(job.trace_id) in (None, 'error', 'skipped_after_timeout'):
                            # never replaces a real result (ok / timeout)
                            self.upsert(job, {'status': 'skipped_after_timeout',
                                              'error_message': 'a smaller size of this combination timed out',
                                              'updated_at': now_str()})
                            status[job.trace_id] = 'skipped_after_timeout'
                        write_row(job, 'skipped_after_timeout', {},
                                  'a smaller size of this combination timed out', now_str(), time.time())
                        self.say('[%d/%d] skip %s (smaller size timed out)' % (done, total, job.trace_id))
                        continue
                    mem = mem_available_gb()
                    disk = shutil.disk_usage(self.trace_root).free / 1e9
                    if mem < self.a.min_free_mem_gb or disk < self.a.min_free_gb:
                        if not running:
                            outcome = 'stopped: %.0f GB memory / %.0f GB disk free' % (mem, disk)
                            self.say('STOP: ' + outcome)
                            self.stop_event.set()
                        break
                    chain.pop(0)
                    busy_groups.add(job.group)
                    t = threading.Thread(target=worker, args=(job,), daemon=True)
                    running[job.trace_id] = (job, now_str(), time.time())
                    t.start()
                pending[:] = [c for c in pending if c]
                if not running and (not pending or self.stop_event.is_set()):
                    break
                try:
                    job, (st, fields, msg, raw) = results.get(timeout=5)
                except queue.Empty:
                    continue
                _, started, t_start = running.pop(job.trace_id)
                busy_groups.discard(job.group)
                if st == 'interrupted':
                    continue
                done += 1
                if st == 'timeout':
                    timed_out.add(job.group)
                names = fields.pop('launch_names', None)
                if names is not None:
                    plain = demangle(names)
                    kern = []
                    for n in names:
                        if kern and kern[-1]['mangled'] == n:
                            kern[-1]['launches'] += 1
                            continue
                        for k in kern:
                            if k['mangled'] == n:
                                k['launches'] += 1
                                break
                        else:
                            kern.append({'name': plain.get(n, n), 'mangled': n, 'launches': 1})
                    fields['kernels_json'] = json.dumps(kern, separators=(',', ':'))
                fields.update(status=st, error_message=msg or None, started_at=started,
                              finished_at=now_str(), wall_sec=round(time.time() - t_start, 1))
                if st == 'ok' or status.get(job.trace_id) != 'ok':
                    self.upsert(job, fields)
                    status[job.trace_id] = st
                with open(self.simlog_path, 'a') as f:
                    f.write('\n===== %s %s %s %s=%s %s (%s)\n%s\n' % (
                        now_str(), job.trace_id, job.bench, job.sp_name, job.sp_value, job.ns_json,
                        st, raw))
                write_row(job, st, fields, msg, started, t_start)
                if st == 'ok':
                    extra = '%s cycles (%.1f us), %s insts, sim %ss, %.0f MB' % (
                        fields['sim_cycles'], fields['sim_time_us'] or 0, fields.get('num_insts'),
                        fields['sim_sec'], fields.get('peak_rss_mb') or 0)
                else:
                    extra = one_line(msg, 200)
                self.say('[%d/%d] %s %s %s=%s %s  %s  (running %d)' % (
                    done, total, st, job.trace_id, job.sp_name, job.sp_value, job.ns_json, extra,
                    len(running)))
        except KeyboardInterrupt:
            outcome = 'interrupted'
            self.say('Ctrl-C: killing %d running simulations; rerun the same command to resume'
                     % len(running))
            self.stop_event.set()
            deadline = time.time() + 60
            while running and time.time() < deadline:
                try:
                    job, _ = results.get(timeout=2)
                    running.pop(job.trace_id, None)
                except queue.Empty:
                    pass
        finally:
            shutil.rmtree(self.staging, ignore_errors=True)
            self.record_run(outcome)
        return outcome == 'finished'

    # ── summary ───────────────────────────────────────────────────────────
    def stats(self, version=None):
        if not os.path.isfile(self.db_path):
            self.say('no %s yet' % self.db_path)
            return
        con = self.db()
        if version is None:
            r = con.execute('SELECT sim_version FROM sims ORDER BY updated_at DESC LIMIT 1').fetchone()
            if not r:
                self.say('simulation.db is empty')
                return
            version = r[0]
        rows = con.execute(
            "SELECT benchmark, MIN(tier), SUM(status='ok'), SUM(status='timeout'), SUM(status='error'),"
            " SUM(status='skipped_after_timeout'), SUM(CASE WHEN status='ok' THEN num_insts END),"
            " SUM(sim_sec), MAX(peak_rss_mb) FROM sims WHERE sim_version=?"
            " GROUP BY benchmark ORDER BY MIN(tier), benchmark", (version,)).fetchall()
        con.close()
        head = ('benchmark', '#ok', '#timeout', '#error', '#skipped', 'insts(ok)', 'sim_time',
                'insts/s', 'peak_mem')
        lines, tot = [], [0, 0, 0, 0, 0, 0.0, 0.0]
        for b, _, ok, to, er, sk, ins, sec, rss in rows:
            ins, sec = ins or 0, sec or 0.0
            lines.append((b, ok, to, er, sk, ins, tc.hms(sec), '%.0f' % (ins / sec) if sec else '-',
                          '%.0f MB' % rss if rss else '-'))
            for i, v in enumerate((ok, to, er, sk, ins, sec, rss or 0)):
                tot[i] = max(tot[i], v) if i == 6 else tot[i] + (v or 0)
        lines.append(('TOTAL', tot[0], tot[1], tot[2], tot[3], tot[4], tc.hms(tot[5]),
                      '%.0f' % (tot[4] / tot[5]) if tot[5] else '-', '%.0f MB' % tot[6]))
        widths = [max(len(str(r[i])) for r in [head] + lines) for i in range(len(head))]
        fmt = '  '.join(('%%-%ds' if i == 0 else '%%%ds') % w for i, w in enumerate(widths))
        self.say('simulation statistics, sim_version %s (%s):\n%s' % (
            version, self.db_path, '\n'.join([fmt % head] + [fmt % tuple(r) for r in lines])))


def gt_kernel_check(sim, gt_db, version):
    """Demo aid: how well do simulated kernel names line up with the timed ones?"""
    if not os.path.isfile(gt_db):
        sim.say('kernel-name check skipped: no %s' % gt_db)
        return
    con = sqlite3.connect(sim.db_path)
    rows = con.execute("SELECT trace_id, benchmark, scaling_param, scaling_value,"
                       " non_scaling_params_json, kernels_json, tier FROM sims"
                       " WHERE sim_version=? AND status='ok'", (version,)).fetchall()
    con.close()
    g = sqlite3.connect('file:%s?mode=ro' % gt_db, uri=True)
    sim.say('kernel names, simulated trace vs ground truth (timed):')
    for tid, b, sp, sv, ns, kj, tier in rows:
        want = None
        for rid, nsj in g.execute(
                "SELECT id, non_scaling_params_json FROM runs WHERE status='ok' AND benchmark=?"
                " AND scaling_param_value=? AND repetition=1", (b, sv)):
            c = tc.Config(tier, b, sp, sv, json.loads(nsj or '{}'))
            if c.trace_id == tid:
                want = set(n for (n,) in g.execute(
                    'SELECT DISTINCT kernel_name FROM kernel_results WHERE run_id=?', (rid,)))
                break
        have = [k['name'] for k in json.loads(kj or '[]')]
        if want is None:
            sim.say('  %-28s no ground-truth run found' % b)
            continue
        both = [n for n in have if n in want]
        extra = [n for n in have if n not in want]
        missing = sorted(want - set(have))
        tag = 'match' if not extra and not missing else ('extra' if not missing else 'MISMATCH')
        sim.say('  %-28s %-8s trace %s | truth %s%s' % (
            b, tag, ','.join(have)[:80], ','.join(sorted(want))[:60],
            (' | timed but not in trace: %s' % ','.join(missing)) if missing else ''))
    g.close()


def demo_pick(jobs):
    """Smallest trace (S0) of every benchmark."""
    best = {}
    for j in jobs:
        k = (j.tier, j.bench)
        if k not in best or (j.s0, j.rank) < (best[k].s0, best[k].rank):
            best[k] = j
    return [best[k] for k in sorted(best)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split('\n', 2)[2])
    ap.add_argument('mode', choices=['setup', 'demo', 'run', 'stats'])
    ap.add_argument('--trace-root', default=tc.DEFAULT_TRACE_ROOT,
                    help='holds index.db and data/ (default: %(default)s)')
    ap.add_argument('--mgpusim', default=DEFAULT_MGPUSIM, help='mgpusim checkout (default: %(default)s)')
    ap.add_argument('--gt-db', default=None, help='ground-truth DB for the demo kernel-name check '
                    '(default: <repo>/%s)' % DEFAULT_GT_DB)
    ap.add_argument('--jobs', type=int, default=None, help='simulations at once (default: 16, demo 8)')
    ap.add_argument('--timeout', type=int, default=None,
                    help='per configuration, seconds (default: 3600, demo 600)')
    ap.add_argument('--min-free-mem-gb', type=int, default=32,
                    help='hold new work below this much available memory (default: %(default)s)')
    ap.add_argument('--min-free-gb', type=int, default=100,
                    help='hold new work below this much free disk on the trace root (default: %(default)s)')
    ap.add_argument('--tiers', default='1,2')
    ap.add_argument('--benchmarks', default='', help='comma-separated subset (default: all)')
    ap.add_argument('--force', action='store_true', help='resimulate everything for this sim_version')
    ap.add_argument('--retry-timeouts', action='store_true',
                    help='resimulate only the timed-out configurations (keep ok ones)')
    ap.add_argument('--skip-build', action='store_true', help='use the existing simulator binary')
    ap.add_argument('--any-branch', action='store_true',
                    help='allow an mgpusim branch other than %s' % MGPUSIM_BRANCH)
    ap.add_argument('--dry-run', action='store_true', help='print the plan only')
    a = ap.parse_args()
    a.repo = os.path.dirname(HERE)
    a.mgpusim = os.path.abspath(os.path.expanduser(a.mgpusim))
    a.gt_db = os.path.abspath(a.gt_db or os.path.join(a.repo, DEFAULT_GT_DB))
    a.benchmarks = set(x.strip() for x in a.benchmarks.split(',') if x.strip())
    if a.jobs is None:
        a.jobs = 8 if a.mode == 'demo' else 16
    if a.timeout is None:
        a.timeout = 600 if a.mode == 'demo' else 3600
    sim = Simulator(a, a.mode)
    if a.mode == 'stats':
        sim.stats()
        return 0
    sim.build_all()
    if a.mode == 'setup':
        sim.say('setup done: simulator %s, traceDsm %s, sim_version %s'
                % (sim.sim_bin, sim.dsm_bin, sim.version))
        return 0
    jobs = sim.jobs_from_index()
    if a.mode == 'demo':
        shutil.rmtree(os.path.dirname(sim.db_path), ignore_errors=True)
        jobs = demo_pick(jobs)
        sim.say('demo: %d benchmarks, smallest trace of each' % len(jobs))
    finished = sim.run(jobs, force=a.force)
    if a.dry_run:
        return 0
    sim.stats(sim.version)
    if a.mode == 'demo':
        gt_kernel_check(sim, a.gt_db, sim.version)
        con = sqlite3.connect(sim.db_path)
        r = con.execute("SELECT MAX(peak_rss_mb), SUM(num_insts)/SUM(sim_sec) FROM sims"
                        " WHERE status='ok' AND sim_version=?", (sim.version,)).fetchone()
        con.close()
        if r and r[0]:
            sim.say('demo: peak simulator memory %.0f MB, %.0f instructions/s per simulation'
                    % (r[0], r[1] or 0))
    sim.snapshot()
    sim.say('log: %s' % sim.log_path)
    return 0 if finished else 130


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit('interrupted; rerun the same command to resume')
