#!/usr/bin/env python3
"""All-in-one HTML report: hardware ground truth vs MGPUSim simulation.

Reads three databases and writes one self-contained page with four tabs:
  Report      simulated vs hardware cycles, per suite and benchmark
  Trace       the trace collection (same page as `trace_collect.py report`)
  Profile     the hardware ground truth (gpubench timing runs)
  Simulation  the simulation runs

Usage (from the repo root):
  python3 scripts/report.py [--gt-db GT] [--trace-db INDEX] [--sim-db SIM]
Defaults: ground_truth_h100_all/ground_truth_all.db,
/scratch/data/enze/gpu_benchmarks_trace/index.db, and the newest
scripts/logs_simulation/<stamp>.db snapshot (not a demo one).
Output: scripts/logs_report/report_<stamp>.html (printed at the end).

Matching: the unit is a configuration (benchmark + scaling value + other
parameters), keyed by trace_id. Hardware time = median over repetitions of
runs.total_time_ms (the benchmark's own timing of its kernels), converted to
cycles at the simulator's clock (sim_freq_mhz, 1755 MHz for H100). Metrics:
MRE (mean signed relative error, (sim - hw) / hw) and MARE (mean |relative
error|) over a benchmark's configurations; LSP (Pearson of log sim vs log hw)
and MALGE (mean |dlog sim - dlog hw| between neighbouring scaling values) per
trajectory (same non-scaling parameters), averaged over a benchmark's
trajectories. Suite = mean of its benchmarks; page = mean of all benchmarks.
Kernel names: the simulated trace's kernels are compared
with the timed ones (a trailing _<n> on a timed name is ignored when only the
base name is in the trace, e.g. mm_kernel_1 -> mm_kernel).
"""
import argparse
import datetime as dt
import glob
import html
import json
import os
import re
import sqlite3
import statistics
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import trace_collect as tc  # noqa: E402

REPO = os.path.dirname(HERE)
DEFAULT_GT = os.path.join(REPO, 'ground_truth_h100_all', 'ground_truth_all.db')
DEFAULT_INDEX = os.path.join(tc.DEFAULT_TRACE_ROOT, 'index.db')
SIM_LOGS = os.path.join(REPO, 'scripts', 'logs_simulation')
OUT_DIR = os.path.join(REPO, 'scripts', 'logs_report')
e = html.escape


def ro(path):
    con = sqlite3.connect('file:%s?mode=ro' % os.path.abspath(path), uri=True, timeout=60)
    con.row_factory = sqlite3.Row
    return con


def newest_snapshot():
    snaps = [p for p in glob.glob(os.path.join(SIM_LOGS, '2*.db')) if not p.endswith('_demo.db')]
    return max(snaps) if snaps else None


def pct(x, signed=True):
    if x is None:
        return '-'
    v = x * 100
    s = '{:,.2f}'.format(abs(v))
    return (('+' if v >= 0 else '−') if signed else '') + s + '%'


def num(n):
    return '{:,}'.format(int(n))


def ms(v):
    """A time in milliseconds, in a readable unit."""
    if v is None:
        return '-'
    if v < 1:
        return '%.3g µs' % (v * 1000)
    if v < 1000:
        return '%.3g ms' % v
    return '%.3g s' % (v / 1000)


def secs(s):
    return tc.hms(s or 0)


# ── load ─────────────────────────────────────────────────────────────────
def load_gt(path):
    """Per trace_id: config, suite, median time, reps, kernel names; plus run stats."""
    con = ro(path)
    runs = [dict(r) for r in con.execute(
        'SELECT id, session_id, tier, suite, benchmark, scaling_param_name, scaling_param_value,'
        ' non_scaling_params_json, repetition, status, error_message, total_time_ms, started_at,'
        ' finished_at FROM runs')]
    kern = {}
    for rid, name in con.execute('SELECT DISTINCT run_id, kernel_name FROM kernel_results'):
        kern.setdefault(rid, set()).add(name)
    plat = con.execute('SELECT * FROM platforms ORDER BY id LIMIT 1').fetchone()
    con.close()
    cfgs = {}
    for r in runs:
        c = tc.Config(r['tier'], r['benchmark'], r['scaling_param_name'], r['scaling_param_value'],
                      json.loads(r['non_scaling_params_json'] or '{}'))
        g = cfgs.setdefault(c.trace_id, {
            'trace_id': c.trace_id, 'tier': c.tier, 'suite': r['suite'], 'benchmark': c.bench,
            'sp': c.sp_name, 'sv': tc.fmt_value(c.sp_value), 'ns_json': c.ns_json,
            'times': [], 'errors': 0, 'kernels': set(), 'wall': 0.0, 'error_message': None})
        try:
            t0 = dt.datetime.strptime(r['started_at'], '%Y-%m-%d %H:%M:%S')
            t1 = dt.datetime.strptime(r['finished_at'], '%Y-%m-%d %H:%M:%S')
            g['wall'] += (t1 - t0).total_seconds()
        except (TypeError, ValueError):
            pass
        if r['status'] == 'ok' and r['total_time_ms'] is not None:
            g['times'].append(r['total_time_ms'])
            g['kernels'] |= kern.get(r['id'], set())
        else:
            g['errors'] += 1
            g['error_message'] = g['error_message'] or r['error_message']
    for g in cfgs.values():
        t = g['times']
        g['median_ms'] = statistics.median(t) if t else None
        g['cv'] = (statistics.pstdev(t) / statistics.mean(t)) if len(t) > 1 and statistics.mean(t) else None
    span = (min((r['started_at'] for r in runs if r['started_at']), default=''),
            max((r['finished_at'] for r in runs if r['finished_at']), default=''))
    return cfgs, runs, (dict(plat) if plat else {}), span


def load_index(path):
    con = ro(path)
    rows = [dict(r) for r in con.execute('SELECT * FROM traces')]
    con.close()
    return rows


def load_sim(path):
    con = ro(path)
    sims = [dict(r) for r in con.execute('SELECT * FROM sims')]
    runs = [dict(r) for r in con.execute('SELECT * FROM runs ORDER BY stamp')]
    con.close()
    versions = sorted(set(s['sim_version'] for s in sims))
    return sims, runs, versions


# ── kernel-name matching ─────────────────────────────────────────────────
def kernel_split(trace_kernels, timed):
    """(profiled names in the trace, unprofiled names in the trace)."""
    names = [k['name'] for k in trace_kernels]
    have = set(names)
    timed_in_trace = set()
    for t in timed:
        if t in have:
            timed_in_trace.add(t)
        else:
            base = re.sub(r'_\d+$', '', t)
            if base in have:
                timed_in_trace.add(base)
    prof = [n for n in names if n in timed_in_trace]
    unprof = [n for n in names if n not in timed_in_trace]
    return prof, unprof


# ── tabs ─────────────────────────────────────────────────────────────────
def trace_tab(index_rows):
    """Body, style and script of trace_collect.py's report page."""
    summary = tc.report_summary(REPO, index_rows)
    buckets = tc.report_buckets(index_rows)
    tot = tc.report_totals(index_rows, summary)
    fd, tmp = tempfile.mkstemp(suffix='.html')
    os.close(fd)
    try:
        tc.report_html(tmp, tot, summary, buckets)
        page = open(tmp).read()
    finally:
        os.remove(tmp)
    style = page[page.index('<style>') + 7:page.index('</style>')]
    main = page[page.index('<main>') + 6:page.index('</main>')]
    script = page[page.index('<script>') + 8:page.index('</script>')]
    return style, main, script, tot


def profile_tab(gt, runs, plat, span):
    cfgs = list(gt.values())
    ok = [c for c in cfgs if c['times']]
    failed = [c for c in cfgs if not c['times']]
    benches = {}
    for c in cfgs:
        benches.setdefault((c['tier'], c['suite'], c['benchmark']), []).append(c)
    wall = sum(c['wall'] for c in cfgs)
    reps = statistics.median([len(c['times']) for c in ok]) if ok else 0
    cvs = [c['cv'] for c in ok if c['cv'] is not None]
    trs = []
    for (tier, suite, b), cs in sorted(benches.items()):
        okc = [c for c in cs if c['times']]
        meds = sorted(c['median_ms'] for c in okc)
        bcv = [c['cv'] for c in okc if c['cv'] is not None]
        nk = sorted(set(len(c['kernels']) for c in okc))
        fail = [c for c in cs if not c['times']]
        chip = '<span class="chip good">all</span>' if not fail else \
            ('<span class="chip bad">%d failed</span>' % len(fail))
        trs.append(
            '<tr data-tier="%d"><td class="n">%d</td><td class="sname">%s</td><td><b class="bname">%s</b> %s</td>'
            '<td class="n" data-v="%d">%d / %d</td><td class="n" data-v="%d">%s</td>'
            '<td class="n" data-v="%f">%s</td><td class="n" data-v="%f">%s</td>'
            '<td class="n" data-v="%f">%s</td><td class="n">%s</td><td class="n" data-v="%f">%s</td>'
            '<td class="err">%s</td></tr>' % (
                tier, tier, e(suite), e(b), chip, len(okc), len(okc), len(cs),
                sum(len(c['times']) for c in cs), num(sum(len(c['times']) for c in cs)),
                meds[0] if meds else -1, ms(meds[0]) if meds else '-',
                meds[-1] if meds else -1, ms(meds[-1]) if meds else '-',
                statistics.median(bcv) if bcv else -1, pct(statistics.median(bcv), False) if bcv else '-',
                '–'.join(str(x) for x in (nk[:1] + nk[-1:] if len(nk) > 1 else nk)) or '-',
                sum(c['wall'] for c in cs), secs(sum(c['wall'] for c in cs)),
                e(tc.one_line(fail[0]['error_message'] or '', 120)) if fail else ''))
    head = ('<p class="headline"><b>%s</b> configurations profiled on hardware '
            '(%s attempted, %s failed every repetition)</p>'
            % (num(len(ok)), num(len(cfgs)), num(len(failed))))
    tiles = [
        ('Configurations profiled', num(len(ok)), 'of %s attempted' % num(len(cfgs))),
        ('Timing runs', num(sum(len(c['times']) for c in ok)),
         '%d repetitions per configuration (median)' % reps),
        ('Run-to-run spread', pct(statistics.median(cvs), False) if cvs else '-',
         'median coefficient of variation'),
        ('Profiling time', '%.1f h' % (wall / 3600), '%s to %s' % (span[0][:16], span[1][:16])),
        ('Platform', e(plat.get('gpu_model', '?')),
         'driver %s, %s' % (e(plat.get('driver_version') or '?'),
                           e((plat.get('compiler') or '').split(',')[-1].strip() or '?'))),
    ]
    note = ('Hardware time of a configuration = median of runs.total_time_ms over its repetitions: '
            "the benchmark's own CUDA-event timing of its kernels (sum over the kernels it times). "
            'Host %s, %s, %s cores, %s GB RAM.' % (
                e(plat.get('hostname') or '?'), e(plat.get('cpu_model') or '?'),
                e(str(plat.get('cpu_cores') or '?')), e(str(plat.get('ram_gb') or '?'))))
    table = table_html('p-t', [
        ('Tier', 1), ('Suite', 0), ('Benchmark', 0), ('Profiled / attempted', 1), ('Runs', 1),
        ('Fastest config', 1), ('Slowest config', 1), ('Median spread', 1), ('Kernels timed', None),
        ('Wall time', 1), ('Failure', None)], trs)
    return head + tiles_html(tiles) + '<p class="muted">%s</p>' % note + \
        '<section><h2>Per benchmark</h2>%s%s</section>' % (tier_filter('p-t', len(benches)), table)


def simulation_tab(sims, sim_runs, version, index_ok, gt):
    rows = [s for s in sims if s['sim_version'] == version]
    ok = [s for s in rows if s['status'] == 'ok']
    st = {}
    for s in rows:
        st[s['status']] = st.get(s['status'], 0) + 1
    sim_sec = sum(s['sim_sec'] or 0 for s in rows)
    conv = sum(s['convert_sec'] or 0 for s in rows)
    insts = sum(s['num_insts'] or 0 for s in ok)
    okt = sum(s['sim_sec'] or 0 for s in ok)
    freq = next((s['sim_freq_mhz'] for s in ok if s['sim_freq_mhz']), None)
    runs = [r for r in sim_runs if r['sim_version'] == version]
    t0 = min((r['started_at'] for r in runs if r['started_at']), default='')
    t1 = max((r['finished_at'] or '' for r in runs), default='')
    benches = {}
    for s in rows:
        benches.setdefault((s['tier'], s['benchmark']), []).append(s)
    tried = set(s['trace_id'] for s in rows)
    not_tried = len([1 for tid in index_ok if tid not in tried])
    trs = []
    for (tier, b), rs in sorted(benches.items()):
        o = [s for s in rs if s['status'] == 'ok']
        ranks = tc.scaling_ranks(REPO, tier, b)
        big = max(o, key=lambda s: (ranks.get(s['scaling_value'], -1), s['trace_s0'] or 0)) if o else None
        c = lambda x: sum(1 for s in rs if s['status'] == x)
        bi = sum(s['num_insts'] or 0 for s in o)
        bt = sum(s['sim_sec'] or 0 for s in o)
        ntot = sum(1 for t in index_ok.values() if t == b)
        chip = '<span class="chip good">all</span>' if len(o) == ntot and ntot else \
            ('<span class="chip bad">none</span>' if not o else '')
        params = ''
        if big:
            p = json.loads(big['non_scaling_params_json'] or '{}')
            params = ', '.join('%s=%s' % kv for kv in p.items())
        err = next((s['error_message'] for s in rs if s['status'] == 'error'), '')
        trs.append(
            '<tr data-tier="%d"><td class="n">%d</td><td><b class="bname">%s</b> %s</td>'
            '<td class="n" data-v="%f">%d / %d<div class="bar"><span style="width:%.1f%%"></span></div></td>'
            '<td class="n">%d</td><td class="n">%d</td><td class="n">%d</td>'
            '<td class="n" data-v="%d">%s</td><td class="n" data-v="%f">%s</td>'
            '<td class="n" data-v="%f">%s</td><td class="n" data-v="%f">%s</td><td>%s</td>'
            '<td class="n" data-v="%f">%s</td><td class="err">%s</td></tr>' % (
                tier, tier, e(b), chip, len(o) / ntot if ntot else 0, len(o), ntot,
                100.0 * len(o) / ntot if ntot else 0, c('timeout'), c('skipped_after_timeout'),
                c('error'), bi, num(bi), bt, secs(bt), bi / bt if bt else -1,
                num(bi / bt) if bt else '-', max((s['peak_rss_mb'] or 0) for s in rs),
                ('%.1f GB' % (max((s['peak_rss_mb'] or 0) for s in rs) / 1024)),
                ('<code>%s=%s</code><div class="muted">%s</div>' % (
                    e(big['scaling_param']), e(big['scaling_value']), e(params))) if big else '-',
                (big['sim_sec'] or 0) if big else -1, secs(big['sim_sec']) if big else '-',
                e(tc.one_line(err or '', 120))))
    head = ('<p class="headline"><b>%s</b> of the <b>%s</b> collected traces simulated successfully</p>'
            % (num(len(ok)), num(len(index_ok))))
    tiles = [
        ('Simulated', num(len(ok)), 'of %s collected traces' % num(len(index_ok))),
        ('Not simulated', num(len(index_ok) - len(ok)),
         '%d timeout, %d skipped after a timeout, %d error%s' % (
             st.get('timeout', 0), st.get('skipped_after_timeout', 0), st.get('error', 0),
             (', %d not attempted' % not_tried) if not_tried else '')),
        ('Simulator time', '%.1f h' % (sim_sec / 3600), '%.1f h more expanding .tracez' % (conv / 3600)),
        ('Simulated instructions', num(insts), '%s warp instructions/s per simulation' %
         (num(insts / okt) if okt else '-')),
        ('Peak memory', '%.1f GB' % (max([s['peak_rss_mb'] or 0 for s in rows] or [0]) / 1024),
         'largest single simulation'),
    ]
    note = ('MGPUSim NVIDIA, <code>%s</code> @ <code>%s</code>, device %s at %s MHz. Collected %s to %s. '
            'Traces are expanded from .tracez to .traceg with Accel-Sim traceDsm before each simulation; '
            'the timeout applies to the simulator only.' % (
                e(next((s['mgpusim_branch'] for s in rows if s['mgpusim_branch']), '?')), e(version),
                e(next((s['device'] for s in ok if s['device']), 'H100')),
                ('%.0f' % freq) if freq else '?', e(t0), e(t1)))
    table = table_html('s-t', [
        ('Tier', 1), ('Benchmark', 0), ('Simulated / traced', 1), ('Timeout', None), ('Skipped', None),
        ('Error', None), ('Instructions', 1), ('Simulator time', 1), ('Instr/s', 1), ('Peak memory', 1),
        ('Largest simulated', None), ('Its time', 1), ('Error message', None)], trs)
    return head + tiles_html(tiles) + '<p class="muted">%s</p>' % note + \
        '<section><h2>Per benchmark</h2>%s%s</section>' % (tier_filter('s-t', len(benches)), table)


def report_data(gt, sims, version):
    """Matched configurations grouped suite -> benchmark -> non-scaling combination."""
    rows = [s for s in sims if s['sim_version'] == version and s['status'] == 'ok' and s['sim_cycles']]
    freq = next((s['sim_freq_mhz'] for s in rows if s['sim_freq_mhz']), 1755.0)
    benches = {}
    unmatched = 0
    for s in rows:
        g = gt.get(s['trace_id'])
        if not g or not g['median_ms']:
            unmatched += 1
            continue
        hw = g['median_ms'] * 1e-3 * freq * 1e6
        prof, unprof = kernel_split(json.loads(s['kernels_json'] or '[]'), g['kernels'])
        b = benches.setdefault(s['benchmark'], {
            'suite': g['suite'], 'tier': s['tier'], 'sp': s['scaling_param'], 'points': []})
        b['points'].append({
            'tid': s['trace_id'], 'sv': s['scaling_value'], 'ns': s['non_scaling_params_json'],
            'sim': s['sim_cycles'], 'hw': hw, 'hw_ms': g['median_ms'], 'reps': len(g['times']),
            'err': (s['sim_cycles'] - hw) / hw, 'kp': len(prof), 'ku': len(unprof),
            'unprof': unprof[:6]})
    out = []
    for name, b in benches.items():
        ranks = tc.scaling_ranks(REPO, b['tier'], name)
        pts = b['points']
        numeric = True
        for p in pts:
            try:
                p['x'] = float(p['sv'])
            except ValueError:
                numeric = False
        for p in pts:
            p['rank'] = ranks.get(p['sv'], 1 << 20)
            if not numeric:
                p['x'] = p['rank']
        combos = {}
        for p in pts:
            combos.setdefault(p['ns'], []).append(p)
        clist = []
        for ns, ps in combos.items():
            ps.sort(key=lambda p: (p['rank'], p['x']))
            d = json.loads(ns or '{}')
            label = ', '.join('%s=%s' % kv for kv in sorted(d.items())) or '(no non-scaling parameters)'
            extra = [p for p in ps if p['ku']]
            c = {'label': label, 'points': ps,
                 'note': ({'kp': extra[0]['kp'], 'ku': extra[0]['ku'], 'names': extra[0]['unprof'],
                           'n': len(extra), 'of': len(ps)} if extra else None)}
            for mode in MODES:
                c[mode] = traj_metrics(ps, mode == 'anc')
            clist.append(c)
        clist.sort(key=lambda c: (-len(c['points']), c['label']))
        suite = b['suite']
        if name.startswith(suite + '_'):
            short = name[len(suite) + 1:]
        elif b['tier'] >= 2 and '_' in name:      # e.g. cuda_sdk: cuda_nbody -> nbody
            short = name.split('_', 1)[1]
        else:                                      # tier1 microbenchmarks keep their names
            short = name
        xs = [p['x'] for p in pts]
        bd = {'name': name, 'short': short, 'suite': suite, 'tier': b['tier'], 'sp': b['sp'],
              'numeric': numeric, 'logx': numeric and min(xs) > 0 and max(xs) / min(xs) >= 8,
              'n': len(pts), 'combos': clist}
        for mode in MODES:
            bd[mode] = bench_metrics([c[mode] for c in clist])
        out.append(bd)
    out.sort(key=lambda b: (b['suite'], b['short']))
    suites = {}
    for b in out:
        suites.setdefault(b['suite'], []).append(b)
    slist = []
    for sname, bs in sorted(suites.items()):
        d = {'name': sname, 'nb': len(bs), 'nc': sum(b['n'] for b in bs)}
        for mode in MODES:
            d[mode] = group_metrics([b[mode] for b in bs])
        slist.append(d)
    overall = {mode: group_metrics([b[mode] for b in out]) for mode in MODES}
    return {'freq': freq, 'benches': out, 'suites': slist, 'overall': overall,
            'unmatched': unmatched, 'n': sum(b['n'] for b in out)}


# key, short name, full name; +1 = higher is better, -1 = lower is better, 0 = signed
METRICS = [('mean', 'MRE', 'Mean Relative Error', 0), ('mae', 'MARE', 'Mean Absolute Relative Error', -1),
           ('lsp', 'LSP', 'Log-Space Pearson', 1), ('malge', 'MALGE', 'Mean Absolute Log-Growth Error', -1)]
MODES = ('raw', 'anc')  # raw simulation / first-point anchored (additive offset per trajectory)
COUNT_KEY = {'mean': 'n_rel', 'mae': 'n_rel', 'lsp': 'n_lsp', 'malge': 'n_malge'}
LSP_MIN_POINTS = 3      # Pearson of 2 points is always +-1
MALGE_MIN_POINTS = 2    # one growth step is enough
SUITE_ROWS_SHOWN = 30


def mean_or_none(vals):
    v = [x for x in vals if x is not None]
    return statistics.mean(v) if v else None


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 1e-24 or vb <= 1e-24:          # a constant series: undefined
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb) ** 0.5


def traj_metrics(ps, anchored=False):
    """Metrics of one trajectory (points sorted by scaling value).

    anchored: the simulation is shifted by the constant hw - sim of the first point, so
    the first point matches exactly; it is then left out of MRE / MARE. A shifted value
    that is not positive cannot be shown on a log axis and is dropped.
    """
    import math
    off = (ps[0]['hw'] - ps[0]['sim']) if anchored else 0.0
    sims = [p['sim'] + off for p in ps]
    rel = [(s - p['hw']) / p['hw'] for s, p in zip(sims, ps) if s > 0]
    if anchored:
        rel = [(s - p['hw']) / p['hw'] for s, p in list(zip(sims, ps))[1:] if s > 0]
    valid = [(s, p['hw']) for s, p in zip(sims, ps) if s > 0 and p['hw'] > 0]
    ls = [math.log(s) for s, _ in valid]
    lh = [math.log(h) for _, h in valid]
    lsp = pearson(ls, lh) if len(ls) >= LSP_MIN_POINTS else None
    malge = None
    if len(ls) >= MALGE_MIN_POINTS:     # natural log; growth between neighbouring scaling values
        malge = statistics.mean(abs((ls[i + 1] - ls[i]) - (lh[i + 1] - lh[i])) for i in range(len(ls) - 1))
    return {'rel': rel, 'mean': statistics.mean(rel) if rel else None,
            'mae': statistics.mean(abs(x) for x in rel) if rel else None,
            'lsp': lsp, 'malge': malge, 'n_rel': len(rel),
            'n_lsp': len(ls) if lsp is not None else 0, 'n_malge': len(ls) if malge is not None else 0,
            'dropped': sum(1 for s in sims if s <= 0), 'off': off}


def bench_metrics(trajs):
    """MRE / MARE over the benchmark's configurations; LSP / MALGE averaged over trajectories."""
    rel = [x for t in trajs for x in t['rel']]
    return {'mean': statistics.mean(rel) if rel else None,
            'mae': statistics.mean(abs(x) for x in rel) if rel else None,
            'lsp': mean_or_none(t['lsp'] for t in trajs), 'malge': mean_or_none(t['malge'] for t in trajs),
            'n_rel': len(rel), 'n_lsp': sum(t['n_lsp'] for t in trajs),
            'n_malge': sum(t['n_malge'] for t in trajs), 'dropped': sum(t['dropped'] for t in trajs)}


def group_metrics(ms):
    """Suite or page: average of the benchmarks that have a value; counts summed over those."""
    d = {}
    for k, _, _, _ in METRICS:
        have = [m for m in ms if m[k] is not None]
        d[k] = statistics.mean(m[k] for m in have) if have else None
        d['nb_' + k] = len(have)
        d['nc_' + k] = sum(m[COUNT_KEY[k]] for m in have)
    d['dropped'] = sum(m['dropped'] for m in ms)
    return d


def fmt_metric(k, v):
    if v is None:
        return '–'
    if k == 'mean':
        return pct(v)
    if k == 'mae':
        return pct(v, False)
    return ('−' if v < 0 else '') + '%.4f' % abs(v)


def dual(raw, anc):
    """Both modes in the page; CSS shows the one that is selected."""
    return '<span class="mr">%s</span><span class="ma">%s</span>' % (raw, anc)


def signed(k, v, text):
    if k == 'mean' and v is not None:
        return '<span class="%s">%s</span>' % ('pos' if v >= 0 else 'neg', text)
    return text


def metric_cells(d, plain=False):
    """Table cells for both modes; plain rows (benchmarks) are grey with no sign colours."""
    out = []
    for k, _, _, _ in METRICS:
        parts = []
        for mode in MODES:
            v = d[mode][k]
            parts.append(fmt_metric(k, v) if plain else signed(k, v, fmt_metric(k, v)))
        out.append('<td class="n">%s</td>' % dual(*parts))
    return ''.join(out)


def metric_stats(d, used_note=None):
    parts = []
    for mode in MODES:
        spans = []
        for k, sh, _, _ in METRICS:
            v = d[mode][k]
            cls = 'stat'
            if k == 'mean' and v is not None:
                cls += ' pos' if v >= 0 else ' neg'
            spans.append('<span class="%s">%s %s</span>' % (cls, sh, fmt_metric(k, v)))
        if mode == 'anc' and used_note:
            spans.append('<span class="muted used">%s</span>' % used_note)
        parts.append(''.join(spans))
    return '<div class="metrics">%s</div>' % dual(*parts)


def metric_head(sh, better):
    if better > 0:
        return '%s <span class="better up">(↑)</span>' % sh
    if better < 0:
        return '%s <span class="better down">(↓)</span>' % sh
    return sh


def anc_used_note(n_rel, n_total, dropped):
    s = 'MRE/MARE on %s of %s configurations (first of each trajectory excluded' % (num(n_rel), num(n_total))
    if dropped:
        s += '; %d dropped: shifted value ≤ 0' % dropped
    return s + ')'


def report_tab(data, version, n_sim_ok):
    if not data['benches']:
        return '<p class="headline">No simulated configuration matches a hardware measurement yet.</p>', '[]'
    nb = len(data['benches'])
    head = ('<p class="headline"><b class="sname">%d</b> suites · <b class="bname">%d</b> benchmarks · '
            '<b>%s</b> configurations compared (simulation vs hardware)</p>'
            % (len(data['suites']), nb, num(data['n'])))
    tiles1 = [
        ('Suites', '<span class="sname">%d</span>' % len(data['suites']), ''),
        ('Benchmarks', '<span class="bname">%d</span>' % nb, ''),
        ('Configurations', num(data['n']), 'of %s simulated' % num(n_sim_ok)),
        ('Clock for cycles', '%.0f MHz' % data['freq'], 'simulated H100 core clock'),
    ]
    tiles2 = []
    for k, sh, full, _ in METRICS:
        vals, subs = [], []
        for mode in MODES:
            o = data['overall'][mode]
            vals.append(signed(k, o[k], fmt_metric(k, o[k])))
            subs.append('average of %d benchmarks · %s configurations' % (o['nb_' + k], num(o['nc_' + k])))
        tiles2.append(('%s (%s)' % (full, sh), dual(*vals), dual(*subs)))
    notes = [
        'Each point is one configuration. <span class="key sim">Simulation</span> = MGPUSim simulated '
        'cycles (<code>%s</code>); <span class="key hw">hardware</span> = median measured time × %.0f MHz. '
        'A trajectory is one line on a chart: the configurations of a benchmark that share the same '
        'non-scaling parameters, ordered by the scaling value (switch between them with the menu). Every '
        'metric is computed per benchmark, then averaged over the benchmarks of a suite, and over all '
        'benchmarks for the page; a benchmark without a value for a metric is left out of those averages, '
        'and the configuration counts under the averages count only what went into them. '
        'Chart axes are logarithmic when the values span a factor of 8 or more. '
        '<span class="warn-inline">!</span> marks a benchmark whose trace contains kernels that the '
        'hardware timing does not include; hover it for details.' % (e(version), data['freq']),
        '<b>First-point anchored</b> (switch at the top): within each trajectory the simulation is shifted '
        'by one constant number of cycles, hardware − simulation at the first (smallest) scaling value, so '
        'the first point matches hardware exactly and every later point moves by the same amount. This '
        'removes a fixed per-run overhead difference. Every metric on this page changes; MRE and MARE then '
        'leave out the first point of each trajectory (its error is 0 by construction), and a shifted value '
        'that is not positive is dropped. Charts show the anchored simulation, with the raw simulation as a '
        'faint dashed line. Grey frames turn light blue in this mode.',
        '<b>MRE</b> (mean relative error) = (simulation − hardware) / hardware for each configuration, '
        "averaged over the benchmark's configurations. It is signed, so over- and underestimates cancel.",
        '<b>MARE</b> (mean absolute relative error) = the average of |simulation − hardware| / hardware '
        'over the configurations. It does not cancel; lower is better.',
        '<b>LSP</b> (log-space Pearson) = the Pearson correlation between log(simulated cycles) and '
        'log(hardware cycles) along a trajectory: does the simulation rise and fall with the hardware? '
        'Range −1 to 1, higher is better. Computed for trajectories with at least %d configurations and '
        'left out where the simulated or the hardware cycles do not change, then averaged over the '
        'trajectories of the benchmark.' % LSP_MIN_POINTS,
        '<b>MALGE</b> (mean absolute log-growth error) = (1 / (N − 1)) Σ |Δln S<sub>i</sub> − '
        'Δln H<sub>i</sub>| along a trajectory of N configurations, where Δln S<sub>i</sub> and '
        'Δln H<sub>i</sub> are the changes of the natural log of simulated and hardware cycles from one '
        'scaling value to the next: does each step grow by the right amount? A constant factor does not '
        'count (simulation = 2 × hardware everywhere gives 0). 0.10 is about a 10%% error in each '
        "step's growth factor; lower is better. Computed for trajectories with at least %d "
        'configurations, then averaged over the trajectories of the benchmark.' % MALGE_MIN_POINTS,
    ]
    note = '<div class="notes">%s</div>' % ''.join('<p>%s</p>' % n for n in notes)
    rows = []
    for s in data['suites']:
        rows.append('<tr class="srow"><td><button type="button" class="link s" data-target="suite-%s">'
                    '%s</button></td><td>%d</td><td class="n">%s</td>%s</tr>' % (
                        e(s['name']), e(s['name']), s['nb'], num(s['nc']), metric_cells(s)))
        for b in data['benches']:
            if b['suite'] != s['name']:
                continue
            rows.append('<tr class="brow"><td></td><td><button type="button" class="link b" '
                        'data-target="bench-%d">%s</button></td><td class="n">%s</td>%s</tr>' % (
                            data['benches'].index(b), e(b['short']), num(b['n']),
                            metric_cells(b, plain=True)))
    for i in range(SUITE_ROWS_SHOWN, len(rows)):
        rows[i] = rows[i].replace('<tr class="', '<tr data-extra="1" hidden class="', 1)
    more = ''
    if len(rows) > SUITE_ROWS_SHOWN:
        more = ('<button type="button" class="expand" id="suite-expand" aria-expanded="false" '
                'data-total="%d" data-shown="%d"><span class="label">Show all %d rows</span>'
                '<svg class="chev" viewBox="0 0 16 16" aria-hidden="true"><path d="M4 6l4 4 4-4" '
                'fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
                'stroke-linejoin="round"/></svg></button>' % (len(rows), SUITE_ROWS_SHOWN, len(rows)))
    mhead = ''.join('<th class="r">%s</th>' % metric_head(sh, better) for _, sh, _, better in METRICS)
    stable = ('<section><h2>Suites and benchmarks</h2><div class="tablewrap mid"><table class="plain" '
              'id="suite-table"><thead><tr><th>Suite</th><th>Benchmarks</th>'
              '<th class="r">Configurations</th>%s</tr></thead><tbody>%s</tbody></table>%s</div></section>'
              % (mhead, ''.join(rows), more))
    panels = []
    for s in data['suites']:
        cards = []
        for b in [b for b in data['benches'] if b['suite'] == s['name']]:
            idx = data['benches'].index(b)
            sel = ''
            if len(b['combos']) > 1:
                opts = ''.join('<option value="%d">%s (%d)</option>' % (j, e(c['label']), len(c['points']))
                               for j, c in enumerate(b['combos']))
                sel = ('<label class="combo"><span class="muted">Non-scaling parameters</span>'
                       '<select id="combo-%d" data-b="%d">%s</select></label>' % (idx, idx, opts))
            elif b['combos']:
                sel = '<p class="muted combo-one">%s</p>' % e(b['combos'][0]['label'])
            used = anc_used_note(b['anc']['n_rel'], b['n'], b['anc']['dropped'])
            cards.append(
                '<article class="card bench" id="bench-%d"><header><span class="tag b">benchmark</span>'
                '<span class="warn" id="warn-%d" tabindex="0" hidden aria-label="kernel note">!'
                '<span class="tip" role="tooltip" id="tip-%d"></span></span>'
                '<h3>%s</h3><span class="muted">%d configurations</span></header>%s%s'
                '<div class="chart" id="chart-%d"></div>'
                '<p class="muted combo-err" id="cerr-%d"></p></article>' % (
                    idx, idx, idx, e(b['short']), b['n'], metric_stats(b, used), sel, idx, idx))
        s_used = anc_used_note(sum(b['anc']['n_rel'] for b in data['benches'] if b['suite'] == s['name']),
                               s['nc'], s['anc']['dropped'])
        panels.append(
            '<section class="suite" id="suite-%s"><header class="suite-head"><span class="tag s">suite</span>'
            '<h2>%s</h2><span class="muted">%d benchmarks · %s configurations</span></header>%s'
            '<div class="cards">%s</div></section>' % (
                e(s['name']), e(s['name']), s['nb'], num(s['nc']), metric_stats(s, s_used), ''.join(cards)))
    keys = [m[0] for m in METRICS] + ['n_rel', 'dropped']
    js = json.dumps([{
        'sp': b['sp'], 'numeric': b['numeric'], 'logx': b['logx'],
        'combos': [{'raw': {k: c['raw'][k] for k in keys}, 'anc': {k: c['anc'][k] for k in keys},
                    'note': c['note'],
                    'p': [[p['x'], p['sv'], p['sim'], round(p['hw']), p['err'], p['hw_ms'], p['reps']]
                          for p in c['points']]} for c in b['combos']]} for b in data['benches']],
        separators=(',', ':'))
    body = (head + tiles_html(tiles1) + tiles_html(tiles2, 'metric-tiles') + note + stable +
            ''.join(panels))
    return body, js


# ── html helpers ─────────────────────────────────────────────────────────
def tiles_html(tiles, extra=''):
    return '<div class="tiles %s">%s</div>' % (extra, ''.join(
        '<div class="tile"><span class="k">%s</span><span class="v">%s</span><span class="s">%s</span></div>'
        % (k, v, s) for k, v, s in tiles))


def tier_filter(tid, n):
    return ('<div class="tierbar" role="group" aria-label="Filter by tier" data-table="%s">'
            '<span class="muted">Show</span>'
            '<button type="button" data-tier="" aria-pressed="true">All %d</button>'
            '<button type="button" data-tier="1" aria-pressed="false">Tier 1</button>'
            '<button type="button" data-tier="2" aria-pressed="false">Tier 2</button>'
            '<span class="muted">click a column header to sort</span></div>' % (tid, n))


def table_html(tid, cols, trs):
    th = []
    for i, (label, numeric) in enumerate(cols):
        if numeric is None:
            th.append('<th>%s</th>' % label)
        else:
            th.append('<th%s><button type="button" data-col="%d"%s aria-sort="none">%s</button></th>' % (
                ' class="r"' if numeric else '', i, ' data-num="1"' if numeric else '', label))
    return ('<div class="tablewrap"><table class="sortable" id="%s"><thead><tr>%s</tr></thead>'
            '<tbody>%s</tbody></table></div>' % (tid, ''.join(th), '\n'.join(trs)))


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Simulation vs Hardware Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
{{trace_style}}
/* report additions: tab bar on top, one panel per tab; red = simulation, blue = hardware */
:root { --anchor: #8ec3ee; --anchor-strong: #2f7fc4; --sim: #c4382b; --hw: #1f5fae; --suite: #6c3fc4; --bench: #1e7a4c; --warn: #e8930c; --warn-on: #3d2400; --warn-bg: #fff3dc; --warn-ink: #6b3f00; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --anchor: #3f77a8; --anchor-strong: #4f95d4; --sim: #f0806f; --hw: #6ea4ea; --suite: #b79cf2; --bench: #6fcf97; --warn: #f0a530; --warn-on: #2a1900; --warn-bg: #3a2a10; --warn-ink: #ffd89a; }
}
:root[data-theme="dark"] { --anchor: #3f77a8; --anchor-strong: #4f95d4; --sim: #f0806f; --hw: #6ea4ea; --suite: #b79cf2; --bench: #6fcf97; --warn: #f0a530; --warn-on: #2a1900; --warn-bg: #3a2a10; --warn-ink: #ffd89a; }
/* suite = purple, benchmark = green, wherever they appear */
.tag { display: inline-block; font-size: 10.5px; font-weight: 600; letter-spacing: .02em; line-height: 1.5;
       padding: 0 6px; border: 1px solid currentColor; border-radius: 4px; align-self: center; }
.tag.s, .sname { color: var(--suite); }
.tag.b, .bname { color: var(--bench); }
.suite-head h2 { color: var(--suite); }
.bench h3 { color: var(--bench); }
#panel-trace #t td b { color: var(--bench); }
.tabs { position: sticky; top: 0; z-index: 5; background: var(--bg); border-bottom: 1px solid var(--line); }
.tabs nav { max-width: 1240px; margin: 0 auto; padding: 10px 20px 0; display: flex; gap: 4px; flex-wrap: wrap; align-items: end; }
.tabs a { color: var(--muted); text-decoration: none; padding: 8px 14px 10px; border-bottom: 2px solid transparent; font-weight: 500; }
.tabs a[aria-current="page"] { color: var(--ink); border-bottom-color: var(--accent); }
.tabs a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.panel[hidden] { display: none; }
.headline { font-size: 18px; margin: 0; max-width: none; }
.tierbar { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.tierbar button, button.jump { font: inherit; font-size: 13px; padding: 4px 12px; border-radius: 999px; border: 1px solid var(--line); background: var(--surface); color: var(--ink); cursor: pointer; }
.tierbar button[aria-pressed="true"] { border-color: var(--accent); color: var(--accent); }
button.jump { border-radius: 6px; padding: 1px 8px; color: var(--suite); border-color: currentColor; font-weight: 500; }
table.sortable, table.plain { border-collapse: collapse; width: 100%; font-size: 13.5px; }
table.sortable { min-width: 1080px; }
table.plain { min-width: 0; }
table.plain th, table.plain td { padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; }
table.plain th { font-weight: 600; font-size: 12.5px; color: var(--muted); }
th.r, td.n, table.plain th.r, table.plain td.n { text-align: right; }
.tablewrap.narrow { max-width: 720px; }
td.err { color: var(--muted); font-size: 12px; max-width: 280px; }
.stat { font-variant-numeric: tabular-nums; font-size: 13px; padding: 1px 8px; border-radius: 999px; border: 1px solid var(--line); }
.pos { color: var(--sim); }
.neg { color: var(--hw); }
.suite { display: grid; gap: 14px; background: var(--surface); border: 1px solid var(--line); border-left: 4px solid var(--suite); border-radius: 10px; padding: 18px; }
.suite-head, .bench header { display: flex; gap: 10px; flex-wrap: wrap; align-items: baseline; }
.suite-head h2 { font-size: 20px; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 460px), 1fr)); gap: 14px; }
.bench { background: var(--bg); gap: 8px; border-left: 3px solid var(--bench); }
.bench h3 { margin: 0; font-size: 16px; font-weight: 600; }
.combo { display: grid; gap: 2px; }
.combo select { font: inherit; font-size: 13px; padding: 4px 6px; border-radius: 6px; border: 1px solid var(--line); background: var(--surface); color: var(--ink); max-width: 100%; }
.combo select:focus-visible { outline: 2px solid var(--accent); }
.combo-one, .combo-err { margin: 0; }
#panel-report .tiles { grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); }
.warn, .warn-inline { display: inline-flex; align-items: center; justify-content: center; width: 17px; height: 17px;
  border-radius: 4px; border: 1px solid currentColor; color: var(--warn); font-weight: 700; font-size: 11.5px; line-height: 1; }
.warn { position: relative; cursor: help; align-self: center; }
.warn[hidden] { display: none; }
.warn:focus-visible { outline: 2px solid var(--warn); outline-offset: 2px; }
.warn .tip { display: none; position: absolute; left: calc(100% + 8px); top: 50%; transform: translateY(-50%);
  z-index: 10; width: max-content; max-width: 340px; padding: 8px 10px; border-radius: 6px; font-weight: 400;
  font-size: 12.5px; line-height: 1.45; text-align: left; white-space: normal;
  background: var(--warn-bg); color: var(--warn-ink); border: 1px solid var(--warn); box-shadow: 0 4px 14px rgba(0,0,0,.12); }
.warn:hover .tip, .warn:focus .tip { display: block; }
.warn-inline { vertical-align: -3px; margin: 0 2px; }
#suite-table tr.srow td { font-weight: 500; border-top: 1px solid var(--line); }
#suite-table tr.srow td:first-child { white-space: nowrap; }
#suite-table tr.brow td { font-size: 13px; padding-top: 5px; padding-bottom: 5px; color: var(--muted); }
button.link { all: unset; cursor: pointer; }
button.link.b { color: var(--bench); }
button.link.s { color: var(--suite); font-weight: 600; }
button.link:hover { text-decoration: underline; }
button.link:focus-visible { outline: 2px solid var(--bench); outline-offset: 2px; }
.better { font-weight: 600; }
.better.up { color: var(--sim); }
.better.down { color: var(--hw); }
.tablewrap.mid { width: fit-content; max-width: 100%; }
#suite-table { width: auto; }
#suite-table th, #suite-table td { white-space: nowrap; }
#suite-table th + th, #suite-table td + td { padding-left: 28px; }
.notes { display: grid; gap: 6px; }
.notes p { margin: 0; max-width: 118ch; color: var(--muted); font-size: 13px; }
.notes b { color: var(--ink); font-weight: 600; }
.metrics { display: flex; gap: 8px; flex-wrap: wrap; }
.tiles.metric-tiles { margin-top: -14px; }
/* raw / first-point anchored: both versions are in the page, one is shown */
body:not(.anchored) .ma, body.anchored .mr { display: none !important; }
.metrics > .mr, .metrics > .ma { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.metrics .used { font-size: 12px; }
.report-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px 28px; flex-wrap: wrap; }
.report-head h1 { margin-bottom: 4px; }
.modebox { display: grid; gap: 6px; justify-items: end; max-width: 380px; }
.modesw { display: inline-flex; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: var(--surface); }
.modesw button { font: inherit; font-size: 13px; font-weight: 500; padding: 6px 14px; border: 0; background: transparent; color: var(--muted); cursor: pointer; }
.modesw button + button { border-left: 1px solid var(--line); }
.modesw button[aria-pressed="true"] { background: var(--ink); color: var(--surface); }
body.anchored .modesw button[aria-pressed="true"] { background: var(--anchor-strong); color: #fff; }
.modesw button:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.mode-desc { margin: 0; font-size: 12.5px; color: var(--muted); text-align: right; }
body.anchored .modesw { border-color: var(--anchor); }
body.anchored #panel-report .tile, body.anchored #panel-report .tablewrap { border-color: var(--anchor); }
body.anchored #panel-report .suite, body.anchored #panel-report .bench {
  border-top-color: var(--anchor); border-right-color: var(--anchor); border-bottom-color: var(--anchor); }
.chart svg .lraw { stroke: var(--sim); fill: none; stroke-width: 1.5; stroke-dasharray: 4 3; opacity: .4; }
.legend2 i.dash { background: repeating-linear-gradient(90deg, var(--sim) 0 4px, transparent 4px 7px) !important; opacity: .55; }
.expand { display: flex; align-items: center; justify-content: center; gap: 6px; width: 100%; padding: 9px;
  font: inherit; font-size: 13px; color: var(--muted); background: var(--surface); border: 0;
  border-top: 1px solid var(--line); cursor: pointer; }
.expand:hover { color: var(--ink); }
.expand:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.expand .chev { width: 14px; height: 14px; transition: transform .2s ease; }
.expand[aria-expanded="true"] .chev { transform: rotate(180deg); }
@media (prefers-reduced-motion: reduce) { .expand .chev { transition: none; } }
.chart svg .grid { stroke: var(--line); stroke-width: 1; }
.chart svg .ax { fill: var(--muted); font: 11px var(--font-body); }
.chart svg .lsim { stroke: var(--sim); fill: none; stroke-width: 2; }
.chart svg .lhw { stroke: var(--hw); fill: none; stroke-width: 2; }
.chart svg .psim { fill: var(--sim); }
.chart svg .phw { fill: var(--hw); }
.key { font-weight: 600; }
.key.sim { color: var(--sim); }
.key.hw { color: var(--hw); }
.legend2 { display: flex; gap: 14px; font-size: 12px; color: var(--muted); }
.legend2 i { display: inline-block; width: 14px; height: 3px; vertical-align: middle; margin-right: 5px; border-radius: 2px; }
</style>
</head>
<body class="anchored">
<div class="tabs"><nav aria-label="Report sections">
  <a href="#report" data-tab="report">Report</a>
  <a href="#trace" data-tab="trace">Trace</a>
  <a href="#profile" data-tab="profile">Profile</a>
  <a href="#simulation" data-tab="simulation">Simulation</a>
</nav></div>
<main class="panel" id="panel-report">
  <header class="report-head">
    <div><h1>MGPUSim NVIDIA vs H100 hardware</h1>
    <p class="muted">Generated {{generated}} from {{sources}}</p></div>
    <div class="modebox">
      <div class="modesw" role="group" aria-label="Simulation values">
        <button type="button" id="mode-raw" data-mode="raw" aria-pressed="false">Raw</button>
        <button type="button" id="mode-anc" data-mode="anc" aria-pressed="true">First-point anchored</button>
      </div>
      <p class="mode-desc"><span class="mr">Simulated cycles as MGPUSim reports them.</span><span class="ma">Each
      trajectory's simulation is shifted by one constant so its first configuration matches hardware; MRE and
      MARE leave that first point out.</span></p>
    </div>
  </header>
  {{report}}
</main>
<main class="panel" id="panel-trace" hidden>
<p class="headline"><b>{{n_traced}}</b> configurations collected as traces</p>
{{trace_main}}
</main>
<main class="panel" id="panel-profile" hidden>
<header><h1>Hardware ground truth</h1></header>
{{profile}}
</main>
<main class="panel" id="panel-simulation" hidden>
<header><h1>Simulation runs</h1></header>
{{simulation}}
</main>
<script>
{{trace_script}}
</script>
<script>
(function () {
  // tabs on the URL hash
  var tabs = ['report', 'trace', 'profile', 'simulation'];
  function show(name) {
    var t = typeof name === 'string' ? name : (location.hash || '#report').slice(1);
    if (tabs.indexOf(t) < 0) t = 'report';
    tabs.forEach(function (n) {
      document.getElementById('panel-' + n).hidden = n !== t;
      document.querySelector('.tabs a[data-tab="' + n + '"]').setAttribute('aria-current', n === t ? 'page' : 'false');
    });
  }
  // switch on click too: the hash may not change where the page is shown (e.g. a file preview)
  document.querySelectorAll('.tabs a').forEach(function (a) {
    a.addEventListener('click', function (ev) {
      ev.preventDefault();
      show(a.dataset.tab);
      try { history.replaceState(null, '', '#' + a.dataset.tab); } catch (err) {}
      window.scrollTo(0, 0);
    });
  });
  window.addEventListener('hashchange', show); show();

  // sortable tables and tier filters (profile / simulation tabs)
  document.querySelectorAll('table.sortable').forEach(function (t) {
    var tbody = t.tBodies[0], heads = t.querySelectorAll('th button');
    heads.forEach(function (b) {
      b.addEventListener('click', function () {
        var col = +b.dataset.col, isNum = !!b.dataset.num;
        var dir = b.getAttribute('aria-sort') === 'ascending' ? 'descending' : 'ascending';
        heads.forEach(function (h) { h.setAttribute('aria-sort', 'none'); });
        b.setAttribute('aria-sort', dir);
        var rows = Array.prototype.slice.call(tbody.rows);
        rows.sort(function (x, y) {
          var a = x.cells[col], c = y.cells[col], r;
          if (isNum) r = parseFloat(a.dataset.v !== undefined ? a.dataset.v : a.textContent) -
                         parseFloat(c.dataset.v !== undefined ? c.dataset.v : c.textContent);
          else r = a.textContent.localeCompare(c.textContent);
          return dir === 'ascending' ? r : -r;
        });
        rows.forEach(function (r) { tbody.appendChild(r); });
      });
    });
  });
  document.querySelectorAll('.tierbar').forEach(function (bar) {
    var t = document.getElementById(bar.dataset.table), bs = bar.querySelectorAll('button');
    bs.forEach(function (b) {
      b.addEventListener('click', function () {
        bs.forEach(function (o) { o.setAttribute('aria-pressed', o === b ? 'true' : 'false'); });
        Array.prototype.forEach.call(t.tBodies[0].rows, function (r) {
          r.hidden = !!b.dataset.tier && r.dataset.tier !== b.dataset.tier;
        });
      });
    });
  });
  var ex = document.getElementById('suite-expand');
  if (ex) ex.addEventListener('click', function () {
    var open = ex.getAttribute('aria-expanded') !== 'true';
    ex.setAttribute('aria-expanded', open ? 'true' : 'false');
    document.querySelectorAll('#suite-table tr[data-extra]').forEach(function (r) { r.hidden = !open; });
    ex.querySelector('.label').textContent = open ? 'Show first ' + ex.dataset.shown + ' rows'
                                                  : 'Show all ' + ex.dataset.total + ' rows';
  });
  document.querySelectorAll('button.jump, button.link').forEach(function (b) {
    b.addEventListener('click', function () {
      document.getElementById(b.dataset.target).scrollIntoView({behavior: 'smooth', block: 'start'});
    });
  });

  // charts
  var B = {{data}};
  var NS = 'http://www.w3.org/2000/svg';
  function el(n, a, txt) {
    var x = document.createElementNS(NS, n);
    for (var k in a) x.setAttribute(k, a[k]);
    if (txt !== undefined) x.textContent = txt;
    return x;
  }
  function fmtC(v) {
    if (v >= 1e9) return +(v / 1e9).toPrecision(3) + 'G';
    if (v >= 1e6) return +(v / 1e6).toPrecision(3) + 'M';
    if (v >= 1e3) return +(v / 1e3).toPrecision(3) + 'k';
    return String(Math.round(v));
  }
  function fmtP(x) {
    var v = x * 100, a = Math.abs(v);
    return (v >= 0 ? '+' : '−') + a.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2}) + '%';
  }
  var MODE = 'anc';
  function draw(i) {
    var b = B[i], sel = document.getElementById('combo-' + i);
    var c = b.combos[sel ? +sel.value : 0], P = c.p, anc = MODE === 'anc', M = c[MODE];
    var off = anc ? P[0][3] - P[0][2] : 0;
    var S = P.map(function (p) { return p[2] + off; });       // simulation shown (anchored or raw)
    var W = 560, H = 250, L = 56, R = 14, T = 12, Bm = 40;
    var xs = P.map(function (p) { return p[0]; });
    var ys = []; P.forEach(function (p, k) { ys.push(p[3]); if (S[k] > 0) ys.push(S[k]); if (anc) ys.push(p[2]); });
    var ylo = Math.min.apply(null, ys), yhi = Math.max.apply(null, ys);
    var lgy = ylo > 0;
    var y0 = lgy ? Math.floor(Math.log10(ylo)) : 0, y1 = lgy ? Math.ceil(Math.log10(yhi)) : yhi;
    if (lgy && y1 === y0) y1 = y0 + 1;
    var fy = function (v) {
      var t = lgy ? (Math.log10(v) - y0) / (y1 - y0) : v / (y1 || 1);
      return T + (H - T - Bm) * (1 - t);
    };
    var xlo = Math.min.apply(null, xs), xhi = Math.max.apply(null, xs), lgx = b.logx;
    var fx = function (v) {
      if (xhi === xlo) return (L + W - R) / 2;
      var t = lgx ? (Math.log(v) - Math.log(xlo)) / (Math.log(xhi) - Math.log(xlo)) : (v - xlo) / (xhi - xlo);
      return L + 10 + (W - R - L - 20) * t;
    };
    var svg = el('svg', {viewBox: '0 0 ' + W + ' ' + H, role: 'img',
                         'aria-label': 'simulated and hardware cycles by ' + b.sp});
    if (lgy) {
      // decades, plus 2x and 5x when the range spans at most two decades
      var mults = (y1 - y0) <= 2 ? [1, 2, 5] : [1];
      for (var d = y0; d <= y1; d++) {
        mults.forEach(function (m) {
          var v = m * Math.pow(10, d);
          if (v > Math.pow(10, y1)) return;
          var yy = fy(v);
          svg.appendChild(el('line', {x1: L, x2: W - R, y1: yy, y2: yy, 'class': 'grid'}));
          svg.appendChild(el('text', {x: L - 6, y: yy + 4, 'class': 'ax', 'text-anchor': 'end'}, fmtC(v)));
        });
      }
    }
    // x labels at data points, dropping any that would collide (the last always stays)
    var keep = [], lastX = -1e9;
    P.forEach(function (p, k) {
      var x = fx(p[0]), w = 7 * String(p[1]).length + 10;
      if (x - lastX >= w) { keep.push(k); lastX = x; }
    });
    var lastK = P.length - 1;
    if (keep[keep.length - 1] !== lastK) {
      while (keep.length && fx(P[lastK][0]) - fx(P[keep[keep.length - 1]][0]) < 7 * String(P[lastK][1]).length + 10) keep.pop();
      keep.push(lastK);
    }
    keep.forEach(function (k) {
      svg.appendChild(el('text', {x: fx(P[k][0]), y: H - Bm + 16, 'class': 'ax', 'text-anchor': 'middle'}, P[k][1]));
    });
    svg.appendChild(el('text', {x: (L + W - R) / 2, y: H - 6, 'class': 'ax', 'text-anchor': 'middle'}, b.sp));
    svg.appendChild(el('text', {x: 12, y: (T + H - Bm) / 2, 'class': 'ax', 'text-anchor': 'middle',
                                transform: 'rotate(-90 12 ' + (T + H - Bm) / 2 + ')'}, 'cycles'));
    function line(vals, cls) {          // breaks where a value cannot be drawn (<= 0)
      var d = '', pen = false;
      vals.forEach(function (v, k) {
        if (!(v > 0)) { pen = false; return; }
        d += (pen ? 'L' : 'M') + fx(P[k][0]).toFixed(1) + ' ' + fy(v).toFixed(1) + ' ';
        pen = true;
      });
      if (d) svg.appendChild(el('path', {d: d, 'class': cls}));
    }
    if (anc) line(P.map(function (p) { return p[2]; }), 'lraw');
    line(P.map(function (p) { return p[3]; }), 'lhw'); line(S, 'lsim');
    P.forEach(function (p, k) {
      var tip = b.sp + '=' + p[1] + '\\nsimulation ' + (S[k] > 0 ? fmtC(S[k]) : '≤ 0') + ' cycles' +
        (anc ? ' (raw ' + fmtC(p[2]) + ', shift ' + (off >= 0 ? '+' : '−') + fmtC(Math.abs(off)) + ')' : '') +
        '\\nhardware ' + fmtC(p[3]) + ' cycles (' + p[5].toPrecision(4) + ' ms, median of ' + p[6] + ')' +
        (S[k] > 0 ? '\\nerror ' + fmtP((S[k] - p[3]) / p[3]) : '') + (anc && k === 0 ? ' (anchor)' : '');
      [[p[3], 'phw'], [S[k], 'psim']].forEach(function (q) {
        if (!(q[0] > 0)) return;
        var g = el('circle', {cx: fx(p[0]), cy: fy(q[0]), r: 3.5, 'class': q[1]});
        g.appendChild(el('title', {}, tip));
        svg.appendChild(g);
      });
    });
    var box = document.getElementById('chart-' + i);
    box.innerHTML = '';
    var lg = document.createElement('div');
    lg.className = 'legend2';
    lg.innerHTML = '<span><i style="background:var(--sim)"></i>' + (anc ? 'simulation (anchored)' : 'simulation') +
      '</span><span><i style="background:var(--hw)"></i>hardware</span>' +
      (anc ? '<span><i class="dash"></i>raw simulation</span>' : '');
    box.appendChild(lg); box.appendChild(svg);
    var fr = function (v) { return v === null ? '–' : (v < 0 ? '−' : '') + Math.abs(v).toFixed(4); };
    var fp = function (v, sg) { return v === null ? '–' : (sg ? fmtP(v) : fmtP(v).slice(1)); };
    document.getElementById('cerr-' + i).textContent = 'This combination: MRE ' + fp(M.mean, 1) +
      ', MARE ' + fp(M.mae, 0) + ', LSP ' + fr(M.lsp) + ', MALGE ' + fr(M.malge) + ', ' + P.length +
      ' configuration' + (P.length > 1 ? 's' : '') +
      (anc ? ' (MRE/MARE on ' + M.n_rel + (M.dropped ? '; ' + M.dropped + ' dropped' : '') + ')' : '');
    var warn = document.getElementById('warn-' + i);
    if (c.note) {
      warn.hidden = false;
      document.getElementById('tip-' + i).textContent = 'This benchmark trace contains ' + c.note.kp +
        ' (profiled) + ' + c.note.ku + ' (unprofiled) kernels, while the truth contains only ' + c.note.kp +
        ' profiled kernels' + (c.note.names.length ? ' (unprofiled: ' + c.note.names.join(', ') + ')' : '') +
        (c.note.n < c.note.of ? '; ' + c.note.n + ' of ' + c.note.of + ' configurations' : '') + '.';
    } else { warn.hidden = true; }
  }
  function setMode(m) {
    MODE = m === 'anc' ? 'anc' : 'raw';
    document.body.classList.toggle('anchored', MODE === 'anc');
    document.querySelectorAll('.modesw button').forEach(function (b) {
      b.setAttribute('aria-pressed', b.dataset.mode === MODE ? 'true' : 'false');
    });
    B.forEach(function (b, i) { draw(i); });
  }
  document.querySelectorAll('.modesw button').forEach(function (b) {
    b.addEventListener('click', function () { setMode(b.dataset.mode); });
  });
  B.forEach(function (b, i) {
    var s = document.getElementById('combo-' + i);
    if (s) s.addEventListener('change', function () { draw(i); });
  });
  setMode('anc');
})();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split('\n', 2)[2])
    ap.add_argument('--gt-db', default=DEFAULT_GT, help='ground truth (default: %(default)s)')
    ap.add_argument('--trace-db', default=DEFAULT_INDEX, help='trace index.db (default: %(default)s)')
    ap.add_argument('--sim-db', default=None,
                    help='simulation snapshot (default: newest scripts/logs_simulation/<stamp>.db)')
    ap.add_argument('--sim-version', default=None,
                    help='which sim_version in the db (default: the one with most ok rows)')
    ap.add_argument('--out', default=None, help='output file (default: scripts/logs_report/report_<stamp>.html)')
    a = ap.parse_args()
    a.sim_db = a.sim_db or newest_snapshot()
    for name, p in (('ground truth', a.gt_db), ('trace index', a.trace_db), ('simulation', a.sim_db)):
        if not p or not os.path.isfile(p):
            sys.exit('%s database not found: %s' % (name, p))
    gt, gt_runs, plat, span = load_gt(a.gt_db)
    index_rows = load_index(a.trace_db)
    sims, sim_runs, versions = load_sim(a.sim_db)
    if not versions:
        sys.exit('no simulation rows in %s' % a.sim_db)
    version = a.sim_version or max(versions, key=lambda v: sum(
        1 for s in sims if s['sim_version'] == v and s['status'] == 'ok'))
    index_ok = {r['trace_id']: r['benchmark'] for r in index_rows if r['status'] == 'ok'}
    t_style, t_main, t_script, t_tot = trace_tab(index_rows)
    data = report_data(gt, sims, version)
    n_sim_ok = sum(1 for s in sims if s['sim_version'] == version and s['status'] == 'ok')
    rep_html, rep_js = report_tab(data, version, n_sim_ok)
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    vals = {
        'trace_style': t_style, 'trace_main': t_main, 'trace_script': t_script,
        'n_traced': num(t_tot['ok']), 'report': rep_html, 'data': rep_js,
        'profile': profile_tab(gt, gt_runs, plat, span),
        'simulation': simulation_tab(sims, sim_runs, version, index_ok, gt),
        'generated': dt.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'sources': ', '.join('<code>%s</code>' % e(os.path.basename(p))
                             for p in (a.gt_db, a.trace_db, a.sim_db)),
    }
    page = PAGE
    for k, v in vals.items():
        page = page.replace('{{%s}}' % k, v)
    out = a.out or os.path.join(OUT_DIR, 'report_%s.html' % stamp)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w') as f:
        f.write(page)
    print('report: %s' % out)
    for mode, label in (('raw', 'raw'), ('anc', 'anchored')):
        o = data['overall'][mode]
        print('  %-8s %s' % (label, ', '.join('%s %s' % (m[1], fmt_metric(m[0], o[m[0]])) for m in METRICS)))
    print('  %d suites, %d benchmarks, %d configurations compared; sim_version %s (%s)' % (
        len(data['suites']), len(data['benches']), data['n'], version, os.path.basename(a.sim_db)))
    if data['unmatched']:
        print('  note: %d simulated configurations have no hardware measurement' % data['unmatched'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
