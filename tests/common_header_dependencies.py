#!/usr/bin/env python3
"""Check actual Make rebuild decisions without requiring either GPU compiler."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import os


def check(repo):
    headers = [Path('common_hip') / n for n in
               ('bench_timing.h', 'bench_warmup.h', 'hip_cuda_compat.h')]
    makefiles = sorted([*repo.glob('tier1/*/hip/Makefile'), *repo.glob('tier2/*/hip/Makefile')])
    if not makefiles:
        raise RuntimeError('No Tier 1/2 Makefiles found')
    targets = []
    for makefile in makefiles:
        text = makefile.read_text()
        match = re.search(r'^TARGET\s*:=\s*(\S+)\s*$', text, re.M)
        if not match:
            raise RuntimeError('Missing production target: ' + str(makefile))
        directory = makefile.parent.relative_to(repo)
        targets.append((directory, match[1]))
        if re.search(r'^bench_output_test:', text, re.M):
            targets.append((directory, 'bench_output_test'))
    results = []
    with tempfile.TemporaryDirectory(prefix='gpubench-header-deps-') as tmp:
        tree = Path(tmp)
        # Copy only inputs needed to evaluate the real Makefiles. Artifacts
        # below are timestamp fixtures; no recipe or compiler is executed.
        for directory in ('tier1', 'tier2', 'tests', 'common_hip'):
            for source in (repo / directory).rglob('*'):
                if source.is_file() and (source.name == 'Makefile' or
                                        source.suffix in ('.hip', '.h', '.cpp')):
                    dest = tree / source.relative_to(repo)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, dest)
                    os.utime(dest, (1600000000, 1600000000))
        for directory, target in targets:
            artifact = tree / directory / target
            artifact.write_text('Timestamp fixture, not a compiled executable.\n')
            os.utime(artifact, (1600000010, 1600000010))

        def query(phase, changed, expected):
            for directory, target in targets:
                for platform in ('cuda', 'rocm'):
                    command = ['make', '-q', '-C', str(tree / directory),
                               'PLATFORM=' + platform, target]
                    proc = subprocess.run(command, capture_output=True, text=True)
                    if proc.returncode not in (0, 1):
                        raise RuntimeError('Make query failed: ' + ' '.join(command) +
                                           '\n' + proc.stdout + proc.stderr)
                    results.append(dict(phase=phase, header=str(changed or ''),
                                        directory=str(directory), target=target,
                                        platform=platform, expected=expected,
                                        actual=proc.returncode, passed=proc.returncode == expected))
        query('unchanged', None, 0)
        for header in headers:
            if not (tree / header).is_file():
                raise RuntimeError('Missing header: ' + str(header))
            os.utime(tree / header, (1600000020, 1600000020))
            query('header changed', header, 1)
            os.utime(tree / header, (1600000000, 1600000000))
        query('restored', None, 0)
    return dict(benchmarks=len(makefiles), targets=len(targets), results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = check(args.repo.resolve())
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    failed = [r for r in report['results'] if not r['passed']]
    for case in failed[:10]:
        print('FAIL:', case)
    if len(failed) > 10:
        print('Additional failed decisions:', len(failed)-10)
    print('{}: {} benchmarks, {} targets, {} make queries, {} failed'.format(
        'FAIL' if failed else 'PASS', report['benchmarks'], report['targets'],
        len(report['results']), len(failed)))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
