#!/usr/bin/env python3
"""Check committed H100 source/configuration identity; does not qualify a GPU."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re


PROFILES = {
    'cuda_nbody': (('bodies', 'timesteps'), 'full_cpu_v1'),
    'altis_particlefilter': (('size',), 'full_final_aux_cpu_fp64_v1'),
}
COMMON = tuple('common/' + name for name in
               ('bench_timing.h', 'bench_warmup.h', 'hip_cuda_compat.h'))
SOURCE_SUFFIXES = {'.hip', '.h', '.hpp', '.cu', '.cuh', '.c', '.cc', '.cpp', '.inc'}


def source_paths(repo, benchmark):
    """Include new benchmark-local helpers rather than trusting a profile's list."""
    folder = repo / 'tier2' / benchmark / 'hip'
    paths = {f'tier2/{benchmark}/hip/{name}' for name in
             (benchmark + '.hip', 'Makefile', 'params.json')}
    paths.update(str(path.relative_to(repo)) for path in folder.rglob('*')
                 if path.is_file() and path.suffix in SOURCE_SUFFIXES)
    if (folder / 'UPSTREAM_LICENSE').is_file():
        paths.add(f'tier2/{benchmark}/hip/UPSTREAM_LICENSE')
    return paths.union(COMMON)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_profile(repo, benchmark, profile=None):
    repo = Path(repo)
    axes, verification = PROFILES[benchmark]
    folder = repo / 'tier2' / benchmark / 'hip'
    if profile is None:
        profile = json.loads((folder / 'h100_profile.json').read_text())
    require(type(profile) is dict, f'{benchmark}: expected a profile object')
    require(profile.get('complete') is True, f'{benchmark}: incomplete qualification')
    for key, expected in [('benchmark', benchmark), ('profile', 'h100_full_cpu_v1'),
                          ('verification', verification), ('process_timeout_seconds', 60),
                          ('vector_add_warmup', True)]:
        actual = profile.get(key)
        require(type(actual) is type(expected) and actual == expected,
                f'{benchmark}: unexpected {key}')

    hashes = profile.get('source_sha256')
    require(type(hashes) is dict, f'{benchmark}: missing source hashes')
    paths = source_paths(repo, benchmark)
    require(set(hashes) == paths,
            f'{benchmark}: source coverage differs; missing={sorted(paths - set(hashes))}, '
            f'unexpected={sorted(set(hashes) - paths)}. Requalify on H100.')
    for relative in sorted(paths):
        expected = hashes[relative]
        require(type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected),
                f'{benchmark}: invalid SHA256 for {relative}')
        actual = hashlib.sha256((repo / relative).read_bytes()).hexdigest()
        require(actual == expected,
                f'{benchmark}: stale source hash for {relative}. Requalify on H100; '
                'do not refresh hashes without qualification.')

    manifest = json.loads((folder / 'params.json').read_text())
    require(manifest.get('benchmark') == benchmark,
            f'{benchmark}: unexpected manifest identity')
    scaling = manifest['scaling_param']
    controls = manifest['non_scaling_params']
    require(scaling['name'] == axes[0], f'{benchmark}: changed scaling axis')
    require(len({p['name'] for p in controls}) == len(controls),
            f'{benchmark}: duplicate manifest control')
    require({p['name'] for p in controls} == set(axes[1:]) | {'vector_add_warmup'},
            f'{benchmark}: changed workload axes; update qualification coverage')
    values = [scaling['values']] + [next(p['values'] for p in controls if p['name'] == name)
                                   for name in axes[1:]]
    for name, choices in zip(axes, values):
        require(type(choices) is list and choices and
                all(type(x) is int and x > 0 for x in choices) and
                len(set(choices)) == len(choices), f'{benchmark}: invalid {name} candidates')
    candidates = set(itertools.product(*values))
    rows = profile.get('configurations')
    require(type(rows) is list, f'{benchmark}: missing configurations')
    seen = set()
    qualified = 0
    for row in rows:
        require(type(row) is dict and all(type(row.get(name)) is int for name in axes),
                f'{benchmark}: invalid configuration identity')
        key = tuple(row[name] for name in axes)
        require(key in candidates and key not in seen,
                f'{benchmark}: unknown or duplicate configuration {key}')
        seen.add(key)
        require(row.get('status') in ('qualified', 'unsupported'),
                f'{benchmark}: unknown classification for {key}')
        qualified += row['status'] == 'qualified'
    require(seen == candidates, f'{benchmark}: missing candidate configurations')
    require(qualified > 0, f'{benchmark}: no qualified collection configurations')
    for name, expected in [('qualified_count', qualified),
                          ('unsupported_count', len(rows) - qualified)]:
        require(type(profile.get(name)) is int and profile[name] == expected,
                f'{benchmark}: inconsistent {name}')
    return len(rows), qualified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--benchmark', choices=PROFILES)
    args = parser.parse_args()
    for benchmark in [args.benchmark] if args.benchmark else PROFILES:
        count, qualified = check_profile(args.repo, benchmark)
        print(f'PASS: {benchmark}: current source identity, {count} candidates, '
              f'{qualified} recorded qualified configurations (no new GPU qualification)')


if __name__ == '__main__':
    main()
