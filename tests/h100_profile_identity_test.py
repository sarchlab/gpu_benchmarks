#!/usr/bin/env python3
"""Exercise actual committed profiles and isolated source/configuration drift."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from h100_profile_identity import check_profile


REPO = Path(__file__).resolve().parents[1]
BENCHMARKS = ('cuda_nbody', 'altis_particlefilter')


class ProfileIdentityTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='h100-profile-identity-')
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        self.profiles = {}
        for benchmark in BENCHMARKS:
            relative = f'tier2/{benchmark}/h100_profile.json'
            profile = json.loads((REPO / relative).read_text())
            self.profiles[benchmark] = profile
            for name in [relative, *profile['source_sha256']]:
                destination = self.repo / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REPO / name, destination)

    def test_current_profiles_and_unrelated_docs(self):
        for benchmark, expected in [('cuda_nbody', (75, 23)), ('altis_particlefilter', (27, 24))]:
            with self.subTest(benchmark=benchmark):
                (self.repo / 'tier2' / benchmark / 'README.md').write_text('Documentation only.\n')
                self.assertEqual(check_profile(self.repo, benchmark), expected)

    def test_every_recorded_input_and_digest(self):
        for benchmark, profile in self.profiles.items():
            for name in profile['source_sha256']:
                with self.subTest(benchmark=benchmark, input=name):
                    bad = copy.deepcopy(profile)
                    bad['source_sha256'][name] = '0' * 64
                    with self.assertRaisesRegex(ValueError, 'stale source hash'):
                        check_profile(self.repo, benchmark, bad)
                    path = self.repo / name
                    original = path.read_bytes()
                    try:
                        path.write_bytes(original + b'\n// changed qualification input\n')
                        with self.assertRaisesRegex(ValueError, 'stale source hash'):
                            check_profile(self.repo, benchmark, profile)
                        path.unlink()
                        with self.assertRaises((ValueError, FileNotFoundError)):
                            check_profile(self.repo, benchmark, profile)
                    finally:
                        path.write_bytes(original)

    def test_required_input_coverage(self):
        for benchmark, profile in self.profiles.items():
            for name in profile['source_sha256']:
                with self.subTest(benchmark=benchmark, missing=name):
                    bad = copy.deepcopy(profile)
                    del bad['source_sha256'][name]
                    with self.assertRaisesRegex(ValueError, 'source coverage differs'):
                        check_profile(self.repo, benchmark, bad)
            extra = self.repo / 'tier2' / benchmark / 'new_runtime_helper.h'
            extra.write_text('// A newly introduced runtime dependency.\n')
            with self.assertRaisesRegex(ValueError, 'source coverage differs'):
                check_profile(self.repo, benchmark, profile)

    def test_profile_identity_and_configuration_corruption(self):
        for benchmark, profile in self.profiles.items():
            changes = [
                ('complete', False), ('complete', 1), ('benchmark', 'other'),
                ('profile', 'other'), ('verification', 'sampled'),
                ('process_timeout_seconds', 61), ('process_timeout_seconds', 60.0),
                ('vector_add_warmup', False), ('source_sha256', {}),
                ('configurations', None), ('configurations', profile['configurations'][:-1]),
                ('configurations', profile['configurations'] + profile['configurations'][:1]),
                ('qualified_count', 0), ('qualified_count', True), ('unsupported_count', -1),
            ]
            for key, value in changes:
                with self.subTest(benchmark=benchmark, field=key, value=value):
                    bad = copy.deepcopy(profile)
                    bad[key] = value
                    with self.assertRaises(ValueError):
                        check_profile(self.repo, benchmark, bad)
            axis = 'bodies' if benchmark == 'cuda_nbody' else 'size'
            for key, value in [(axis, True), (axis, 999), ('status', 'pending')]:
                bad = copy.deepcopy(profile)
                bad['configurations'][0][key] = value
                with self.assertRaises(ValueError):
                    check_profile(self.repo, benchmark, bad)

    def test_changed_candidates_cannot_be_hidden_by_refreshing_manifest_hash(self):
        for benchmark, profile in self.profiles.items():
            name = f'tier2/{benchmark}/params.json'
            path = self.repo / name
            original = json.loads(path.read_text())
            for change in ('add-size', 'remove-size', 'add-axis', 'add-timestep'):
                if change == 'add-timestep' and benchmark != 'cuda_nbody':
                    continue
                with self.subTest(benchmark=benchmark, change=change):
                    manifest = copy.deepcopy(original)
                    if change == 'add-size':
                        manifest['scaling_param']['values'].append(1234567)
                    elif change == 'remove-size':
                        manifest['scaling_param']['values'].pop()
                    elif change == 'add-axis':
                        manifest['non_scaling_params'].append(dict(name='new_work', values=[1, 2]))
                    else:
                        next(x for x in manifest['non_scaling_params'] if x['name'] == 'timesteps')['values'].append(100)
                    path.write_text(json.dumps(manifest))
                    bad = copy.deepcopy(profile)
                    bad['source_sha256'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
                    with self.assertRaises(ValueError):
                        check_profile(self.repo, benchmark, bad)

    def test_command_reports_stale_common_header_even_under_python_optimization(self):
        script = REPO / 'tests/h100_profile_identity.py'
        command = [sys.executable, '-O', str(script), '--repo', str(self.repo)]
        valid = subprocess.run(command, capture_output=True, text=True, timeout=20)
        self.assertEqual(valid.returncode, 0, valid.stderr)
        self.assertEqual(valid.stdout.count('PASS:'), 2)
        path = self.repo / 'common/bench_timing.h'
        path.write_bytes(path.read_bytes() + b'\n// shared input changed\n')
        for benchmark in BENCHMARKS:
            bad = subprocess.run(command + ['--benchmark', benchmark], capture_output=True,
                                 text=True, timeout=20)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn('stale source hash for common/bench_timing.h', bad.stderr)
            self.assertIn('Requalify on H100', bad.stderr)


if __name__ == '__main__':
    unittest.main()
