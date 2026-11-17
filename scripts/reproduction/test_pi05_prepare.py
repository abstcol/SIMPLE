"""Offline preparation checks: paired sweeps, frozen inputs and legacy commands."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import prepare_pi05_suite as prepare
from pi05_common import select_tasks, sha256, validate_manifest


class PrepareTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        self.name = 'G1WholebodyLocomotionPickBetweenTablesTeleop-v0'
        rows = [dict(episode_index=i, environment_config=json.dumps({
            'uid': 'carry', 'metadata': {'dr_level': 0},
            'dr_state_dict': {'scene': {'uid': 'hssd:scene6'}, 'condition': i}})) for i in range(10)]
        self.rows = rows
        episodes = self.data / 'eval' / self.name / 'dr-level-0/meta/episodes.jsonl'
        episodes.parent.mkdir(parents=True)
        episodes.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        stats = self.data / 'train' / self.name / 'meta/stats.json'
        stats.parent.mkdir(parents=True)
        stats.write_text(json.dumps({field: {k: [0] * width for k in ('mean', 'std', 'q01', 'q99')}
                                    for field, width in [('states', 32), ('action', 36)]}))
        contract = dict(env_id='simple/' + self.name, task_uid='carry', controller='decoupled',
            weight_task=self.name, max_steps=1200, success_criteria=.4,
            checkpoint_subdir='openpi-05/carry/40000', metadata_source={
                'episodes_sha256': sha256(episodes.read_bytes()), 'train_stats_sha256': sha256(stats.read_bytes())})
        self.catalog = self.root / 'catalog.json'
        # Other tasks deliberately have no local data: selection must precede reads.
        self.catalog.write_text(json.dumps({'contracts': [contract] + [
            {**contract, 'env_id': f'simple/other_{i}'} for i in range(5)],
            'model_repo': 'models', 'model_revision': 'frozen', 'data_repo': 'data', 'data_revision': 'frozen'}))

    def invoke(self, *options, output='out', selected=True):
        dest = self.root / output
        argv = ['prepare', '--data-root', str(self.data), '--output', str(dest)]
        if selected:
            argv += ['--tasks', self.name]
        with patch.object(prepare, 'CATALOG_PATH', self.catalog), patch('sys.argv', argv + list(options)), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            prepare.main()
        return dest

    def test_nominal_retains_two_seeds_and_legacy_ids(self):
        out = self.invoke()
        records = json.loads((out / (self.name + '.json')).read_text())
        self.assertEqual(len(records), 20)
        self.assertEqual(records[0]['trial_id'], 'episode_000_seed_0')
        self.assertEqual({r['model_seed'] for r in records}, {0, 1})
        self.assertTrue(all('physics' not in r for r in records))
        self.assertEqual(json.loads((out / 'summary.json').read_text())['total_trials'], 20)

    def test_sweep_is_paired_with_unique_ids_and_original_task_contract(self):
        out = self.invoke('--model-seeds', '0', '--mass-scales', '1', '10', '20')
        records = json.loads((out / (self.name + '.json')).read_text())
        summary = json.loads((out / 'summary.json').read_text())
        self.assertEqual(len(records), 30)
        self.assertEqual(len({r['trial_id'] for r in records}), 30)
        self.assertEqual(summary['tasks'][0]['trials'], summary['total_trials'])
        self.assertEqual(summary['tasks'][0]['manifest_sha256'], sha256((out / (self.name + '.json')).read_bytes()))
        for i, row in enumerate(self.rows):
            group = [r for r in records if r['episode_index'] == i]
            self.assertEqual({r['physics']['mass_scale'] for r in group}, {1, 10, 20})
            self.assertTrue(all(r['environment_config'] == row['environment_config'] for r in group))
            self.assertTrue(all(r['model_seed'] == r['sim_seed'] == 0 for r in group))
            self.assertTrue(all(r['max_steps'] == 1200 and r['success_criteria'] == .4 for r in group))
            self.assertTrue(all('hand_profile' not in r and r['physics']['enabled'] for r in group))
        self.assertEqual([r['physics']['mass_scale'] for r in records[:6]], [1, 10, 20, 10, 20, 1])
        validate_manifest(records)

    def test_single_scale_legacy_ids_and_pilot_accounting(self):
        out = self.invoke('--pilot', '--model-seeds', '0', '--mass-scale', '2')
        records = json.loads((out / (self.name + '.json')).read_text())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['trial_id'], 'pilot_episode_000_seed_0')
        self.assertFalse(records[0]['counts_toward_main'])
        self.assertEqual(records[0]['physics'], {'enabled': True, 'mass_scale': 2})

    def test_close_scales_get_distinct_round_trip_labels(self):
        out = self.invoke('--pilot', '--model-seeds', '0', '--mass-scales', '1', '1.0000000001')
        records = json.loads((out / (self.name + '.json')).read_text())
        self.assertEqual(len({r['trial_id'] for r in records}), 2)

    def test_invalid_requests_do_not_create_output(self):
        for i, options in enumerate([
            ['--mass-scales', '0'], ['--mass-scales', 'nan'], ['--mass-scales', 'inf'],
            ['--mass-scales', '1', '1.0'], ['--mass-scale', '2', '--mass-scales', '2'],
            ['--model-seeds', '0', '0'], ['--model-seeds', '-1'], ['--model-seeds', str(2**63)],
            ['--tasks', 'unknown'], ['--tasks', self.name, self.name],
        ]):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                self.invoke(*options, output=f'invalid-{i}')
            self.assertFalse((self.root / f'invalid-{i}').exists())

    def test_input_hash_mismatch_does_not_create_output(self):
        stats = self.data / 'train' / self.name / 'meta/stats.json'
        stats.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'metadata mismatch'):
            self.invoke()
        self.assertFalse((self.root / 'out').exists())

    def test_task_selection_is_shared_and_preserves_requested_order(self):
        items = [{'env_id': 'simple/a'}, {'env_id': 'simple/b'}]
        self.assertEqual(select_tasks(items, ['b', 'a']), items[::-1])
        self.assertEqual(select_tasks(items), items)
        with self.assertRaisesRegex(ValueError, 'unique'):
            select_tasks(items, ['a', 'a'])


if __name__ == '__main__':
    unittest.main()
