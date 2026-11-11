"""Lightweight tests for scoring and fail-fast accounting; no simulator needed."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location('pi05_suite', Path(__file__).with_name('run_pi05_suite.py'))
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


def trial(index=0):
    return dict(trial_id=f'trial_{index}', env_id='simple/G1WholebodyBendPickMP-v0',
                controller='amo', episode_index=index, model_seed=0, sim_seed=0,
                environment_config={'dr_state_dict': {'scene': {'uid': 'scene_102344280'}}},
                max_steps=400, success_criteria=.9)


class SuiteTest(unittest.TestCase):
    def test_registry_clear_recreates_robot_history_without_unregistering_classes(self):
        path = Path(__file__).resolve().parents[2] / 'src/simple/core/registry.py'
        registry_spec = importlib.util.spec_from_file_location('isolated_registry', path)
        registry = importlib.util.module_from_spec(registry_spec)
        registry_spec.loader.exec_module(registry)

        class TaskRegistry(registry.RegistryMixin):
            pass

        class RobotRegistry(registry.RegistryMixin):
            pass

        @RobotRegistry.register('robot')
        class Robot:
            def __init__(self):
                self.controller_history = []

        @TaskRegistry.register('task')
        class Task:
            def __init__(self):
                self.robot = RobotRegistry.make('robot')

        first = TaskRegistry.make('task')
        first.robot.controller_history.append('previous trial')
        self.assertIs(first, TaskRegistry.make('task'))
        self.assertIs(TaskRegistry._instances, RobotRegistry._instances)
        registered_classes = dict(TaskRegistry._registry)
        suite.clear_registry_instances(TaskRegistry, RobotRegistry)
        second = TaskRegistry.make('task')
        self.assertIsNot(first, second)
        self.assertIsNot(first.robot, second.robot)
        self.assertEqual(second.robot.controller_history, [])
        self.assertEqual(TaskRegistry._registry, registered_classes)

    def test_success_on_last_budgeted_step_is_not_lost_or_trailed(self):
        raw, phases = SimpleNamespace(_success=False), []

        def step(phase):
            phases.append(phase)
            raw._success = len(phases) == 3
            return raw._success, len(phases) == 3

        success, termination = suite.rollout(step, raw, 3, 50)
        self.assertTrue(success)
        self.assertEqual(termination, 'terminated')
        self.assertEqual(phases, ['scored'] * 3)

    def test_tail_preserves_scored_success_even_if_object_drops(self):
        raw, phases = SimpleNamespace(_success=False), []

        def step(phase):
            phases.append(phase)
            raw._success = phase == 'scored'
            return raw._success, False

        success, _ = suite.rollout(step, raw, 400, 50)
        self.assertTrue(success)
        self.assertTrue(raw._success)
        self.assertEqual(phases, ['scored'] + ['trailing'] * 50)

    def test_interface_error_propagates(self):
        def step(_):
            raise ValueError('wrong action dimension')

        with self.assertRaisesRegex(ValueError, 'action dimension'):
            suite.rollout(step, SimpleNamespace(_success=False), 400, 50)

    def test_configs_preserve_original_and_contracts_cannot_mix(self):
        source = trial()
        frozen = deepcopy(source)
        suite.validate_manifest([source])
        self.assertEqual(source, frozen)
        self.assertEqual(suite.decode_environment_config(source['environment_config'])
                         ['dr_state_dict']['scene']['uid'], 'scene_scene3')
        second = trial(1)
        second['controller'] = 'decoupled'
        with self.assertRaisesRegex(ValueError, 'one task'):
            suite.validate_manifest([source, second])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            suite.validate_manifest([source, source])
        source['trial_id'] = '../outside'
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            suite.validate_manifest([source])

    def test_fail_fast_reports_every_unrun_trial_and_forbids_overwrite(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, output = root / 'manifest.json', root / 'output'
            manifest.write_text(json.dumps([trial(i) for i in range(3)]))
            argv = ['suite', '--manifest', str(manifest), '--output', str(output)]
            with patch('sys.argv', argv), patch.object(suite, 'run_trial') as run:
                run.side_effect = [dict(trial_id='trial_0', outcome='failure'),
                                   ValueError('bad checkpoint')]
                with self.assertRaisesRegex(ValueError, 'checkpoint'):
                    suite.main()
                self.assertEqual(run.call_count, 2)
                summary = json.loads((output / 'summary.json').read_text())
                self.assertEqual([r['outcome'] for r in summary['results']],
                                 ['failure', 'error', 'unrun'])
                self.assertEqual(summary['unrun'], ['trial_2'])
                self.assertEqual(summary['completed'], 1)
                with self.assertRaises(FileExistsError):
                    suite.main()
                self.assertEqual(run.call_count, 2)


if __name__ == '__main__':
    unittest.main()
