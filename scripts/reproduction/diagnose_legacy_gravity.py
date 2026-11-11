"""Replay-only diagnostic of the synthetic weight used before commit 6c58d8c.

This intentionally restores the old extra force for this process only. It is not
a valid PHYS baseline or a replacement for the corrected nominal evaluation.
Run via EVAL_WBC_WRAPPER; all other evaluation inputs and budgets stay unchanged.
"""
from run_sonic_eval import ROOT
import json
import runpy
import sys
from pathlib import Path

from simple.engines.mujoco import MujocoSimulator

if len(sys.argv) < 3 or sys.argv[2] != 'replay':
    raise SystemExit('This diagnostic is restricted to reference replay.')
if '--eval-dir' not in sys.argv:
    raise SystemExit('An explicit diagnostic --eval-dir is required.')
output = Path(sys.argv[sys.argv.index('--eval-dir') + 1])
output.mkdir(parents=True, exist_ok=False)
records = []


def legacy_object_gravity(self, mj_obj, is_container=False):
    mj_obj.xfrc_applied[:] = 0
    mj_obj.xfrc_applied[2] = -9.81 * (1.0 if is_container else 0.1)
    records.append({'body': mj_obj.name, 'container': is_container,
                    'global_gravity': self.mjModel.opt.gravity.tolist(),
                    'extra_force': mj_obj.xfrc_applied.tolist()})


MujocoSimulator._apply_object_gravity = legacy_object_gravity
try:
    runpy.run_module('simple.cli.eval_wbc', run_name='__main__')
finally:
    (output / 'legacy-gravity.json').write_text(json.dumps({
        'scope': 'Diagnostic only: restore pre-6c58d8c synthetic force; no training',
        'source_commit': 'a60cd3caff4b4d5df6cc7257d7ffaeaa50c5cc3b',
        'applied_forces': records,
    }, indent=2) + '\n')
