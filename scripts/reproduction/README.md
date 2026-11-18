# Reproduction tools

For pretrained π0.5 evaluation with MuJoCo physics and Isaac images, start with
[setup](isaac-blackwell.md), then [evaluation](pi05-suite.md). The
[target mass switch](pi05-suite.md#simple-phys-target-mass-switch) uses the same
evaluation pipeline. The reference carry command below runs directly in MuJoCo.
[Compatibility](compatibility.md) documents every category of fork behavior.

| File | Responsibility |
|---|---|
| `pi05-tasks.json` | Frozen model/data revisions, six task contracts and metadata hashes |
| `prepare_pi05_suite.py` | Validate inputs and freeze scene/seed manifests |
| `launch_pi05_suite.py` | Start one model service per task and manage evaluation processes |
| `run_pi05_suite.py` | Reset, run and record trials; preserve error/unrun accounting |
| `serve_pi05.py` | Load official model/transforms and reset sampling seed per trial |
| `run_reference_carry.py` | Check two-arm lift, transport and placement using the existing GR00T controller |
| `pi05_common.py` | Small standard-library validation and serialization helpers |
| `hand_diagnostics.py` | Optional AMO hand-parameter comparison and contact telemetry |
| `prepare_g1_assets.py` | Install the checksum-pinned G1/AMO asset bundle |
| `*-requirements.txt` | Separate simulator and model dependencies |

Use the launcher for normal evaluation. Direct evaluator/server commands are for
debugging. There is no separate rendering or replay pipeline to configure.
MuJoCo-only images remain available through `--sim-mode mujoco` for diagnostics,
using the same simulator environment; they are not the visual benchmark baseline.

## Tests

From the repository root:

```bash
python scripts/reproduction/test_pi05_suite.py
python scripts/reproduction/test_pi05_prepare.py
source .env.isaac
PYTHONPATH="$PWD/src:$PYTHONPATH" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv-isaac/bin/python -m pytest -q scripts/reproduction/test_*.py
```

The first two commands use only the standard library. The pytest command requires pytest in
the simulator environment and checks real MuJoCo force/hand behavior, Franka
compatibility, diagnostics and evaluation accounting. Runtime changes also need
a small live inference pilot; unit tests do not establish visual or task success.
Keep pilot outputs separate from formal scores.

## Reference carry

Run from the repository root in the simulator environment. Use the two-table
carry manifest produced by the mass sweep preparer:

```bash
python scripts/reproduction/run_reference_carry.py \
  --manifest PATH --trial episode_000_seed_0_mass_10 --output NEW_DIR
```

The default run checks lift and hold. Add `--carry` to continue to the second
table after the hold check passes. The script uses simulator object poses and
measured contacts to guide wrist IK and the existing GR00T controller. The output
includes `overview.mp4`, `trajectory.json` and `result.json`, which records the
first official success separately from hands-free placement. The
[SIMPLE-Phys project plan](../../docs/projects/simple-phys-project.md) records the
verified scene and experiment evidence.
