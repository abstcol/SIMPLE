# Reproduction tools

The supported workflow is **MuJoCo physics + Isaac images + pretrained π0.5**.
Start with [setup](isaac-blackwell.md), then [evaluation](pi05-suite.md).
[Compatibility](compatibility.md) documents every category of fork behavior.

| File | Responsibility |
|---|---|
| `pi05-tasks.json` | Frozen model/data revisions, six task contracts and metadata hashes |
| `prepare_pi05_suite.py` | Validate inputs and freeze scene/seed manifests |
| `launch_pi05_suite.py` | Start one model service per task and manage evaluation processes |
| `run_pi05_suite.py` | Reset, run and record trials; preserve error/unrun accounting |
| `serve_pi05.py` | Load official model/transforms and reset sampling seed per trial |
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
source .env.isaac
PYTHONPATH="$PWD/src:$PYTHONPATH" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv-isaac/bin/python -m pytest -q scripts/reproduction/test_*.py
```

The first command uses only the standard library. The second requires pytest in
the simulator environment and checks real MuJoCo force/hand behavior, Franka
compatibility, diagnostics and evaluation accounting. Runtime changes also need
a small live inference pilot; unit tests do not establish visual or task success.
Keep pilot outputs separate from formal scores.

The [SIMPLE-Phys plan](../../docs/projects/simple-phys-project.md) describes future
research scope. It is separate from the current pretrained-policy evaluation.
