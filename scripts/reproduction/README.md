# SIMPLE reproduction tools

Start with the guide for the experiment you want to run:

| Purpose | Guide / entry point |
|---|---|
| Six-task pretrained pi0.5 evaluation (MuJoCo or Isaac) | [pi05-suite.md](pi05-suite.md): prepare → launch → evaluate; optional persistent Isaac sessions and independent GPU workers |
| G1 Isaac rendering on RTX 50-series | [isaac-blackwell.md](isaac-blackwell.md): `render_g1_isaac.py` |
| Earlier BendPick floor-appearance pilot | [pi05.md](pi05.md): `run_pi05.py` |
| G1 motion-planner demonstration or action replay | [G1 instructions](#g1-whole-body-bend-and-pick-pure-mujoco): `run_g1_bend_pick.py` |
| Installation and component diagnostics | Instructions below: `probe.py`, `run_mujoco_robot.py` |

The floor-appearance pilot and motion-planner demonstration use different protocols
from the six-task evaluation. Keep their outputs separate. Shared pi0.5 metadata
validation and JSON/hash helpers live in `pi05_common.py`; it needs only Python's
standard library. The CLI scripts retain their existing names and arguments.

## Installation diagnostics

These scripts first distinguish package/platform compatibility from MuJoCo
component functionality. They do not modify SIMPLE's simulator, task logic,
dependency lock, global Git configuration, driver, system Python, or other projects.

Use one clone under the server's existing persistent `code/` directory. Keep
environments, managed Python, caches, data, and experiment outputs inside it.
The bootstrap creates `.venv-probe` with Python 3.10 and CPU PyTorch, deliberately
excluding Isaac Sim, cuRobo, SONIC, policy weights, and benchmark data. It is not
equivalent to the README's full installation. Source imports use `src/`.

On a slow shared filesystem, optionally set `UV_CACHE_DIR` to a disposable local
scratch directory for package extraction. This cache is reconstructible; the
interpreter and final `.venv-probe` still reside inside the persistent project.
For slow download links, `PROBE_WHEELHOUSE` may point to an offline wheel directory
downloaded from the same two official indexes for CPython 3.10, Linux x86_64 and
a compatible manylinux tag. Include transitive dependencies, transfer with a
SHA-256 manifest, and verify it before installation. The pinned versions are
unchanged and offline mode never falls back to a network index.

```bash
UV_BIN=/path/to/existing/uv bash scripts/reproduction/bootstrap_probe.sh
.venv-probe/bin/python scripts/reproduction/probe.py \
  --uv /path/to/existing/uv \
  --output experiments/reproduction/<unique-run-id>
```

Exit 2 means the reproduction remains incomplete. Each subprocess has a time
limit and a separate log. Existing result directories are never overwritten.
The Isaac package check is a dry run. The official task attempt uses unmodified
`scripts/test_env.py`, pure MuJoCo, 30 steps, and a 120-second timeout. A missing
cuRobo error in this partial environment is not a hardware incompatibility claim.
`mujoco_smoke.png` is a synthetic falling-sphere component check, not a robot
rollout, policy success, or paper result. Physics results are saved even if EGL fails.

## Full reproduction prerequisites

The upstream README requires an RTX GPU and Ubuntu 22.04 for its Isaac Sim 4.5
and MuJoCo 3.3 pipeline. NVIDIA lists GPUs without RT cores (including A100) as
unsupported for Isaac Sim. Check the selected Isaac package's actual wheel
compatibility against the host glibc; do not upgrade system glibc in place.

- [SIMPLE installation and evaluation](https://github.com/physical-superintelligence-lab/SIMPLE#installation)
- [Dual-simulator architecture](https://github.com/physical-superintelligence-lab/SIMPLE/blob/main/docs/source/core/architecture.md)
- [NVIDIA requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)

After obtaining a compatible simulation host: initialize the pinned submodules
and LFS assets, perform the official full installation including cuRobo, validate
the official task, then acquire the chosen Psi0 checkpoint and evaluation episodes.
The policy server may run on the A100 while the Isaac client runs on an RTX host.
Record both commits, asset/checkpoint identifiers, commands, seed, episode count,
domain-randomization level, success statistics, and videos. Do not substitute a
MuJoCo-only image pipeline for the published visual benchmark.

Follow the PR-before-server-validation workflow. If actual task reproduction is
blocked, retain the draft PR with the exact tested commit and logs; do not merge
or present component checks as a successful full reproduction.


## Pure MuJoCo robot task

The A100 can run the MuJoCo path. This phase uses the existing registered
`simple/FrankaTabletopGraspMP-v0` task, Franka FR3 robot, GraspNet object 63 and
HSSD scene3, with `sim_mode=mujoco`. No Isaac or SONIC dependencies are installed.
The initial acceptance criterion is reset, camera observations and controlled
robot motion, not successful grasping or visual benchmark parity.

`bootstrap_mujoco.sh` creates `.venv` using the existing persistent Python 3.10,
installs pinned CUDA 11.8 PyTorch and cuRobo dependencies from verified offline
wheels, and builds the upstream cuRobo submodule at its pinned commit. Its CUDA
compiler/runtime headers live in `.tools/cuda-11.8`; system CUDA is unchanged.
Download the three component archives from NVIDIA's CUDA 11.8 redistribution
index. Their official SHA-256 values are checked before extraction.

```bash
git submodule update --init --depth 1 third_party/curobo
# Set machine-specific UV_BIN, MUJOCO_WHEELHOUSE, CUDA_ARCHIVES_DIR in the shell.
bash scripts/reproduction/bootstrap_mujoco.sh
PYTHONPATH=src .venv/bin/python -m pytest -q scripts/reproduction/test_franka_qpos.py
.venv/bin/python scripts/reproduction/run_mujoco_robot.py \
  --output experiments/reproduction/<unique-run> --steps 120 --seeds 7 8
```

The runner executes cuRobo GPU forward kinematics, verifies real joint motion and
finite simulation state, and saves per-episode videos, first/last images and
trajectories. It reports the task's actual reward and termination without changing
the success checker. Seeded sinusoidal joint commands are a functional test,
not a trained grasp policy. Assets stay in the project's `data/` directory.

Franka control setup accepts both `finger_actuator1/2` (legacy assets) and
`finger_joint1/2` (the published FR3 asset). Tests exercise both name layouts
through real MuJoCo model/data objects. The shared Franka position getter also
iterates joint values; its test checks order, names and scalar values. FR3 has
its own position getter and is validated by the task rollout.

## G1 whole-body bend and pick (pure MuJoCo)

The next acceptance scope is the unchanged `simple/G1WholebodyBendPickMP-v0`
with AMO balance control, original scene0, cracker box 0, three distractors and
cached Dex3 grasps. Download the three ZIP names listed by
`prepare_g1_assets.py` from `USC-PSI-Lab/SIMPLE` on Hugging Face. Materialize the
three `src/simple/robots/policy/*.pt` Git LFS files from upstream, then place
copies beside the ZIPs in an input directory. The installer verifies all six
SHA-256 hashes and retains the original project data layout. Existing GraspNet
assets and the CUDA/cuRobo environment from the preceding setup are required.

```bash
.venv/bin/python scripts/reproduction/prepare_g1_assets.py --inputs <verified-inputs>
.venv/bin/python -u scripts/reproduction/run_g1_bend_pick.py \
  --output experiments/reproduction/<unique-g1-run> --seeds 7 8 9
```

The runner uses the production `MotionPlannerAgent` and its multi-phase
`synthesize()`/execute sequence. It records the original task's success signal
but completes the planned final stand, checking actual right-hand contact and
sustained object lift to reject transient tosses. No object welding, qpos
teleportation, gravity disabling, or reward/threshold changes are used. The
additional overview camera affects only rendering. Videos show the entire
standing/bending/grasping/lifting sequence at 50 fps; telemetry and action logs
are retained even on failures. It stops after the first passing attempt, so
this is a functional demonstration, not an unbiased success-rate estimate.

G1 hand motors now refresh their existing PD position targets at each physics
substep, including `height_adjust` and `stand`. Previously those locomotion
commands retained the last grasp torque without feedback. Torque replay is
preserved. The real-MuJoCo regression test checks the grasp-to-lift transition
and replay behavior without requiring the full robot checkpoint:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q scripts/reproduction/test_g1_hand_hold.py
```

For humanoid tasks with global gravity enabled, object synthetic gravity must
be zero. The engine previously applied both, doubling a 0.1 kg object's weight.
The gravity regression tests measure free-fall acceleration in real MuJoCo for
both global-gravity and legacy synthetic-gravity modes, including containers.
Normal gravity, object masses, contact/friction parameters and reward thresholds
are retained; this is a bug-fixed functional reproduction, not an unmodified
upstream benchmark score.

A seed alone does not guarantee an identical cuRobo plan; independent replanning
can fail even after one successful demonstration. Freeze a passing episode's
`initial-layout.json` and `actions.json` to reproduce that trajectory in a fresh
physics process:

```bash
.venv/bin/python -u scripts/reproduction/run_g1_bend_pick.py \
  --output experiments/reproduction/<unique-replay> \
  --replay-episode experiments/reproduction/<planning-run>/seed-8
```

Replay resets the original task to the saved initial scene, then sends the saved
commands through live AMO and MuJoCo. It does not assign recorded robot/object
states during the rollout. It runs the same contact, sustained-lift and original
success checks, records a new video, and hashes its input artifacts. Reports
must distinguish fresh motion planning from action-trace reproduction.

## SIMPLE-Phys project plan

See the [SIMPLE-Phys project definition and implementation plan](../../docs/projects/simple-phys-project.md)
for confirmed research requirements, proposed milestones, and pending compatibility checks.
The PHYS benchmark is planned work; the existing G1 reproduction is not its completed evaluation.
