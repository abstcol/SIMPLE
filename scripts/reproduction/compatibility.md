# Compatibility and upstream differences

Comparison base: upstream SIMPLE
[`2c38755`](https://github.com/physical-superintelligence-lab/SIMPLE/tree/2c38755fe78140d3511a5ca0ebc0b1f20352ddd9).
This inventory describes the fork; it does not assert equivalence to the paper.

## Physics and task behavior

The evaluation launcher and evaluator default to `--physics-profile upstream`.
This applies the upstream extra downward object force even when global gravity
is enabled, and computes AMO hand torque once per action, holding it over physics
substeps. `--physics-profile project` selects the earlier fork behavior: no extra
object force with global gravity, and hand PD refreshed every physics substep.
Normal gravity remains enabled in both modes. Results record the profile and
actual applied object forces. Never pool these profiles into one success rate.

For backward compatibility, the generic engine/robot classes still have their
earlier fork defaults; these evaluation scripts explicitly set the profile before
reset. Calling other upstream entry points does not automatically select the
reproduction launcher's settings. Relevant source: `engines/mujoco.py` and
`robots/g1_wholebody.py` under `src/simple/`.

Default hand stiffness, damping and limits are the current upstream values.
An AMO trial may explicitly set `hand_profile: pre_nov9` to restore only named
hand stiffness/software torque limits from upstream `b75c6786`, after stabilization.
This is a diagnostic intervention, not a baseline. `hand_diagnostics.py` records
parameters and end-of-control-step contacts/forces; brief substep events may be missed.

`G1WholebodyXMoveBendPickTeleop-v0` is restored from upstream commit
`7089306435d75368a113ca713c0c4dacdc76529e`, including registration, because the
frozen checkpoint refers to this historical task. It is not an alias for a new task.
Franka compatibility accepts both published finger-actuator naming schemes and
fixes iteration over joint values. No Franka demo is part of the humanoid suite.

## Rendering and lifecycle

- RTX 5090 uses Isaac 5.1/Python 3.11 instead of upstream main's Isaac 4.5/Python
  3.10. Separate environment requirements leave upstream `pyproject.toml` and
  `uv.lock` unchanged. This is not pixel-equivalent validation.
- Plain USD references avoid applying both Isaac 5.1's automatic units/up-axis
  conversion and SIMPLE's existing scene transforms. Source assets are unchanged.
- Actors load on demand. Isaac renders once per completed control step; unused
  MuJoCo policy cameras are skipped. A MuJoCo overview is still recorded.
- `--isaac-session task` reuses Kit, releases the old stage and creates fresh
  task/robot/controller state per trial. Default `trial` mode uses a new process.
  Registry instance caches are cleared before each reset. Model history and seed
  reset on every trial, even though its service remains alive per task.
- GPU selection, CPU thread limits and project-local Kit/cache paths affect
  placement and runtime. WBC interpolation uses wall time, so speed changes can
  affect trajectories; do not describe them as physics-neutral guarantees.

These changes live in `engines/isaac_app.py`, `engines/isaacsim.py`,
`envs/base_dual_env.py`, `envs/loco_manipulation.py`, `envs/sonic_loco_manip.py`,
and the reproduction scripts. Some HSSD `props/*.usd` references remain unresolved
in the available assets. Rendered images are not certified complete upstream replicas.

## Model, inputs and scoring

Official Psi0 source is pinned to `805fcda9c9654e272be2a6673fe5290944fcfb04`.
`serve_pi05.py` loads its released π0.5 architecture, official transforms and
transformers replacement, with 36 actions, horizon 30, token limit 250 and quantile
normalization. The active head RGB image is resized/padded to 224×224; wrist slots
are masked. The task camera supplies 640×360 RGB with its original pose and 110°
horizontal field of view. No texture replacement is applied.

Each weight uses frozen published training `meta/stats.json`, following the
[official guide](https://github.com/physical-superintelligence-lab/Psi0/blob/805fcda9c9654e272be2a6673fe5290944fcfb04/baselines/pi05/README.md).
Checkpoint-specific statistics were not supplied with these weights; agreement
with the authors' actual training normalization is unresolved. Do not substitute
newer dataset statistics. The evaluator verifies manifest/statistics hashes and
Isaac camera dimensions/intrinsics; it records images and request timing.

Success uses the released executable task rules. BendPick stops at 9 cm lift;
its additional `lift_scores.paper_5cm` field interprets the paper's 5 cm prose and
does not change termination. Contact rewards retain their upstream call cadence.
AMO performs 60 stand steps; decoupled WBC has its stabilization phase. Stabilization
and the decoupled 50-step success-video tail are excluded from scored steps.
Infrastructure errors stop the batch and leave remaining trials `unrun`.

The websocket timeout accommodates slow cold inference, without altering action
budgets. The four `decoupled` tasks use GR00T Balance/Walk, not SONIC token control,
despite the `g1_sonic` robot class name.

## Cleanup migration

The public launcher now defaults to Isaac images and upstream physics. Old commands
that relied on implicit MuJoCo/project defaults must specify those options explicitly.
Both image modes now use `.venv-isaac`; the old cross-environment Python package
overlays were removed. Standalone probes, demos, floor-texture pilots and replay
entry points were retired; existing outputs and inputs remain in place. Submodule
URLs use HTTPS so cloning does not require a GitHub SSH key.
