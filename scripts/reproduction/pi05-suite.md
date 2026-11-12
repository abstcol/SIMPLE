# Six-task pi0.5 MuJoCo evaluation

Protocol: the ten published **Level 0 initial conditions × model seeds 0 and 1**,
with simulation seed 0, for 20 trials per task (120 total). These are ten paired
conditions, not twenty independent scenes. Use the original MuJoCo appearance;
this suite has no gray-floor intervention and is not an Isaac-rendered evaluation.
No training is required.

The entry points have one responsibility each:
`prepare_pi05_suite.py` freezes manifests, `launch_pi05_suite.py` manages model
services and task processes, and `run_pi05_suite.py` executes and records trials.
They share the lightweight validation/serialization functions in `pi05_common.py`.
`serve_pi05.py` supplies the unchanged official model and transforms.

`pi05-tasks.json` freezes official model/data revisions, source-file SHA256 hashes,
normalization statistics, each task's controller, step budget and success rule.
The model architecture/transforms are unchanged. Each checkpoint uses its own
published training `meta/stats.json`; checkpoint-specific statistics are not
published, so equivalence to the authors' training normalization remains unverified.

| Checkpoint/task | Controller | Steps | Official success |
|---|---|---:|---|
| BendPickMP | AMO | 400 | 9 cm lift |
| TabletopGraspMP | AMO | 800 | 1.8 cm lift |
| XMovePick (environment: XMovePickTeleop) | GR00T Balance/Walk | 800 | 9 cm lift |
| XMoveBendPickTeleop | GR00T Balance/Walk | 800 | 8 cm lift |
| HandoverTeleop | GR00T Balance/Walk | 800 | accumulated target/container contact |
| LocomotionPickBetweenTablesTeleop | GR00T Balance/Walk | 1200 | accumulated container contact without hand contact |

Lift criteria do not require sustained holding. Contact rewards are stateful; the
runner adds no extra reward/success calls. A decoupled success receives the official
50-step video tail, excluded from scoring. The official pure-MuJoCo execution timing
is retained: stabilization is paced, rollout is not artificially paced. WBC
interpolation uses wall time, which is recorded for diagnosis.

The XMoveBendPickTeleop task was removed from current upstream. Its task class is
restored from official commit `7089306435d75368a113ca713c0c4dacdc76529e`, not replaced
with BendPickTeleop. The four decoupled tasks use GR00T WBC; the `g1_sonic` robot
class name does not imply SONIC token control.

## Preparation and execution

Keep weights in `data/checkpoints/pi05-suite/<weight_task>/model.safetensors` and
metadata in `data/pi05-suite/{train,eval}/<environment_id_without_simple_prefix>`.
Verify every weight against the frozen Hugging Face LFS SHA256 before running.
The official eval archives contain `.pre-fk` backups; use their current
`episodes.jsonl`, whose published hash is checked by the manifest generator.

```bash
python scripts/reproduction/prepare_pi05_suite.py \
  --data-root data/pi05-suite --output experiments/reproduction/pi05-suite/manifests
python scripts/reproduction/prepare_pi05_suite.py \
  --data-root data/pi05-suite --output experiments/reproduction/pi05-suite/pilot-manifests --pilot
python scripts/reproduction/test_pi05_suite.py
```

Run the six paired pilots (12 trials) on the PR commit. Record the tested commit and results
in the PR; merge after runtime validation, then run the frozen 120-trial manifests
on merged main. Pilot attempts are separate and never included in the denominator.
An infrastructure error stops execution and records all remaining trials as unrun;
it must not be counted as a policy failure or silently excluded.

`run_pi05_suite.py --manifest <task.json> --output <new_directory> --port 8015`
connects to `serve_pi05.py --task <weight_task> --checkpoint <file> --stats <file>`.
Each trial rebuilds the simulator/controller and explicitly resets model history
and sampling seed. Existing output directories cannot be overwritten.

The model environment is the existing project-local `.venv-pi05`; simulation
uses the existing Python 3.10 dependencies and a separate dependency overlay.
The required extra packages are CPU `onnxruntime==1.20.1`, `coloredlogs==15.0.1`,
`flatbuffers==25.9.23`, and `humanfriendly==10.0`. Initialize the pinned submodules and check that the two official GR00T
Balance/Walk ONNX files and required URDF meshes are real files. The standalone
`decoupled_wbc` commit includes its required assets as ordinary Git blobs.
Use SIMPLE's pinned standalone packages, `decoupled_wbc@5b304e4d643fe89cd000068dcbba7e7519539caa`
and `gear_sonic@e61eeb1dec9678dcc02f97c5ac27ca6a4fe82128`, with `third_party`
before GR00T on `PYTHONPATH`. GR00T's bundled packages have the same import names
but incompatible reset and navigation APIs. The standalone controller consumes all
four navigation fields, including the turning flag and target yaw.
Keep all environments, inputs and caches within the existing project tree.

Evidence per trial includes configuration/provenance, reset and stabilized state,
all policy images, raw action chunks, executed commands, per-step state/reward/time,
and complete head/overview videos. Verify video frame counts and paired initial
states before interpreting seed differences. Preserve failed episodes and report
success by task, scene and seed; these six task-specific models are not a common-task
VLA ranking.

The batch launcher starts one local model server at a time and cleans up only its
own processes. After the manifests and inputs are ready:

```bash
.venv-sonic/bin/python scripts/reproduction/launch_pi05_suite.py \
  --manifests experiments/reproduction/pi05-suite/pilot-manifests \
  --data-root data/pi05-suite --weights-root data/checkpoints/pi05-suite \
  --output experiments/reproduction/pi05-suite/pilot-01
# After PR validation and merge, use manifests (120 trials) and a fresh output.
```

`--tasks` accepts environment-name suffixes for an explicitly recorded task subset.
The launcher records Git SHA, commands, environment overrides and child exit codes.
It rejects an occupied service port and changed manifests/statistics.

The client heartbeat timeout is 600 seconds: the official server performs inference
synchronously, so a cold first inference on slow storage can exceed the default
20-second pong deadline. This transport allowance does not change model inputs,
outputs, task budgets or simulated time; each request's wall time is recorded.
