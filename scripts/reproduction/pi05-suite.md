# Six-task pretrained pi0.5 evaluation

Protocol: the ten published **Level 0 initial conditions × model seeds 0 and 1**,
with simulation seed 0, for 20 trials per task (120 total). These are ten paired
conditions, not twenty independent scenes. No training is required.

`--sim-mode mujoco` (default) uses MuJoCo physics and images; `--sim-mode mujoco_isaac`
uses the same physics with live Isaac images. Keep the two modes in separate output
directories. Neither mode applies the earlier pilot's gray-floor intervention.

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
The server records the statistics hash, which the evaluator checks against the
manifest before inference. It also checks the live Isaac camera intrinsics and
image dimensions against the task camera.

| Checkpoint/task | Controller | Steps | Official success |
|---|---|---:|---|
| BendPickMP | AMO | 400 | 9 cm lift |
| TabletopGraspMP | AMO | 800 | 1.8 cm lift |
| XMovePick (environment: XMovePickTeleop) | GR00T Balance/Walk | 800 | 9 cm lift |
| XMoveBendPickTeleop | GR00T Balance/Walk | 800 | 8 cm lift |
| HandoverTeleop | GR00T Balance/Walk | 800 | accumulated target/container contact |
| LocomotionPickBetweenTablesTeleop | GR00T Balance/Walk | 1200 | accumulated container contact without hand contact |

### BendPick comparison with the paper

The [paper's single-task Table 1](https://arxiv.org/html/2606.08278v2#S4.T1)
reports pi0.5 BendPick Level 0 as 10/10. Its Table 7 describes a lift above 5 cm,
whereas the released task and `simple.cli.eval` default to reward >= 0.9 with
a 0.1-m lift scale (9 cm above the initial target height). Keep that executable
stopping rule; each BendPick result additionally records `lift_scores.paper_5cm`
when the scored trajectory exceeds 5 cm relative to the same initial height.
This is a secondary interpretation of the paper's prose, not a verified copy of
the authors' unpublished evaluation command. It does not stop a trial early.
Do not combine single-task results with the paper's separate multi-task table.

The [official pi0.5 guide](https://github.com/physical-superintelligence-lab/Psi0/blob/805fcda9c9654e272be2a6673fe5290944fcfb04/baselines/pi05/README.md)
allows either recomputing normalization or rewriting published training statistics.
We use its latter route, with the frozen `c8258285d134d0da644fdbf1a304988398fc3819`
dataset revision. On 2026-11-12 the data repository replaced the old 100-episode
BendPick archive with a 91-episode decoupled-WBC version and different statistics;
the pi0.5 weight remained unchanged. Do not mix those new statistics into this
old checkpoint's baseline. The weight directory contains only `model.safetensors`,
so checkpoint-specific normalization equivalence remains unresolved.

The released task-specific model config is 36 actions, horizon 30, token limit
250, quantile normalization and 224x224 aspect-preserving padded RGB images.
Only `head_stereo_left` is active; wrist image slots are zero-filled and masked.
The task supplies 640x360 RGB at 50 Hz, with its original camera pose, 110-degree
horizontal field of view, scene, materials and lights. No crop, flip, texture
replacement or task-dependent normalization fitting is added. The paper appendix
mentions horizon 16 generically; the task-specific released config specifies 30.
Isaac 5.1 remains necessary for this Blackwell deployment, so it is not a claim of
pixel equivalence to official Isaac 4.5. See [isaac-blackwell.md](isaac-blackwell.md).

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

## Isaac closed-loop evaluation on RTX 5090

The same frozen manifests can be used with `--sim-mode mujoco_isaac`. MuJoCo
still advances physics; the live Isaac head-camera image is sent to pi0.5 on
every policy query. This mode does not replay saved actions. The existing task
budgets, controllers, scene configurations, model seeds and success rules are
unchanged. Isaac renders once after every completed control step. Decoupled tasks
use SIMPLE's `sync_isaac_every_substep=False` option; all four MuJoCo physics
substeps still execute, and every policy/video frame comes from the current state.
No extra rollout pacing is added; renderer/inference wall time
can affect the original WBC interpolation, so this is not a texture-only ablation.

On Blackwell, use the isolated `.venv-isaac` simulator and `.venv-pi05` model
with CUDA 12.8 PyTorch. Preserve the earlier model environment's package pins
except for the compatible PyTorch/torchvision/CUDA stack. Apply the same official
Psi0 transformers replacements in `.venv-pi05` only. The simulator additionally
needs the WBC/client packages in `isaac-blackwell-requirements.txt`, including
Pinocchio 2.7.0 and ONNX Runtime 1.20.1, plus the same pinned standalone WBC
submodules. Follow [isaac-blackwell.md](isaac-blackwell.md)
for rendering requirements and its known asset/version limitations.
Keep Isaac's pinned `websockets==12.0` (the official synchronous client supports
its interface); the separate model service uses `websockets==15.0.1`. Version 12's
synchronous client has no automatic heartbeat, recorded as a null ping timeout.

The Unitree SDK also imports `cyclonedds==0.10.2`. If no Python 3.11 wheel is
available, build its C library inside the project before installing the binding:

```bash
git clone https://github.com/eclipse-cyclonedds/cyclonedds.git .tools/cyclonedds-source
git -C .tools/cyclonedds-source checkout 9995905bce6c4cf9f740d6438bbf7fcfd1c83dfd
cmake -S .tools/cyclonedds-source -B .tools/cyclonedds-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD/.tools/cyclonedds" \
  -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
cmake --build .tools/cyclonedds-build -j 4
cmake --install .tools/cyclonedds-build
export CYCLONEDDS_HOME="$PWD/.tools/cyclonedds"
uv --no-config pip install --python .venv-isaac/bin/python cyclonedds==0.10.2
```

Retain `CYCLONEDDS_HOME` in the private `.env.isaac` activation file. The
in-process GR00T evaluator uses the SDK's simulator types; it does not connect
to a physical robot.

```bash
source .env.isaac
.venv-isaac/bin/python scripts/reproduction/launch_pi05_suite.py \
  --sim-mode mujoco_isaac --isaac-session task --gpu 0 \
  --manifests experiments/reproduction/pi05-isaac/manifests \
  --data-root data/pi05-suite --weights-root data/checkpoints/pi05-suite \
  --output experiments/reproduction/pi05-isaac/formal-01
```

Copy the previously frozen manifests byte-for-byte; run separate pilots on the PR
before merging and starting the formal suite. By default each Isaac trial runs
in a fresh process because Kit cannot restart after shutdown; the task-session
option below defers shutdown until the whole task is finished. The model service stays alive
for the task, but history and sampling seed reset per trial. Shutdown errors stop
the batch and are infrastructure errors, never policy failures.

`head_stereo_left.mp4` records actual Isaac policy observations. AMO tasks also
save `front_stereo_left.mp4` from Isaac. The extra `overview.mp4` remains a MuJoCo
diagnostic view and is explicitly identified as such in results. Per-query PNGs,
image hashes and request/action records verify the closed loop. Report all 20
trials per task; choose success/failure examples only from that frozen set. If a
task has no success or no failure, report that instead of altering the protocol.

### Reuse Kit and distribute independent batches

`--isaac-session task` keeps one Kit application alive for a task. Each trial
still creates fresh Task/Robot/controller instances and a new MuJoCo model. On
close it pauses camera acquisition, stops the Isaac world, clears the World
singleton, opens a new USD stage, then releases the old camera handles. Stage
events release render graphs and subscriptions before handle cleanup. The application
and runtime caches remain loaded; model history and seeds reset as before. The suite records its
summary before final Kit shutdown. `--isaac-session trial` retains the original
process-per-trial fallback. Scene setup loads current actors on demand instead
of preloading and compiling materials for the entire object library.
Both modes render every completed control step and
save complete videos; unused MuJoCo policy cameras are rendered only in debug or
MuJoCo mode, while the recorded MuJoCo overview remains available.

`--gpu N` binds CUDA by the selected GPU UUID, chooses the corresponding Isaac
and MuJoCo EGL device, limits Kit to eight CPU threads, isolates Kit's writable
cache/data directories and defaults the model port to `8015 + N`. To use several
GPUs, use the command above with disjoint `--tasks`, GPU indices and output
directories. For a pilot, select the pilot manifests and a fresh pilot output;
no distributed model service is needed.

Validate repeated and changed initial conditions, input images, fresh stages,
controller/reset state and GPU placement before long runs. Report measured batch
wall time, including startup, separately from per-trial timing. WBC wall-clock
interpolation remains a timing-dependent limitation; speed changes do not imply
bitwise-identical trajectories. Keep optimization pilots separate from formal
results and do not silently merge records from different execution versions.
