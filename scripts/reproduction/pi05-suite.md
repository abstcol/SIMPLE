# Six-task π0.5 evaluation

The protocol is **10 published Level 0 initial conditions × model seeds 0 and 1**,
with simulation seed 0: 20 trials per task, 120 total. These are ten paired scenes,
not twenty independent scenes. Each task uses its own released checkpoint.

| Environment suffix after `G1Wholebody` | Controller | Scored steps | Executable success rule |
|---|---|---:|---|
| `BendPickMP-v0` | AMO | 400 | 9 cm lift |
| `TabletopGraspMP-v0` | AMO | 800 | 1.8 cm lift |
| `XMovePickTeleop-v0` | GR00T Balance/Walk | 800 | 9 cm lift |
| `XMoveBendPickTeleop-v0` | GR00T Balance/Walk | 800 | 8 cm lift |
| `HandoverTeleop-v0` | GR00T Balance/Walk | 800 | Accumulated target/container contact |
| `LocomotionPickBetweenTablesTeleop-v0` | GR00T Balance/Walk | 1200 | Accumulated container contact without hand contact |

The lift criteria do not require sustained holding. XMovePick's weight name omits
`Teleop`; the catalog handles this mapping. GR00T Balance/Walk is not SONIC token
control. Read [compatibility](compatibility.md) before comparing to paper results.

## Prepare inputs

Complete [setup](isaac-blackwell.md) first. Existing verified inputs can be reused.
The source of truth is [pi05-tasks.json](pi05-tasks.json):

- Weights: `USC-PSI-Lab/psi-model@74d7abaa61b51dfe4e95f539a83738c276ce27b2`.
  Download each contract's `checkpoint_subdir/model.safetensors` to
  `data/checkpoints/pi05-suite/<weight_task>/model.safetensors`.
- Metadata: dataset `USC-PSI-Lab/psi-data@c8258285d134d0da644fdbf1a304988398fc3819`.
  Each contract names the eval/train ZIP, archive hash and exact member to extract.
  Put `episodes_member` under `data/pi05-suite/eval/` and `train_stats_member`
  under `data/pi05-suite/train/`. Only these metadata members are needed for evaluation;
  `.pre-fk` backups are not inputs.

For example, the BendPick inputs can be obtained with the model environment's HF CLI
and `unzip` (training archives can be large; keep downloads on persistent storage):

```bash
.venv-pi05/bin/hf download USC-PSI-Lab/psi-model \
  --revision 74d7abaa61b51dfe4e95f539a83738c276ce27b2 \
  openpi-05/G1WholebodyBendPickMP-v0/20000/model.safetensors \
  --local-dir data/downloads/psi-model
mkdir -p data/checkpoints/pi05-suite/G1WholebodyBendPickMP-v0
cp data/downloads/psi-model/openpi-05/G1WholebodyBendPickMP-v0/20000/model.safetensors \
  data/checkpoints/pi05-suite/G1WholebodyBendPickMP-v0/
.venv-pi05/bin/hf download USC-PSI-Lab/psi-data --repo-type dataset \
  --revision c8258285d134d0da644fdbf1a304988398fc3819 \
  simple-eval/G1WholebodyBendPickMP-v0.zip simple/G1WholebodyBendPickMP-v0.zip \
  --local-dir data/downloads/psi-data
unzip -n data/downloads/psi-data/simple-eval/G1WholebodyBendPickMP-v0.zip \
  G1WholebodyBendPickMP-v0/dr-level-0/meta/episodes.jsonl -d data/pi05-suite/eval
unzip -n data/downloads/psi-data/simple/G1WholebodyBendPickMP-v0.zip \
  G1WholebodyBendPickMP-v0/meta/stats.json -d data/pi05-suite/train
```

Repeat with the other catalog entries; preparation currently expects all six tasks'
metadata. Verify archives against catalog hashes and weights against the pinned
Hugging Face LFS SHA256. The manifest generator independently rejects changed
metadata hashes, wrong dimensions, duplicate scenes or non-Level-0 inputs.
Newer training metadata is not interchangeable with this frozen baseline.

```bash
python scripts/reproduction/prepare_pi05_suite.py \
  --data-root data/pi05-suite --output experiments/reproduction/pi05/manifests
python scripts/reproduction/prepare_pi05_suite.py \
  --data-root data/pi05-suite --output experiments/reproduction/pi05/pilot-manifests --pilot
```

The pilot selects scene 0 with both model seeds per task. It is separate from the
formal score. Keep manifests unchanged once an experiment starts.

## Launch

```bash
source .env.isaac
python scripts/reproduction/launch_pi05_suite.py \
  --manifests experiments/reproduction/pi05/pilot-manifests \
  --data-root data/pi05-suite --weights-root data/checkpoints/pi05-suite \
  --output experiments/reproduction/pi05/pilot-01 \
  --isaac-session task --gpu 0
```

After checking pilots, use the formal `manifests` and a fresh output directory.
Existing outputs are never overwritten. The launcher uses `.venv-pi05` for one
model service per task and `.venv-isaac` for simulation. It resets model history
and sampling seed for every trial and stops only its own processes.

| Option | Behavior |
|---|---|
| `--tasks G1WholebodyBendPickMP-v0` | Run only selected environment names; omit for all six |
| `--isaac-session task` | Reuse Isaac within a task; rebuild each scene/controller |
| `--isaac-session trial` | Default: fresh Isaac process for each trial; slower diagnostic fallback |
| `--gpu N` | Bind one physical GPU and default service port to `8015 + N` |
| `--physics-profile upstream` | Default: upstream extra object force and per-action hand torque |
| `--physics-profile project` | Explicit legacy fork physics; report separately |
| `--sim-mode mujoco_isaac` | Default: live Isaac policy images, MuJoCo physics |
| `--sim-mode mujoco` | Diagnostic MuJoCo policy images in the same simulator environment |

For multiple GPUs, start independent launchers with disjoint tasks, different
`--gpu` values and separate outputs. Inference/rendering wall time can affect WBC
interpolation; this optimization does not guarantee identical trajectories.

## Read results

- `launch.json`: commit, commands, environment overrides, child status and timing.
- `<task>/evaluations/summary.json`: successes, completed count, per-trial outcome
  and unrun IDs. Infrastructure errors stop the batch; they are not policy failures.
- `<trial>/result.json`: task score, actual physics/hand settings, timing and provenance.
- `<trial>/head_stereo_left.mp4`: policy-camera video; AMO also saves an Isaac front view.
  `overview.mp4` is an additional MuJoCo view, not model input.
- Per-query PNGs and request/action records, executed commands, state arrays and
  telemetry allow inspection of failed attempts. AMO also saves hand diagnostics.

Report all requested trials, with scene and seed identities. Select success/failure
videos from that fixed set; if a task has no success or no failure, report that.
For BendPick, `lift_scores.paper_5cm` is a secondary metric; the primary executable
criterion stays at 9 cm. Normalization equivalence and Isaac 4.5 pixel parity remain
unresolved, so these runs are not a claim of reproducing the paper's success rate.
