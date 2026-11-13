# SIMPLE humanoid evaluation

This fork of [SIMPLE](https://github.com/physical-superintelligence-lab/SIMPLE)
provides a focused workflow for evaluating six released, task-specific π0.5
policies on the Unitree G1. MuJoCo advances physics, Isaac Sim renders policy
observations, and the original AMO or GR00T Balance/Walk controller executes actions.
No policy training is required.

**Start here:** [install](scripts/reproduction/isaac-blackwell.md) →
[prepare inputs and run](scripts/reproduction/pi05-suite.md) →
[understand upstream differences](scripts/reproduction/compatibility.md).

The tested deployment uses Ubuntu 22.04, RTX 5090, Python 3.11, Isaac Sim 5.1
and MuJoCo 3.3.6. This is a compatibility fork, not a claim that we have reproduced
the paper's success rates or Isaac 4.5 pixels. The evaluation defaults to upstream
object-force and hand-feedback behavior. Earlier fork physics is an explicit option.

## Run an evaluation

After setup and input preparation, run from the repository root:

```bash
source .env.isaac
python scripts/reproduction/prepare_pi05_suite.py \
  --data-root data/pi05-suite --output experiments/reproduction/pi05/manifests
python scripts/reproduction/launch_pi05_suite.py \
  --manifests experiments/reproduction/pi05/manifests \
  --data-root data/pi05-suite --weights-root data/checkpoints/pi05-suite \
  --output experiments/reproduction/pi05/run-01 \
  --isaac-session task --gpu 0
```

This evaluates 10 published Level 0 scenes × model seeds 0 and 1 per task.
Use `--tasks G1WholebodyBendPickMP-v0` for one task, or prepare manifests with
`--pilot` for two trials per task. Every run requires a new output directory.
`--isaac-session task` avoids restarting Isaac between trials of the same task.

Results contain per-task summaries, per-trial scores, images, actions, states and
videos. Policy videos use Isaac; the overview video is a MuJoCo diagnostic view.
See the [evaluation guide](scripts/reproduction/pi05-suite.md) for the six tasks,
outputs, explicit compatibility options and interpretation of success rates.

## Repository layout

| Path | Purpose |
|---|---|
| `src/simple/` | Simulation, robots, controllers, tasks and policy interfaces |
| `scripts/reproduction/` | Input preparation, batch launcher, evaluator, model server and tests |
| `third_party/` | Pinned upstream dependencies; retain their licenses |
| `data/` | Local assets, weights and task metadata; large files are not committed |
| `experiments/reproduction/` | Local run outputs; not committed |
| `docs/source/` | Upstream architecture and API documentation |
| `docs/projects/simple-phys-project.md` | Proposed physics-robustness research plan; not a completed benchmark |

Early component probes, arm demos, floor-texture experiments and standalone action
replay scripts have been retired. Their source remains in Git history. Existing
weights, environments and experiment directories are not removed by this cleanup.
For the broader upstream training, teleoperation and data-generation workflows,
see the [official documentation](https://psi-lab.ai/SIMPLE/docs/).

## Development

Keep source changes separate from machine-local configuration (`.env.isaac`),
data and results. Do not mix trials from different commits or physics profiles.
The [reproduction tools](scripts/reproduction/README.md) list responsibilities and
test commands. Describe behavioral differences and validation in each PR.

## Attribution and license

SIMPLE and the released policies are the work of the original SIMPLE/Psi0 authors.
This fork retains the [MIT license](license.md); external models, datasets and
submodules retain their own terms. Cite the original work when using it.

## Citation

> Please also consider citing `Psi-0` if you use its training code.

```
@article{wei2026simple,
  title={SIMPLE: Simulation-Based Policy Learning and Evaluation for Humanoid Loco-manipulation},
  author={Wei, Songlin and Ni, Zhenhao and Liu, Jie and Zhao, Zhenyu and Ye, Junjie and Jing, Hongyi and Xia, Junkai and Liu, Xiawei and Leong, Michael and Heng, Liang and Huang, Di and Wang, Yue},
  journal={arXiv preprint arXiv:2606.08278},
  year={2026}
}
```

```
@article{wei2026psi0,
  title={{$\Psi_0$}: An Open Foundation Model Towards Universal Humanoid Loco-Manipulation},
  author={Wei, Songlin and Jing, Hongyi and Li, Boqian and Zhao, Zhenyu and Mao, Jiageng and Ni, Zhenhao and He, Sicheng and Liu, Jie and Liu, Xiawei and Kang, Kaidi and others},
  journal={arXiv preprint arXiv:2603.12263},
  year={2026}
}
```

