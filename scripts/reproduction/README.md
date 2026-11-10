# SIMPLE reproduction diagnostics

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
