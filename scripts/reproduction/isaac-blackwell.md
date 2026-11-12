# Isaac rendering on Blackwell

Use a separate project-local environment for RTX 50-series GPUs. SIMPLE's pinned
Isaac Sim 4.5 renderer predates Blackwell support. This deployment uses Isaac Sim
5.1, Python 3.11 and PyTorch CUDA 12.8 on Ubuntu 22.04; it is a compatibility path,
not a claim of pixel equivalence to the published 4.5 benchmark.

Keep the existing `data/` layout and AMO assets. Build the pinned cuRobo submodule
for the new Python/CUDA environment; do not copy an older compiled environment.
The base project's Python requirement and default lockfile remain unchanged.

The rendering check consumes one saved AMO trial from the six-task evaluation:

```bash
export OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1
export PYTHONPATH="$PWD/src"
.venv-isaac/bin/python scripts/reproduction/render_g1_isaac.py \
  --trial experiments/reproduction/isaac5090/source-trial \
  --output experiments/reproduction/isaac5090/render-01
```

It recreates the initial scene, performs the existing AMO stand warmup, and sends
the recorded commands through live MuJoCo. Isaac Sim renders the synchronized
state into the head/front camera videos. It never assigns saved qpos during the
rollout. `--steps 2` is available for a short startup check. Existing output folders
are refused. Results include input hashes, runtime versions, finite physics states,
full videos, and first/last images. This is action replay, not new VLA inference.

`SIMPLE_ISAAC_PORTABLE_ROOT` passes Kit's documented `--portable-root` option so
logs, settings and extension caches stay inside the project. The replay script
defaults it to `.cache/isaac-kit`; existing SIMPLE entry points are unchanged when
the variable is unset. Python/XDG/Hugging Face/Torch caches also stay project-local.

References: [Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html),
[Python installation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html),
[Blackwell and 4.5](https://github.com/isaac-sim/IsaacSim/issues/671),
[Kit portable mode](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/107.0.3/guide/configuring.html#portable-mode).

Server installation and runtime validation are pending; record the actual package
versions and results after verification.
