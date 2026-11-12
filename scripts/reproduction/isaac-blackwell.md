# Isaac rendering on Blackwell

Use a separate project-local environment for RTX 50-series GPUs. SIMPLE's pinned
Isaac Sim 4.5 renderer predates Blackwell support. This deployment uses Isaac Sim
5.1, Python 3.11 and PyTorch CUDA 12.8 on Ubuntu 22.04; it is a compatibility path,
not a claim of pixel equivalence to the published 4.5 benchmark.

Keep the existing `data/` layout and AMO assets. Build the pinned cuRobo submodule
for the new Python/CUDA environment; do not copy an older compiled environment.
The base project's Python requirement and default lockfile remain unchanged.

From the project root, with `uv` and CUDA 12.8's compiler available:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv-isaac" UV_LINK_MODE=copy
export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python-isaac"
export UV_PYTHON_BIN_DIR="$PWD/.tools/python-isaac/bin"
uv --no-config python install 3.11
uv --no-config venv --python 3.11 .venv-isaac
uv --no-config pip install --python .venv-isaac/bin/python \
  'isaacsim[all,extscache]==5.1.0' \
  'torch==2.7.0+cu128' 'torchvision==0.22.0+cu128' \
  -r scripts/reproduction/isaac-blackwell-requirements.txt \
  --extra-index-url https://pypi.nvidia.com \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match
git submodule update --init third_party/curobo
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=2
uv --no-config pip install --python .venv-isaac/bin/python \
  --no-deps --no-build-isolation './third_party/curobo[isaacsim]'
uv --no-config pip check --python .venv-isaac/bin/python
```

Use a fresh environment; omit the `venv` creation command when resuming an existing
installation. The dependency list preserves Isaac's numpy/matplotlib/trimesh pins.
Isaac also needs the official `vMaterials_2.zip` resource under `data/`; SIMPLE
downloads and extracts it on first use if absent. Retain the existing asset
version markers when migrating `data/` between servers.

If Kit reports duplicate Vulkan ICDs for one GPU, select the verified NVIDIA
manifest for this process, for example
`export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json`. Do not remove the
server's driver files. See the [Vulkan loader documentation](https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderDriverInterface.md).

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
The first/last USD root-layer snapshots retain scene transforms for inspection.
Math libraries default to four threads, matching the existing suite launcher.
Cold startup can spend several minutes compiling materials and ray-tracing
pipelines; a stack dump after five minutes diagnoses initialization without
terminating it. Later runs reuse the project-local caches.

Isaac 5.1's reference helper automatically reconciles asset units and up axes.
SIMPLE already interprets mesh coordinates and explicitly rotates its HSSD rooms.
Applying both conversions shrank the tested cracker box to scale 0.0001 and
rotated the room twice. The renderer now adds plain USD references to preserve
SIMPLE's coordinate conventions. Source assets and MuJoCo physics are unchanged.

`SIMPLE_ISAAC_PORTABLE_ROOT` passes Kit's documented `--portable-root` option so
logs, settings and extension caches stay inside the project. The replay script
defaults it to `.cache/isaac-kit`; existing SIMPLE entry points are unchanged when
the variable is unset. Python/XDG/Hugging Face/Torch caches also stay project-local.

References: [Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html),
[Python installation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html),
[Blackwell and 4.5](https://github.com/isaac-sim/IsaacSim/issues/671),
[Kit portable mode](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/107.0.3/guide/configuring.html#portable-mode).

Validation on 2026-11-12 used Ubuntu 22.04, RTX 5090, driver 580.105.08,
CUDA compiler 12.8.93, Python 3.11.15, Isaac Sim 5.1.0.0, PyTorch 2.7.0+cu128,
and NumPy 1.26.0. The pinned cuRobo revision built for `sm_120`; package dependency
checks, CUDA matrix multiplication, and SIMPLE imports passed.

At commit `eb13c2619430a4a7e780f9fbe03d0bf1b276def2`, replaying the saved
`G1WholebodyBendPickMP-v0` trial `episode_001_seed_0` completed 202 commands after
the 60-step stand warmup. Both cameras produced 203 RGB frames at 640×360/50 fps,
the task success flag was true, and all physics states were finite. Head/front
images showed the correctly oriented room and the cracker box in the grasp.
The reference-loading fix produced exactly identical qpos/qvel arrays to the
preceding replay with incorrect visual transforms. The warm run took 131.75 s.
See [PR #8](https://github.com/abstcol/SIMPLE/pull/8) for final-head validation.

This validates one AMO task and rendering compatibility. Fresh VLA inference,
SONIC tasks, and equivalence to official Isaac 4.5 images are not established.
The supplied HSSD scene still contains unresolved upstream `props/*.usd`
references in its log; do not infer complete asset or pixel parity from this check.
