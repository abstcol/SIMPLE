# Setup: Isaac rendering on RTX 50-series

Run commands from the repository root in a persistent workspace. The tested stack
is Ubuntu 22.04, Python 3.11, Isaac Sim 5.1, PyTorch 2.7.0/CUDA 12.8 and MuJoCo 3.3.6.
Use an RTX GPU, a compatible NVIDIA driver and the CUDA 12.8 compiler for cuRobo.
The base project's Python requirement and lockfile remain the upstream 4.5 route;
do not run `uv sync` over these separate environments.

Existing installations can keep their `.env.isaac`, environments, data and caches.
The following is a manual installation recipe; a clean-machine installation has
not yet been validated as a one-command installer.

## Source and submodules

```bash
git clone https://github.com/abstcol/SIMPLE.git
cd SIMPLE
git submodule update --init third_party/curobo third_party/AMO \
  third_party/decoupled_wbc third_party/gear_sonic \
  third_party/GR00T-WholeBodyControl third_party/unitree_sdk2_python
git lfs pull --include='src/simple/robots/policy/*.pt'
mkdir -p .tools
git clone https://github.com/physical-superintelligence-lab/Psi0.git .tools/psi0-source/Psi0
git -C .tools/psi0-source/Psi0 checkout 805fcda9c9654e272be2a6673fe5290944fcfb04
```

Keep the pinned submodule revisions. The standalone `decoupled_wbc` and
`gear_sonic` packages must take precedence over GR00T's bundled versions; the
launcher supplies this import order. The Balance/Walk ONNX files and AMO `.pt`
files must be real weights, not Git LFS pointer text.

## Two environments

Use an existing `uv` installation. Keep downloaded Python, caches and build outputs
inside the project. On a fresh installation:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv" UV_LINK_MODE=copy
export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
export UV_PYTHON_BIN_DIR="$PWD/.tools/python/bin"
uv --no-config python install 3.11
uv --no-config venv --python 3.11 .venv-isaac
uv --no-config venv --python 3.11 .venv-pi05
uv --no-config pip install --python .venv-isaac/bin/python \
  'isaacsim[all,extscache]==5.1.0' \
  'torch==2.7.0+cu128' 'torchvision==0.22.0+cu128' \
  -r scripts/reproduction/isaac-blackwell-requirements.txt \
  --extra-index-url https://pypi.nvidia.com \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=2
uv --no-config pip install --python .venv-isaac/bin/python \
  --no-deps --no-build-isolation './third_party/curobo[isaacsim]'
uv --no-config pip install --python .venv-pi05/bin/python \
  -r scripts/reproduction/pi05-requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match
cp -r .tools/psi0-source/Psi0/src/openpi/models_pytorch/transformers_replace/* \
  .venv-pi05/lib/python3.11/site-packages/transformers/
```

The replacement above is supplied by the official policy implementation. Apply it
only in the model environment. Isaac keeps `websockets==12.0`; the model uses 15.0.1.
JAX is CPU-only for input transforms; PyTorch runs model inference on CUDA.

The Unitree simulator types import CycloneDDS. If a wheel is unavailable, build
its C library locally before installing the Python binding:

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
uv --no-config pip check --python .venv-isaac/bin/python
uv --no-config pip check --python .venv-pi05/bin/python
```

## Local activation and assets

Create a private `.env.isaac` with these settings, sourcing it from the project root:

```bash
export VIRTUAL_ENV="$PWD/.venv-isaac"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD/third_party:$PWD/third_party/GR00T-WholeBodyControl:$PWD/third_party/unitree_sdk2_python:$PWD/.tools/psi0-source/Psi0/src/openpi/openpi-client/src"
export CYCLONEDDS_HOME="$PWD/.tools/cyclonedds"
export CUDA_HOME=/usr/local/cuda
export MUJOCO_GL=egl
export HF_HOME="$PWD/.cache/huggingface"
export OPENPI_DATA_HOME="$PWD/.cache/openpi"
# Set these after accepting NVIDIA's terms; allow root only in a root-run container.
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_ALLOW_ROOT=1
```

The launcher supplies GPU placement, thread limits and project-local cache paths.
If Vulkan reports duplicate ICDs, select the verified NVIDIA manifest for this
process, e.g. `VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json`; do not delete
server driver files.

SIMPLE loads its scene/robot/object archives from the official
[asset repository](https://huggingface.co/datasets/USC-PSI-Lab/SIMPLE) into `data/`.
Preserve `data/_versions.json` and the installed `.zip.version` markers when
migrating an existing project. Isaac also requires `vMaterials_2.zip`; first use
may download assets and compile materials. For the earlier pinned G1 bundle,
`prepare_g1_assets.py --inputs <directory>` verifies and installs the six files
listed in that script; other task assets still come from SIMPLE's normal loader.

Cache the official tokenizer before the launcher's offline model service starts:

```bash
PYTHONPATH="$PWD/.tools/psi0-source/Psi0/src" JAX_PLATFORMS=cpu \
  OPENPI_DATA_HOME="$PWD/.cache/openpi" \
  .venv-pi05/bin/python -c 'from openpi.models.tokenizer import PaligemmaTokenizer; PaligemmaTokenizer(250)'
```

Continue with [input preparation and evaluation](pi05-suite.md). See
[compatibility](compatibility.md) for asset/rendering limitations and the exact
meaning of the physics options. Driver and installation requirements:
[Isaac 5.1 documentation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).
