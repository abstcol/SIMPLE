#!/usr/bin/env bash
# Build an isolated MuJoCo + CUDA/cuRobo runtime, without Isaac Sim.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
UV_BIN=${UV_BIN:-$(command -v uv || true)}
: "${UV_BIN:?Set UV_BIN to an existing uv executable}"
: "${MUJOCO_WHEELHOUSE:?Set MUJOCO_WHEELHOUSE to verified CPython 3.10 Linux wheels}"
: "${CUDA_ARCHIVES_DIR:?Set CUDA_ARCHIVES_DIR to verified NVIDIA 11.8 redistribution archives}"
export UV_CACHE_DIR=${UV_CACHE_DIR:-$ROOT/.cache/uv-mujoco}
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=120
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv-probe/bin/python}
if [[ ! -f .venv/pyvenv.cfg ]]; then
  "$UV_BIN" --no-config venv --python "$PYTHON_BIN" .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 10)'
"$UV_BIN" --no-config pip install --python .venv/bin/python \
  --no-index --find-links "$MUJOCO_WHEELHOUSE" \
  -r scripts/reproduction/probe-requirements.txt \
  -r scripts/reproduction/mujoco-requirements.txt 'torch==2.7.0+cu118'

export CUDA_HOME="$ROOT/.tools/cuda-11.8"
.venv/bin/python - "$CUDA_ARCHIVES_DIR" "$CUDA_HOME" <<'PY'
import hashlib, pathlib, subprocess, sys
archives, target = map(pathlib.Path, sys.argv[1:])
expected = {
    'cuda_nvcc-linux-x86_64-11.8.89-archive.tar.xz': '7ee8450dbcc16e9fe5d2a7b567d6dec220c5894a94ac6640459e06231e3b39a5',
    'cuda_cudart-linux-x86_64-11.8.89-archive.tar.xz': '56129e0c42df03ecb50a7bb23fc3285fa39af1a818f8826b183cf793529098bb',
    'cuda_cccl-linux-x86_64-11.8.89-archive.tar.xz': '99d77d9e4c75d5e4663e473577f1871e65bca4ea0b9023f544a3556f0c1776c7',
}
target.mkdir(parents=True, exist_ok=True)
for name, digest in expected.items():
    archive = archives / name
    if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f'CUDA archive hash mismatch: {name}')
    subprocess.run(['tar', '-xJf', str(archive), '--strip-components=1', '-C', str(target)], check=True)
if not (target / 'lib64').exists():
    (target / 'lib64').symlink_to('lib', target_is_directory=True)
PY
export PATH="$ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR="$ROOT/.cache/torch-extensions"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-$(.venv/bin/python -c \
  'import torch; assert torch.cuda.is_available(); print("%d.%d" % torch.cuda.get_device_capability())')}
export MAX_JOBS=${MAX_JOBS:-2}
test "$(git -C third_party/curobo rev-parse HEAD)" = 5810bc9cb69129a0e1a8aad47704bd6da35370c0
"$CUDA_HOME/bin/nvcc" --version
"$UV_BIN" --no-config pip install --python .venv/bin/python --no-deps \
  --no-build-isolation 'third_party/curobo[isaacsim]'
"$UV_BIN" --no-config pip check --python .venv/bin/python
.venv/bin/python - <<'PY'
import torch
import curobo.curobolib.lbfgs_step_cu
import curobo.curobolib.kinematics_fused_cu
import curobo.curobolib.line_search_cu
import curobo.curobolib.tensor_step_cu
import curobo.curobolib.geom_cu
assert torch.cuda.is_available()
assert torch.version.cuda == '11.8'
print('CUDA runtime:', torch.__version__, torch.cuda.get_device_name())
print('cuRobo CUDA extensions imported successfully')
PY
