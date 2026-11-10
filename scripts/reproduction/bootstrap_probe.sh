#!/usr/bin/env bash
# An isolated diagnostic environment, NOT the full SIMPLE installation.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
UV_BIN=${UV_BIN:-$(command -v uv || true)}
if [[ -z "$UV_BIN" ]]; then
  echo 'Set UV_BIN to an existing uv executable.' >&2
  exit 1
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.cache/uv-probe}"
export UV_PYTHON_INSTALL_DIR="$ROOT/.cache/python-probe"
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=120
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX
if [[ ! -f "$ROOT/.venv-probe/pyvenv.cfg" ]]; then
  "$UV_BIN" --no-config venv --python 3.10 "$ROOT/.venv-probe"
fi
"$ROOT/.venv-probe/bin/python" -c \
  'import sys; assert sys.version_info[:2] == (3, 10), "Expected Python 3.10"'
"$UV_BIN" --no-config pip install --python "$ROOT/.venv-probe/bin/python" \
  --index-url https://pypi.org/simple \
  -r "$ROOT/scripts/reproduction/probe-requirements.txt"
"$UV_BIN" --no-config pip install --python "$ROOT/.venv-probe/bin/python" \
  --index-url https://download.pytorch.org/whl/cpu 'torch==2.7.0+cpu'
echo 'Probe environment ready. Full Isaac Sim / cuRobo / policy evaluation is NOT installed.'
