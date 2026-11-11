#!/usr/bin/env bash
# Source from a task shell. No global configuration or HOME redirection.
SIMPLE_REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export SONIC_TOOLS_DIR="$SIMPLE_REPRO_ROOT/.tools/sonic"
export SONIC_ENV_FILE="$SONIC_TOOLS_DIR/sonic_env.sh"
export EVAL_WBC_PYTHON="$SIMPLE_REPRO_ROOT/.venv-sonic/bin/python"
export EVAL_WBC_WRAPPER="$SIMPLE_REPRO_ROOT/scripts/reproduction/run_sonic_eval.py"
export EVAL_WBC_LOG_DIR="${EVAL_WBC_LOG_DIR:-$SIMPLE_REPRO_ROOT/experiments/reproduction/psi0-sonic/logs}"
export CUDA_HOME="$SIMPLE_REPRO_ROOT/.tools/cuda-11.8"
export CUDAToolkit_ROOT="$CUDA_HOME"
export LD_LIBRARY_PATH="$SONIC_TOOLS_DIR/compiler/lib:$SONIC_TOOLS_DIR/cyclonedds/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME="$SIMPLE_REPRO_ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$SIMPLE_REPRO_ROOT/.cache/runtime"
export CUDA_CACHE_PATH="$SIMPLE_REPRO_ROOT/.cache/cuda"
export TORCH_HOME="$SIMPLE_REPRO_ROOT/.cache/torch"
export TORCH_EXTENSIONS_DIR="$SIMPLE_REPRO_ROOT/.cache/torch-extensions"
export UV_CACHE_DIR="$SIMPLE_REPRO_ROOT/.cache/uv"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl
export HEADLESS=1
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
