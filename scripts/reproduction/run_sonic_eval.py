"""Run official SONIC evaluator with a separate dependency overlay."""
from pathlib import Path
import os
import runpy
import site
import sys

ROOT = Path(__file__).resolve().parents[2]
if sys.version_info[:2] != (3, 10):
    raise RuntimeError("SONIC evaluator requires Python 3.10, matching the MuJoCo base")
base = ROOT / ".venv/lib/python3.10/site-packages"
if not base.is_dir():
    raise RuntimeError(f"Missing validated MuJoCo base: {base}")
# Process base editable-install .pth files, keeping overlay packages first.
site.addsitedir(str(base))
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "third_party"),
                str(ROOT / "third_party/unitree_sdk2_python")]
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache/runtime"))
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ROOT / ".cache/torch-extensions"))
runpy.run_module("simple.cli.eval_wbc", run_name="__main__")
