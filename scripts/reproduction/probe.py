"""Bounded diagnostics. A MuJoCo smoke pass is not a SIMPLE benchmark pass."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def run(argv, out: Path, name: str, timeout=120, env=None):
    try:
        proc = subprocess.run(argv, cwd=ROOT, env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=timeout)
        code, output = proc.returncode, proc.stdout
    except subprocess.TimeoutExpired as exc:
        code = 124
        output = exc.stdout or b""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        output += "\nTIMEOUT\n"
    except OSError as exc:
        code, output = 127, str(exc)
    (out / f"{name}.log").write_text(output)
    print(f"{name}: exit={code}", flush=True)
    return {"command": argv, "exit_code": code, "log": f"{name}.log"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uv", default="uv")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier attempt.
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.update(PYTHONPATH=str(ROOT / "src"), MUJOCO_GL="egl",
               HF_HOME=str(ROOT / ".cache/huggingface-probe"),
               XDG_CACHE_HOME=str(ROOT / ".cache/probe"),
               UV_CACHE_DIR=str(ROOT / ".cache/uv-probe"),
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    result = {"time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "scope": "diagnostics only; not policy or task reproduction",
              "platform": platform.platform(), "libc": platform.libc_ver(),
              "python": sys.version, "checks": {}}
    checks = result["checks"]
    checks["commit"] = run(["git", "rev-parse", "HEAD"], out, "commit")
    checks["gpu"] = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                         "--format=csv,noheader"], out, "gpu")
    checks["packages"] = run([args.uv, "--no-config", "pip", "freeze", "--python", sys.executable],
                             out, "packages", env=env)
    checks["isaac_wheel"] = run([
        args.uv, "--no-config", "pip", "install", "--dry-run", "--no-deps",
        "--python", sys.executable, "--index-url", "https://pypi.nvidia.com",
        "isaacsim==4.5.0"], out, "isaac_wheel", timeout=180, env=env)
    checks["version"] = run([sys.executable, "-c",
                             "import simple; print(simple.__version__)"], out, "version", env=env)
    checks["environment_registry"] = run([sys.executable, "scripts/list_env.py"],
                                         out, "environment_registry", env=env)
    checks["mujoco_smoke"] = run([sys.executable,
                                 "scripts/reproduction/mujoco_smoke.py", str(out)],
                                 out, "mujoco_smoke", env=env)
    # Keep the official task and implementation intact. Bound time and steps.
    # This is an attempt using the PARTIAL probe environment; missing cuRobo is
    # an expected limitation of this environment, not proof that MuJoCo cannot run.
    checks["official_mujoco_task"] = run([
        sys.executable, "scripts/test_env.py", "--sim-mode", "mujoco",
        "--headless", "--no-webrtc", "--max-episode-steps", "30",
        "--save-dir", str(out / "task")], out, "official_mujoco_task", env=env)
    result["full_benchmark_reproduced"] = False
    result["diagnostic_checks_passed"] = all(checks[k]["exit_code"] == 0 for k in
        ("version", "environment_registry", "mujoco_smoke"))
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    # 2 means an incomplete reproduction, even when component checks pass.
    return 2


if __name__ == "__main__":
    sys.exit(main())
