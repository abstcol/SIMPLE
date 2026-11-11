"""Probe pretrained Psi0 with a rendered nominal task observation (no rollout)."""
from run_sonic_eval import ROOT  # initialize the isolated evaluator's import paths
import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from PIL import Image
from simple.agents.sonic_eval_agent import Psi0HttpPolicy
from simple.cli.render_decoupled_wbc import _load_episode_configs, _load_episodes
from simple.evals.sonic_wbc import _sonic_config


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--port", type=int, default=8014)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = ROOT / "data/simple-eval/G1WholebodyXMoveBendCarryBoxSonic-v0/dr-level-0"
    episodes = _load_episodes(str(dataset))
    config = _load_episode_configs(str(dataset))[args.episode]
    env = gym.make("simple/G1WholebodyXMoveBendCarryBoxSonic-v0", sim_mode="mujoco",
                   render_hz=50, physics_dt=0.005, headless=True, webrtc=False,
                   max_episode_steps=30000, sonic_config=_sonic_config())
    try:
        raw = env.unwrapped
        config["uid"] = raw.task.uid
        env.reset(options={"state_dict": config, "task_id": f"episode_{args.episode}", "dr_level": 0})
        row = episodes[args.episode].iloc[0]
        raw.mjData.qpos[:7] = np.asarray(row["observation.base_pose"])
        raw.mjData.qvel[:] = 0
        raw.mjData.qvel[:6] = np.asarray(row["observation.base_vel"])
        for name, value in zip(raw.task.robot.joint_names, row["observation.state"]):
            raw.mujoco.joints[name].qpos = value
        mujoco.mj_forward(raw.mjModel, raw.mjData)
        obs = raw._get_obs()
        image = np.asarray(obs["head_stereo_left"])
        state = np.asarray(obs["joint_qpos"], dtype=np.float32)
        assert raw.isaac is None and state.shape == (43,) and image.dtype == np.uint8
        Image.fromarray(image).save(args.output / "observation.png")
        np.save(args.output / "state.npy", state)
        client = Psi0HttpPolicy("127.0.0.1", args.port, timeout=180)
        client.reset(episodes[args.episode], args.episode)
        started = time.perf_counter()
        action = client.infer(image=image, state=state, instruction=raw.task.instruction)
        assert action.shape == (24, 78) and np.isfinite(action).all(), action.shape
        np.save(args.output / "action.npy", action)
        result = {"passed": True, "scope": "single inference; no action executed",
                  "episode": args.episode, "sim_mode": "mujoco", "instruction": raw.task.instruction,
                  "image_shape": list(image.shape), "state_shape": list(state.shape),
                  "action_shape": list(action.shape), "seconds": time.perf_counter()-started,
                  "server_info": client.info, "requests": client.request_records}
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k not in ("server_info", "requests")}))
    finally:
        env.close()


if __name__ == "__main__":
    main()
