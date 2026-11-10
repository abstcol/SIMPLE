"""Run the registered Franka task with bounded joint commands and record evidence.

This validates robot-task execution, not a trained policy or Isaac visual scores.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache/runtime'))
os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(ROOT / '.cache/torch-extensions'))
sys.path.insert(0, str(ROOT / 'src'))

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
import simple.envs
from simple.core.action import ActionCmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 8])
    args = parser.parse_args()
    if args.steps < 30:
        parser.error('Use at least 30 steps to exercise dynamics and control.')
    args.output.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available(), 'CUDA PyTorch required for cuRobo'
    result = {
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'curobo_commit': subprocess.check_output(['git', '-C', 'third_party/curobo', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'env_id': 'simple/FrankaTabletopGraspMP-v0', 'sim_mode': 'mujoco',
        'robot': 'franka_fr3', 'target': 'graspnet1b:63', 'scene': 'hssd:scene3',
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'scope': 'real robot model / task rollout; bounded joint commands, not a learned policy benchmark',
        'episodes': [],
    }
    started = time.monotonic()
    try:
        for seed in args.seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            episode_dir = args.output / f'seed-{seed}'
            episode_dir.mkdir()
            env = gym.make(result['env_id'], sim_mode='mujoco', headless=True,
                           webrtc=False, scene_uid=result['scene'],
                           target_object=result['target'], max_episode_steps=args.steps,
                           disable_env_checker=True)
            writer = None
            try:
                obs, info = env.reset(seed=seed)
                unwrapped = env.unwrapped
                assert unwrapped.isaac is None, 'Isaac must not be created'
                robot = unwrapped.task.robot
                # Execute GPU forward kinematics, not merely an import check.
                kin = robot.kin_model
                kin_q = torch.tensor([[robot.init_joint_states.get(name, 0.0)
                                       for name in kin.joint_names]], device='cuda', dtype=torch.float32)
                state = kin.get_state(kin_q)
                assert torch.isfinite(state.ee_position).all()
                torch.cuda.synchronize()
                print('cuRobo GPU FK:', state.ee_position.tolist(), flush=True)
                cameras = [k for k, v in obs.items() if isinstance(v, np.ndarray)
                           and v.ndim == 3 and v.shape[-1] == 3]
                assert cameras, 'No camera observations'
                camera = 'front_stereo_left' if 'front_stereo_left' in cameras else cameras[0]
                frame = np.asarray(obs[camera], dtype=np.uint8)
                assert frame.std() > 1, 'Empty camera image'
                Image.fromarray(frame).save(episode_dir / 'first.png')
                writer = imageio.get_writer(episode_dir / 'rollout.mp4', fps=30,
                                           codec='libx264', macro_block_size=1)
                writer.append_data(frame)
                joints = [np.array(obs['joint_qpos'], copy=True)]
                rewards, target_z = [], [float(info['target'][2])]
                initial = {k: float(v) for k, v in robot.init_joint_states.items()
                           if not k.startswith('finger')}
                terminated = truncated = False
                for step in range(args.steps):
                    targets = initial.copy()
                    targets['joint1'] += 0.18 * np.sin(2 * np.pi * (step + 1) / args.steps)
                    targets['joint7'] += 0.08 * np.sin(4 * np.pi * (step + 1) / args.steps)
                    action = ActionCmd('move_qpos_with_eef', target_qpos=targets,
                                       eef_state='open_eef' if step < args.steps // 2 else 'close_eef')
                    obs, reward, terminated, truncated, info = env.step(action)
                    assert np.isfinite(unwrapped.mujoco.mjData.qpos).all()
                    assert np.isfinite(unwrapped.mujoco.mjData.qvel).all()
                    assert np.isfinite(obs['joint_qpos']).all()
                    frame = np.asarray(obs[camera], dtype=np.uint8)
                    writer.append_data(frame)
                    joints.append(np.array(obs['joint_qpos'], copy=True))
                    rewards.append(float(reward))
                    target_z.append(float(info['target'][2]))
                    if terminated or truncated:
                        break
                movement = float(np.max(np.abs(np.asarray(joints) - joints[0])))
                assert movement > .01, f'Robot did not respond to control: {movement}'
                Image.fromarray(frame).save(episode_dir / 'last.png')
                np.savez_compressed(episode_dir / 'trajectory.npz',
                                    joint_qpos=joints, rewards=rewards, target_z=target_z)
                entry = {'seed': seed, 'steps': len(rewards), 'camera': camera,
                         'cameras': cameras, 'image_shape': list(frame.shape),
                         'max_joint_change': movement, 'terminated': bool(terminated),
                         'truncated': bool(truncated), 'max_reward': max(rewards),
                         'simulated_seconds': float(unwrapped.mujoco.mjData.time),
                         'robot_task_passed': True}
                result['episodes'].append(entry)
                print(json.dumps(entry), flush=True)
            finally:
                if writer is not None:
                    writer.close()
                env.close()
        result['passed'] = True
    except Exception as exc:
        result['passed'] = False
        result['error'] = repr(exc)
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
