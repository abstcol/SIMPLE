"""Render an AMO pi0.5 action trace through live MuJoCo and Isaac Sim.

This is recorded-command replay, not a fresh VLA inference or benchmark score.
"""
import argparse
import faulthandler
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

from pi05_common import config_hash, decode_environment_config, validate_manifest, write_json

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True, help='Saved formal trial directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, help='Limit the render smoke test; default replays all commands')
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error('--steps must be positive')
    trial = validate_manifest([json.loads((args.trial / 'trial.json').read_text())])[0]
    if trial['controller'] != 'amo':
        parser.error('This replay entry point supports the AMO tasks only')
    config = decode_environment_config(trial['environment_config'])
    if config_hash(config) != trial['environment_config_sha256']:
        raise ValueError('Source trial configuration hash mismatch')
    commands = json.loads((args.trial / 'commands.json').read_text())
    commands = commands[:args.steps] if args.steps else commands
    if not commands:
        raise ValueError('Source trial has no commands')
    args.output.mkdir(parents=True, exist_ok=False)
    for key, relative in {'XDG_CACHE_HOME': '.cache/isaac-runtime',
                          'XDG_DATA_HOME': '.cache/isaac-data',
                          'HF_HOME': '.cache/huggingface',
                          'CUDA_CACHE_PATH': '.cache/cuda-isaac',
                          'TORCH_EXTENSIONS_DIR': '.cache/torch-extensions-isaac',
                          'SIMPLE_ISAAC_PORTABLE_ROOT': '.cache/isaac-kit'}.items():
        os.environ.setdefault(key, str(ROOT / relative))
    os.environ.setdefault('MUJOCO_GL', 'egl')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(key, '4')
    sys.path.insert(0, str(ROOT / 'src'))

    import gymnasium as gym
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image
    import torch
    import simple.envs  # noqa: F401: registers task classes
    from simple.core.action import ActionCmd
    from simple.envs.wrappers import StandStabilizationWrapper

    random.seed(trial['sim_seed'])
    np.random.seed(trial['sim_seed'])
    torch.manual_seed(trial['sim_seed'])
    result = {'passed': False, 'execution': 'recorded_action_replay',
              'source_trial': trial['trial_id'], 'env_id': trial['env_id'],
              'sim_mode': 'mujoco_isaac', 'requested_steps': len(commands),
              'isaacsim_version': importlib.metadata.version('isaacsim'),
              'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name(),
              'inputs_sha256': {name: hashlib.sha256((args.trial / name).read_bytes()).hexdigest()
                                for name in ('trial.json', 'commands.json')}}
    env, writers, states = None, {}, []
    started = time.monotonic()
    faulthandler.dump_traceback_later(300, repeat=True)
    try:
        print('Creating G1 environment', flush=True)
        env = StandStabilizationWrapper(gym.make(trial['env_id'], sim_mode='mujoco_isaac',
                                                 headless=True, webrtc=False, render_hz=50))
        print('Resetting scene and warming up AMO', flush=True)
        obs, info = env.reset(seed=trial['sim_seed'], options={'state_dict': config})
        faulthandler.cancel_dump_traceback_later()
        raw = env.unwrapped
        assert raw.isaac is not None
        raw.isaac.world.stage.GetRootLayer().Export(str(args.output / 'scene_first.usda'))
        for name in ('head_stereo_left', 'front_stereo_left'):
            writers[name] = imageio.get_writer(args.output / f'{name}.mp4', fps=50,
                                               codec='libx264', macro_block_size=1)

        def record():
            for name, writer in writers.items():
                image = np.asarray(obs[name])
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.std() < 1:
                    raise ValueError(f'Empty or invalid Isaac image: {name}: {image.shape}')
                writer.append_data(image)
                if not states:
                    Image.fromarray(image).save(args.output / f'{name}_first.png')
            qpos, qvel = raw.mujoco.mjData.qpos, raw.mujoco.mjData.qvel
            if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
                raise FloatingPointError('Nonfinite MuJoCo state')
            states.append((qpos.copy(), qvel.copy()))

        record()
        for index, command in enumerate(commands, 1):
            obs, reward, terminated, truncated, info = env.step(ActionCmd(command['type'], **command['parameters']))
            record()
            if index % 50 == 0 or index == len(commands):
                print(f'Rendered {index}/{len(commands)}; reward={float(reward):.3f}', flush=True)
        for name in writers:
            Image.fromarray(obs[name]).save(args.output / f'{name}_last.png')
        raw.isaac.world.stage.GetRootLayer().Export(str(args.output / 'scene_last.usda'))
        result.update(passed=True, task_success=bool(raw._success), final_reward=float(reward))
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        for writer in writers.values():
            writer.close()
        if states:
            np.savez_compressed(args.output / 'states.npz', qpos=np.stack([s[0] for s in states]),
                                qvel=np.stack([s[1] for s in states]))
        result.update(video_frames=len(states), executed_steps=max(0, len(states) - 1),
                      wall_seconds=time.monotonic() - started)
        write_json(args.output / 'result.json', result)
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
