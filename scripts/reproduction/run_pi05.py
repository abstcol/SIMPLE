"""Paired MuJoCo trials using the official pi0.5 agent and BendPick/AMO task."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import gymnasium as gym
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image
import simple.envs
from simple.baselines.pi05 import Pi05Agent
from simple.envs.wrappers import StandStabilizationWrapper

from pi05_common import decode_environment_config


def trial(args, episode, variant):
    output = args.output / f'episode_{episode["episode_index"]}' / variant
    output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    conf = decode_environment_config(episode['environment_config'])
    (output / 'environment_config.json').write_text(json.dumps(conf, indent=2))
    env = gym.wrappers.TimeLimit(StandStabilizationWrapper(gym.make(
        'simple/G1WholebodyBendPickMP-v0', sim_mode='mujoco', headless=True,
        webrtc=False, render_hz=50)), max_episode_steps=args.max_steps)
    writers, chunks, requests, telemetry = {}, [], [], []
    result = {'variant': variant, 'seed': args.seed, 'success': False}
    started = time.monotonic()
    try:
        obs, info = env.reset(seed=args.seed, options={'state_dict': conf})
        raw = env.unwrapped
        sim = raw.mujoco
        np.savez(output / 'initial_state.npz', qpos=sim.mjData.qpos, qvel=sim.mjData.qvel)
        if variant == 'plain_floor':
            # Only the ground's visual material changes; physics and geometry do not.
            material = sim.mjModel.material('groundplane').id
            sim.mjModel.mat_texid[material, :] = -1
            sim.mjModel.mat_rgba[material] = [.25, .25, .25, 1.]
            obs = raw.refresh_obs()
        agent = Pi05Agent(raw.task.robot, '127.0.0.1', args.port)
        infer = agent.client.infer

        def recorded_infer(request):
            image = request['observation/image']
            Image.fromarray(image).save(output / f'input_{len(chunks):03d}.png')
            response = infer(request)
            action = np.asarray(response['actions'])
            assert action.shape == (30, 36) and np.isfinite(action).all()
            chunks.append(action.copy())
            requests.append({'state': np.asarray(request['states']).tolist(),
                             'prompt': request['prompt'], 'reset': request['reset'],
                             'image_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                             'server_timing': response.get('server_timing')})
            return response

        agent.client.infer = recorded_infer
        agent.reset()
        for name in ('head_stereo_left', 'front_stereo_left'):
            writers[name] = imageio.get_writer(output / f'{name}.mp4', fps=50,
                                               codec='libx264', macro_block_size=1)
            writers[name].append_data(obs[name])
        for step in range(args.max_steps):
            action = agent.get_action(obs, info=info, instruction=raw.task.instruction)
            obs, reward, terminated, truncated, info = env.step(action)
            telemetry.append({'step': step + 1, 'reward': float(reward),
                              'target_z': float(info['target'][2]),
                              'pelvis_z': float(sim.mjData.qpos[2])})
            for name, writer in writers.items():
                writer.append_data(obs[name])
            if step % 50 == 0:
                print(variant, episode['episode_index'], telemetry[-1], flush=True)
            if terminated or truncated:
                break
        result.update(success=bool(raw._success), steps=len(telemetry),
                      termination='success' if raw._success else 'budget_or_termination',
                      inference_count=len(chunks))
    except Exception as error:
        result['error'] = repr(error)
        raise
    finally:
        for writer in writers.values():
            writer.close()
        env.close()
        np.savez_compressed(output / 'actions.npz', **{f'chunk_{i}': a for i, a in enumerate(chunks)})
        for name, data in [('requests', requests), ('telemetry', telemetry),
                           ('result', dict(result, wall_seconds=time.monotonic() - started))]:
            (output / f'{name}.json').write_text(json.dumps(data, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--episode-start', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=400)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--port', type=int, default=8015)
    parser.add_argument('--variants', nargs='+', choices=['original', 'plain_floor'],
                        default=['original', 'plain_floor'])
    args = parser.parse_args()
    episodes = [json.loads(line) for line in (args.data_dir / 'meta/episodes.jsonl').read_text().splitlines()]
    for episode in episodes[args.episode_start:args.episode_start + args.episodes]:
        for variant in args.variants:
            trial(args, episode, variant)


if __name__ == '__main__':
    main()
