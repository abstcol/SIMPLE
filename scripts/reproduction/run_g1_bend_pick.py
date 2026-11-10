"""Execute the original G1 BendPick motion-planning task and retain all attempts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache/runtime'))
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(ROOT / '.cache/torch-extensions'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, str(ROOT / 'src'))

import gymnasium as gym
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
import transforms3d as t3d
from PIL import Image
import simple.envs
from simple.agents.mp import MotionPlannerAgent
from simple.mp.curobo import CuRoboPlanner


def run_episode(seed, output, max_steps):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output.mkdir()
    result = {'seed': seed, 'passed': False, 'phases': []}
    env = overview = agent = None
    writers, samples, actions = {}, [], []
    start = time.monotonic()
    try:
        env = gym.make('simple/G1WholebodyBendPickMP-v0', sim_mode='mujoco',
                       headless=True, webrtc=False, render_hz=50,
                       max_episode_steps=max_steps)
        obs, info = env.reset(seed=seed)
        raw = env.unwrapped
        task, sim = raw.task, raw.mujoco
        assert raw.isaac is None
        assert task.robot.uid == 'g1_wholebody'
        assert sim.mjModel.opt.gravity[2] < -9
        result['joint_qpos_addresses'] = {sim.mjModel.joint(i).name: int(sim.mjModel.jnt_qposadr[i]) for i in range(sim.mjModel.njnt)}
        initial_z = float(info['target'][2])
        initial_pelvis = float(sim.mjData.qpos[2])
        result.update(initial_target_z=initial_z, initial_pelvis_z=initial_pelvis,
                      target=task.target.asset.uid, instruction=task.instruction)
        (output / 'initial-layout.json').write_text(json.dumps(task.state_dict(), default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x), indent=2))
        # Enlarge only the offscreen rendering buffer for the evidence camera.
        sim.mjModel.vis.global_.offwidth = max(960, sim.mjModel.vis.global_.offwidth)
        sim.mjModel.vis.global_.offheight = max(720, sim.mjModel.vis.global_.offheight)
        overview = mujoco.Renderer(sim.mjModel, height=720, width=960)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [-0.3, 0.0, 0.4]
        camera.distance, camera.azimuth, camera.elevation = 2.7, 135, -15
        for name in ('overview', 'front_stereo_left'):
            writers[name] = imageio.get_writer(output / f'{name}.mp4', fps=50,
                                               codec='libx264', macro_block_size=1)
        target_body = sim.mj_objects['target'].id
        target_geoms = set(np.flatnonzero(sim.mjModel.geom_bodyid == target_body))
        hand_bodies = {i for i in range(sim.mjModel.nbody)
                       if 'right_hand' in (mujoco.mj_id2name(sim.mjModel, mujoco.mjtObj.mjOBJ_BODY, i) or '')}
        hand_geoms = set(np.flatnonzero(np.isin(sim.mjModel.geom_bodyid, list(hand_bodies))))

        def record(obs, info, reward=0., terminated=False, stage='reset'):
            assert np.isfinite(sim.mjData.qpos).all() and np.isfinite(sim.mjData.qvel).all()
            contact = any((c.geom1 in target_geoms and c.geom2 in hand_geoms) or
                          (c.geom2 in target_geoms and c.geom1 in hand_geoms)
                          for c in sim.mjData.contact[:sim.mjData.ncon])
            rpy = t3d.euler.quat2euler(sim.mjData.qpos[3:7])
            sample = dict(step=len(actions), stage=stage, time=float(sim.mjData.time),
                          target_z=float(info['target'][2]), pelvis_z=float(sim.mjData.qpos[2]),
                          pelvis_roll=float(rpy[0]), pelvis_pitch=float(rpy[1]),
                          hand_contact=bool(contact), reward=float(reward),
                          task_success=bool(terminated), qpos=sim.mjData.qpos.copy().tolist())
            samples.append(sample)
            overview.update_scene(sim.mjData, camera=camera)
            frame = overview.render()
            writers['overview'].append_data(frame)
            writers['front_stereo_left'].append_data(obs['front_stereo_left'])
            if len(samples) == 1 or stage != samples[-2]['stage']:
                Image.fromarray(frame).save(output / f'{len(actions):04d}-{stage}.png')
            # Abort genuinely fallen episodes, while retaining their evidence.
            assert abs(rpy[0]) < 1.0 and abs(rpy[1]) < 1.2, 'Robot fell/tilted excessively'
            return sample

        record(obs, info)
        print(f'Seed {seed}: reset complete; initializing cuRobo planner', flush=True)
        planner = CuRoboPlanner(robot=task.robot, plan_dt=.01, plan_batch_size=1,
                               easy_motion_gen=False, ignore_target_collisions=False)
        agent = MotionPlannerAgent(task, planner, plan_batch_size=1)
        phase = 0
        while True:
            phase += 1
            print(f'Seed {seed}: synthesizing phase {phase}', flush=True)
            state = agent.synthesize()
            result['phases'].append({'phase': phase, 'state': state, 'queued_actions': len(agent)})
            assert state is not False, f'Planning phase {phase} failed'
            while len(agent):
                action = agent.get_action(obs, info)
                actions.append({'type': action.type, 'parameters': action.parameters})
                obs, reward, terminated, truncated, info = env.step(action)
                if action.type == 'loco_command':
                    stage = action.parameters.get('motion_type', 'loco')
                    if stage == 'height_adjust':
                        stage = 'bend' if action.parameters['command'][3] < -.1 else 'lift'
                else:
                    stage = 'grasp'
                sample = record(obs, info, reward, terminated, stage)
                if len(actions) % 50 == 0:
                    print(f'Seed {seed}: step {len(actions)}, {stage}, reward={reward:.3f}, '
                          f'lift={sample["target_z"]-initial_z:.3f}m, contact={sample["hand_contact"]}', flush=True)
                # Record the original success signal, but finish the original queued
                # stand/lift sequence to check that success was not a transient toss.
                assert not truncated, 'Step limit reached before completing the task sequence'
            if state != 'phase_break':
                break
        tail = samples[-50:]
        bent = [s['pelvis_z'] for s in samples if s['stage'] == 'bend']
        result.update(
            steps=len(actions), original_success=any(s['task_success'] for s in samples),
            max_reward=max(s['reward'] for s in samples),
            max_lift=max(s['target_z'] for s in samples)-initial_z,
            final_lift=samples[-1]['target_z']-initial_z,
            tail_min_lift=min(s['target_z'] for s in tail)-initial_z,
            pelvis_lowering=initial_pelvis-min(bent) if bent else 0.,
            final_pelvis_z=samples[-1]['pelvis_z'],
            hand_contact_frames=sum(s['hand_contact'] for s in samples),
            tail_contact_frames=sum(s['hand_contact'] for s in tail),
        )
        assert result['original_success'], 'Original task success condition was not met'
        assert result['pelvis_lowering'] > .15, 'No meaningful body lowering'
        assert result['hand_contact_frames'] > 5, 'No sustained hand/object contact'
        assert result['tail_min_lift'] > .08, 'Object did not remain lifted during final stand'
        assert result['tail_contact_frames'] > 25, 'Object was not held during final stand'
        result['passed'] = True
    except Exception as exc:
        result['error'] = repr(exc)
        traceback.print_exc()
    finally:
        if agent is not None:
            agent.reset()
        for writer in writers.values():
            writer.close()
        if overview is not None:
            overview.close()
        if env is not None:
            env.close()
        result['elapsed_seconds'] = time.monotonic()-start
        (output / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
        (output / 'telemetry.json').write_text(json.dumps(samples)+'\n')
        (output / 'actions.json').write_text(json.dumps(actions, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))+'\n')
        print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[7])
    p.add_argument('--max-steps', type=int, default=1500)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available()
    result = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  env_id='simple/G1WholebodyBendPickMP-v0', sim_mode='mujoco',
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  task_modified=False, attempts=[])
    result['amo_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (ROOT/'src/simple/robots/policy').glob('*.pt')}
    for seed in args.seeds:
        result['attempts'].append(run_episode(seed, args.output/f'seed-{seed}', args.max_steps))
        result['passed'] = any(x['passed'] for x in result['attempts'])
        (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        if result['passed']:
            break
    if not result['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
