"""Evaluate one pi0.5 task manifest with official agents in pure MuJoCo.

Each trial clears cached Task/Robot instances before constructing its environment
and controller; the external model server stays loaded. Videos cover every policy
step, including the official decoupled tail.
Physics seeds and model sampling seeds are separate. Existing outputs are never
overwritten, and an infrastructure/interface error stops the suite explicitly.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import time
import traceback

from pi05_common import decode_environment_config, validate_manifest, write_json


def effective_threshold(task, env_id):
    # These two official tasks ignore their success_criteria attribute entirely.
    hardcoded = {'simple/G1WholebodyTabletopGraspMP-v0': .9,
                 'simple/G1WholebodyXMoveBendPickTeleop-v0': .8}
    if env_id in hardcoded:
        return hardcoded[env_id]
    return getattr(task, 'success_criteria', task.metadata.get('success_criteria'))


def clear_registry_instances(*registries):
    # This evaluator owns the process. Keep registered classes and asset caches;
    # discard Task/Robot singletons so AMO/controller history cannot cross trials.
    for registry in registries:
        registry._instances.clear()


def rollout(step, raw, max_steps, trailing_steps):
    terminated = truncated = False
    termination = 'max_steps'
    for _ in range(max_steps):
        try:
            terminated, truncated = step('scored')
        except StopIteration:
            termination = 'agent_exhausted'
            break
        if terminated or truncated:
            termination = 'terminated' if terminated else 'time_limit'
            break
    # Success can occur on the last budgeted step (both flags True). Capture
    # it before the official video tail, which is never part of the score.
    success = bool(raw._success)
    if terminated and not truncated:
        for _ in range(trailing_steps):
            try:
                step('trailing')
            except StopIteration:
                break
        raw._success = success
    return success, termination


def run_trial(trial, output, host, port):
    import gymnasium as gym
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image
    import torch
    import simple.envs  # noqa: F401: registers the official tasks
    from simple.tasks.registry import TaskRegistry
    from simple.robots.registry import RobotRegistry

    output.mkdir()
    write_json(output / 'trial.json', trial)
    conf = decode_environment_config(trial['environment_config'])
    write_json(output / 'environment_config.json', conf)
    result = {'trial_id': trial['trial_id'], 'outcome': 'error', 'success': None,
              'scored_steps': 0, 'trailing_steps': 0, 'inference_count': 0,
              'sim_mode': 'mujoco', 'rollout_pacing': 'official_pure_mujoco_no_sleep',
              'video_fps': 50, 'controller_recreated': True}
    writers, chunks, requests, telemetry, stabilization = {}, [], [], [], []
    commands, states = [], []
    env = agent = overview = None
    start = time.monotonic()
    try:
        random.seed(trial['sim_seed'])
        np.random.seed(trial['sim_seed'])
        torch.manual_seed(trial['sim_seed'])
        kwargs = dict(sim_mode='mujoco', headless=True, render_hz=50)
        decoupled = trial['controller'] == 'decoupled'
        if decoupled:
            import decoupled_wbc
            import gear_sonic
            dependency_root = Path(__file__).resolve().parents[2] / 'third_party'
            sources = {module.__name__: str(Path(module.__file__).resolve())
                       for module in (decoupled_wbc, gear_sonic)}
            for name, path in sources.items():
                if Path(path).parent != (dependency_root / name).resolve():
                    raise RuntimeError(f'{name} must use the pinned standalone submodule, got {path}')
            result['dependency_sources'] = sources
            from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
            from simple.baselines.pi05_decoupled_wbc import Pi05DecoupledWbcAgent
            sonic_config = SimLoopConfig().load_wbc_yaml()
            sonic_config['ENV_NAME'] = 'simple'
            kwargs['sonic_config'] = sonic_config
            control_dt = 4 * sonic_config['SIMULATE_DT']
            agent_class = Pi05DecoupledWbcAgent
        else:
            from simple.baselines.pi05 import Pi05Agent
            from simple.envs.wrappers import StandStabilizationWrapper
            agent_class = Pi05Agent
        clear_registry_instances(TaskRegistry, RobotRegistry)
        result['task_robot_registry_instances_cleared'] = True
        env = gym.make(trial['env_id'], **kwargs)
        if not decoupled:
            env = StandStabilizationWrapper(env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=trial['max_steps'])
        raw = env.unwrapped
        threshold = effective_threshold(raw.task, trial['env_id'])
        if threshold is None or not math.isclose(threshold, trial['success_criteria']):
            raise ValueError(f'Task threshold {threshold} != manifest {trial["success_criteria"]}')
        result['effective_success_criteria'] = threshold
        result['task_success_attribute'] = getattr(raw.task, 'success_criteria', None)
        result['task_metadata'] = dict(raw.task.metadata)
        agent = agent_class(raw.task.robot, host, port,
                            **({'sonic_config': sonic_config} if decoupled else {}))
        # The synchronous official server cannot answer pings during cold GPU
        # inference. Keep a bounded heartbeat timeout without changing inference.
        agent.client._ws.ping_timeout = 600
        result['client_ping_timeout_seconds'] = 600
        metadata = agent.client.get_server_metadata()
        write_json(output / 'server_metadata.json', metadata)
        if metadata.get('action_dim') != 36 or metadata.get('action_horizon') != 30:
            raise ValueError(f'Unexpected model action contract: {metadata}')
        if trial.get('weight_task') and metadata.get('task') != trial['weight_task']:
            raise ValueError(f'Server task {metadata.get("task")} != {trial["weight_task"]}')
        infer = agent.client.infer

        def recorded_infer(request):
            request = dict(request)
            first = len(requests) == 0
            if bool(request.get('reset')) != first:
                raise ValueError('The agent must reset model history exactly once per trial')
            if first:
                request['model_seed'] = trial['model_seed']
            image = np.asarray(request['observation/image'])
            state = np.asarray(request['states'])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f'Invalid policy image: {image.shape} {image.dtype}')
            if state.shape != (28,) or not np.isfinite(state).all():
                raise ValueError(f'Invalid policy state: {state.shape}')
            Image.fromarray(image).save(output / f'input_{len(requests):03d}.png')
            record = {'state': state.tolist(), 'prompt': request['prompt'],
                      'reset': first, 'model_seed': request.get('model_seed'),
                      'image_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                      'scored_step': result['scored_steps'],
                      'trailing_step': result['trailing_steps']}
            requests.append(record)
            query_start = time.monotonic()
            try:
                response = infer(request)
                action = np.asarray(response['actions'])
                if action.dtype.kind in 'fiu':
                    chunks.append(action.copy())
                if action.shape != (30, 36) or not np.isfinite(action).all():
                    raise ValueError(f'Invalid model actions: {action.shape}')
                record['server_timing'] = response.get('server_timing')
                return response
            except Exception as error:
                record['error'] = repr(error)
                raise
            finally:
                record['wall_seconds'] = time.monotonic() - query_start

        agent.client.infer = recorded_infer
        obs, info = env.reset(seed=trial['sim_seed'], options={'state_dict': conf})
        sim = raw.mujoco
        result['reset_state_stage'] = 'before_stabilization' if decoupled else 'after_amo_warmup'
        result['initial_state_stage'] = 'after_stabilization'
        np.savez(output / 'reset_state.npz', qpos=sim.mjData.qpos, qvel=sim.mjData.qvel)
        agent.reset()  # Official decoupled reset precedes the stabilization ramp.
        if decoupled:
            agent._wbc_policy.lower_body_policy.use_policy_action = True
            while not raw.task.robot.stabilized and len(stabilization) < 300:
                tick = time.monotonic()
                action = agent.get_stabilize_action(obs)
                obs, reward, _, _, info = raw.step(action)  # outside TimeLimit
                raw.update_viewer()
                raw.update_reward()
                time.sleep(max(0, control_dt - (time.monotonic() - tick)))
                stabilization.append({'step': len(stabilization) + 1,
                                      'wall_seconds': time.monotonic() - tick,
                                      'reward': float(reward),
                                      'pelvis_z': float(sim.mjData.qpos[2])})
            agent._wbc_policy.lower_body_policy.gait_indices = torch.zeros((1), dtype=torch.float32)
            result['stabilization_steps'] = len(stabilization)
            result['robot_stabilized'] = bool(raw.task.robot.stabilized)
        else:
            result['stabilization_steps'] = 60
            result['robot_stabilized'] = None  # AMO uses a fixed warmup, no velocity gate.
        np.savez(output / 'initial_state.npz', qpos=sim.mjData.qpos, qvel=sim.mjData.qvel)
        result['instruction'] = raw.task.instruction

        # Teleop guarantees only the head camera. A separate MuJoCo camera
        # records an overview without changing the policy observation or model.
        sim.mjModel.vis.global_.offwidth = max(640, sim.mjModel.vis.global_.offwidth)
        sim.mjModel.vis.global_.offheight = max(480, sim.mjModel.vis.global_.offheight)
        overview = mujoco.Renderer(sim.mjModel, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 3.5, 135, -25
        for name in ('head_stereo_left', 'overview'):
            writers[name] = imageio.get_writer(output / f'{name}.mp4', fps=50,
                                               codec='libx264', macro_block_size=1)

        def record_frame():
            writers['head_stereo_left'].append_data(obs['head_stereo_left'])
            camera.lookat[:] = sim.mjData.qpos[:3] + np.array([.4, 0, -.1])
            overview.update_scene(sim.mjData, camera=camera)
            writers['overview'].append_data(overview.render())
            states.append((sim.mjData.qpos.copy(), sim.mjData.qvel.copy()))

        record_frame()

        def policy_step(phase):
            nonlocal obs, info
            tick = time.monotonic()
            action = agent.get_action(obs, info=info, instruction=raw.task.instruction)
            commands.append({'phase': phase, 'type': action.type,
                             'parameters': deepcopy(action.parameters)})
            obs, reward, terminated, truncated, info = env.step(action)
            # Never call compute_reward/check_success here: contact rewards in
            # official tasks are stateful, and env.step already invokes both.
            if not np.isfinite(sim.mjData.qpos).all() or not np.isfinite(sim.mjData.qvel).all():
                raise FloatingPointError('Nonfinite MuJoCo state')
            record_frame()
            telemetry.append({'step': len(telemetry) + 1, 'phase': phase,
                              'wall_seconds': time.monotonic() - tick,
                              'sim_time': float(sim.mjData.time), 'reward': float(reward),
                              'terminated': bool(terminated), 'truncated': bool(truncated),
                              'task_success': bool(raw._success),
                              'pelvis_pose': sim.mjData.qpos[:7].tolist(),
                              'objects': {key: value.tolist() for key, value in info.items()
                                          if isinstance(value, np.ndarray) and value.shape == (7,)}})
            result[f'{phase}_steps'] += 1
            if len(telemetry) == 1 or len(telemetry) % 50 == 0 or terminated or truncated:
                print(trial['trial_id'], phase, len(telemetry), float(reward), flush=True)
            return bool(terminated), bool(truncated)

        success, result['termination'] = rollout(policy_step, raw, trial['max_steps'],
                                                 trailing_steps=50 if decoupled else 0)
        result['success'] = success
        result['outcome'] = 'success' if success else 'failure'
    except BaseException as error:
        result.update(outcome='error', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        # Persist evidence even if inference, rendering, or a controller fails.
        cleanup_errors = []
        for resource in [*writers.values(), overview, getattr(getattr(agent, 'client', None), '_ws', None), env]:
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    cleanup_errors.append(repr(error))
        result.update(wall_seconds=time.monotonic() - start, inference_count=len(requests),
                      video_frames=len(states), cleanup_errors=cleanup_errors)
        np.savez_compressed(output / 'actions.npz', **{f'chunk_{i}': a for i, a in enumerate(chunks)})
        if states:
            np.savez_compressed(output / 'states.npz', qpos=np.stack([s[0] for s in states]),
                                qvel=np.stack([s[1] for s in states]))
        write_json(output / 'result.json', result)
        for name, data in [('requests', requests), ('commands', commands),
                           ('telemetry', telemetry), ('stabilization', stabilization)]:
            try:
                write_json(output / f'{name}.json', data)
            except Exception as error:
                cleanup_errors.append(f'{name}.json: {error!r}')
        write_json(output / 'result.json', result)
        if cleanup_errors and result['outcome'] != 'error':
            result.update(outcome='error', error='Failed to close output resources')
            write_json(output / 'result.json', result)
            raise RuntimeError(result['error'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8015)
    args = parser.parse_args()
    trials = validate_manifest(json.loads(args.manifest.read_text()))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'manifest.json', trials)
    summary = {'status': 'running', 'requested': len(trials),
               'results': [{'trial_id': t['trial_id'], 'outcome': 'unrun'} for t in trials]}
    write_json(args.output / 'summary.json', summary)
    try:
        for index, trial in enumerate(trials):
            summary['results'][index] = run_trial(trial, args.output / trial['trial_id'], args.host, args.port)
            write_json(args.output / 'summary.json', summary)
        summary['status'] = 'complete'
    except BaseException as error:
        summary['status'] = 'error'
        result_path = args.output / trial['trial_id'] / 'result.json'
        summary['results'][index] = (json.loads(result_path.read_text()) if result_path.exists()
                                     else {'trial_id': trial['trial_id'], 'outcome': 'error', 'error': repr(error)})
        raise
    finally:
        summary['completed'] = sum(r['outcome'] in ('success', 'failure') for r in summary['results'])
        summary['successes'] = sum(r['outcome'] == 'success' for r in summary['results'])
        summary['unrun'] = [r['trial_id'] for r in summary['results'] if r['outcome'] == 'unrun']
        write_json(args.output / 'summary.json', summary)


if __name__ == '__main__':
    main()
