"""Evaluate one pi0.5 task manifest with official agents and live policy images.

Each trial clears cached Task/Robot instances before constructing its environment
and controller; the external model server stays loaded. Videos cover every policy
step, including the official decoupled tail.
Physics seeds and model sampling seeds are separate. Existing outputs are never
overwritten, and an infrastructure/interface error stops the suite explicitly.
"""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

from pi05_common import decode_environment_config, sha256, validate_manifest, write_json


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


def task_isaac_session():
    from simple.envs.base_dual_env import isaac_session
    return isaac_session()


def run_trial(trial, output, host, port, sim_mode='mujoco', isaac_session='trial'):
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
              'sim_mode': sim_mode, 'rollout_pacing': 'no_added_rollout_sleep',
              'execution': 'closed_loop_inference',
              'policy_image_source': 'isaac' if 'isaac' in sim_mode else 'mujoco',
              'overview_image_source': 'mujoco',
              'video_fps': 50, 'controller_recreated': True,
              'process_id': os.getpid(), 'isaac_session': isaac_session}
    writers, chunks, requests, telemetry, stabilization = {}, [], [], [], []
    commands, states = [], []
    env = agent = overview = None
    start = time.monotonic()
    try:
        random.seed(trial['sim_seed'])
        np.random.seed(trial['sim_seed'])
        torch.manual_seed(trial['sim_seed'])
        kwargs = dict(sim_mode=sim_mode, headless=True, webrtc=False, render_hz=50)
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
        if 'isaac' in sim_mode:
            import importlib.metadata
            if raw.isaac is None or not raw.render_isaac:
                raise RuntimeError('Isaac policy rendering is not enabled')
            if decoupled:
                # Isaac mirrors the completed control step; retain every MuJoCo
                # physics substep without rendering its intermediate states.
                raw.sync_isaac_every_substep = False
            result['isaac_render_cadence'] = 'every_control_step'
            result['isaac_sync_every_substep'] = getattr(raw, 'sync_isaac_every_substep', None)
            result['isaacsim_version'] = importlib.metadata.version('isaacsim')
        result.update(torch_version=torch.__version__, mujoco_version=mujoco.__version__)
        threshold = effective_threshold(raw.task, trial['env_id'])
        if threshold is None or not math.isclose(threshold, trial['success_criteria']):
            raise ValueError(f'Task threshold {threshold} != manifest {trial["success_criteria"]}')
        result['effective_success_criteria'] = threshold
        result['task_success_attribute'] = getattr(raw.task, 'success_criteria', None)
        result['task_metadata'] = dict(raw.task.metadata)
        # Do not hold a websocket open during potentially long USD/material loading.
        result['setup_seconds'] = time.monotonic() - start
        reset_start = time.monotonic()
        obs, info = env.reset(seed=trial['sim_seed'], options={'state_dict': conf})
        result['reset_seconds'] = time.monotonic() - reset_start
        if raw.isaac is not None:
            result['isaac_app_id'] = id(raw.simulation_app)
            result['isaac_stage'] = raw.isaac.world.stage.GetRootLayer().identifier
        sim = raw.mujoco
        result['reset_state_stage'] = 'before_stabilization' if decoupled else 'after_amo_warmup'
        result['initial_state_stage'] = 'after_stabilization'
        np.savez(output / 'reset_state.npz', qpos=sim.mjData.qpos, qvel=sim.mjData.qvel)
        agent_start = time.monotonic()
        agent = agent_class(raw.task.robot, host, port,
                            **({'sonic_config': sonic_config} if decoupled else {}))
        # The synchronous official server cannot answer pings during cold GPU
        # inference. Keep a bounded heartbeat timeout without changing inference.
        if hasattr(agent.client._ws, 'ping_timeout'):
            agent.client._ws.ping_timeout = 600
        # Isaac's websockets 12 sync client has no automatic heartbeat.
        result['client_ping_timeout_seconds'] = getattr(agent.client._ws, 'ping_timeout', None)
        metadata = agent.client.get_server_metadata()
        write_json(output / 'server_metadata.json', metadata)
        if metadata.get('action_dim') != 36 or metadata.get('action_horizon') != 30:
            raise ValueError(f'Unexpected model action contract: {metadata}')
        if trial.get('weight_task') and metadata.get('task') != trial['weight_task']:
            raise ValueError(f'Server task {metadata.get("task")} != {trial["weight_task"]}')
        expected_stats = trial.get('metadata_source', {}).get('train_stats_sha256')
        if expected_stats and metadata.get('stats_sha256') != expected_stats:
            raise ValueError('Model server normalization differs from the frozen manifest')
        camera = raw.task.layout.cameras['head_stereo_left']
        result['policy_camera'] = {
            'name': 'head_stereo_left', 'resolution': list(camera.resolution),
            'intrinsics': [camera.fx, camera.fy, camera.cx, camera.cy],
            'position': camera.pose.position, 'quaternion': camera.pose.quaternion,
        }
        if raw.isaac is not None:
            actual = raw.isaac.cameras['head_stereo_left'].get_intrinsics_matrix()
            expected = np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]])
            if not np.allclose(actual, expected, atol=1e-3):
                raise ValueError(f'Isaac camera intrinsics differ from task: {actual} != {expected}')
            result['policy_camera']['isaac_intrinsics'] = actual
        infer = agent.client.infer

        def recorded_infer(request):
            request = dict(request)
            first = len(requests) == 0
            if bool(request.get('reset')) != first:
                raise ValueError('The agent must reset model history exactly once per trial')
            if first:
                request['model_seed'] = trial['model_seed']
            image = np.asarray(request['observation/image'])
            if not np.array_equal(image, obs['head_stereo_left']):
                raise ValueError('Policy request does not match the live head camera')
            state = np.asarray(request['states'])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f'Invalid policy image: {image.shape} {image.dtype}')
            width, height = result['policy_camera']['resolution']
            if image.shape != (height, width, 3):
                raise ValueError(f'Policy camera resolution mismatch: {image.shape}')
            if state.shape != (28,) or not np.isfinite(state).all():
                raise ValueError(f'Invalid policy state: {state.shape}')
            Image.fromarray(image).save(output / f'input_{len(requests):03d}.png')
            record = {'state': state.tolist(), 'prompt': request['prompt'],
                      'reset': first, 'model_seed': request.get('model_seed'),
                      'image_sha256': sha256(image.tobytes()),
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
        result['agent_setup_seconds'] = time.monotonic() - agent_start
        agent.reset()  # Official decoupled reset precedes the stabilization ramp.
        stabilize_start = time.monotonic()
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
        result['stabilization_seconds'] = time.monotonic() - stabilize_start
        np.savez(output / 'initial_state.npz', qpos=sim.mjData.qpos, qvel=sim.mjData.qvel)
        result['instruction'] = raw.task.instruction

        # Teleop guarantees only the head camera. A separate MuJoCo camera
        # records an overview without changing the policy observation or model.
        sim.mjModel.vis.global_.offwidth = max(640, sim.mjModel.vis.global_.offwidth)
        sim.mjModel.vis.global_.offheight = max(480, sim.mjModel.vis.global_.offheight)
        overview = mujoco.Renderer(sim.mjModel, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 3.5, 135, -25
        names = ['head_stereo_left', 'overview']
        if 'isaac' in sim_mode and 'front_stereo_left' in obs:
            names.append('front_stereo_left')
        for name in names:
            writers[name] = imageio.get_writer(output / f'{name}.mp4', fps=50,
                                               codec='libx264', macro_block_size=1)

        def record_frame():
            writers['head_stereo_left'].append_data(obs['head_stereo_left'])
            if 'front_stereo_left' in writers:
                writers['front_stereo_left'].append_data(obs['front_stereo_left'])
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
        if trial['env_id'] == 'simple/G1WholebodyBendPickMP-v0':
            # Keep the executable upstream 9-cm stopping rule. Score the paper's
            # 5-cm description separately on the same trajectory, without early stopping.
            maximum = max(row['reward'] for row in telemetry if row['phase'] == 'scored')
            result['lift_scores'] = {'code_9cm': success, 'paper_5cm': maximum > 0.5,
                                     'max_clipped_lift_m': maximum * 0.1}
    except BaseException as error:
        result.update(outcome='error', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        # Persist evidence even if inference, rendering, or a controller fails.
        cleanup_errors = []
        for resource in [*writers.values(), overview, getattr(getattr(agent, 'client', None), '_ws', None)]:
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
        # Save evidence before releasing the stage. Task sessions defer Kit
        # shutdown until the suite summary is saved; trial mode closes it here.
        if env is not None:
            env.close()
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
    parser.add_argument('--sim-mode', choices=('mujoco', 'mujoco_isaac'), default='mujoco')
    parser.add_argument('--isaac-session', choices=('trial', 'task'), default='trial',
                        help='Keep Kit alive per task while rebuilding every trial stage/controller')
    parser.add_argument('--trial-index', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.isaac_session == 'task' and (args.sim_mode != 'mujoco_isaac' or args.trial_index is not None):
        parser.error('--isaac-session task requires a complete Isaac task manifest')
    trials = validate_manifest(json.loads(args.manifest.read_text()))
    if args.trial_index is not None:
        if not 0 <= args.trial_index < len(trials):
            parser.error('trial-index is outside the manifest')
        trial = trials[args.trial_index]
        run_trial(trial, args.output / trial['trial_id'], args.host, args.port, args.sim_mode)
        return
    with task_isaac_session() if args.isaac_session == 'task' else nullcontext():
        evaluate_manifest(args, trials)


def evaluate_manifest(args, trials):
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'manifest.json', trials)
    summary = {'status': 'running', 'requested': len(trials),
               'results': [{'trial_id': t['trial_id'], 'outcome': 'unrun'} for t in trials]}
    write_json(args.output / 'summary.json', summary)
    def interrupted(signum, _frame):
        raise SystemExit(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for index, trial in enumerate(trials):
            if args.sim_mode == 'mujoco_isaac' and args.isaac_session == 'trial':
                # SimulationApp cannot be restarted safely inside one process.
                command = [sys.executable, '-u', str(Path(__file__).resolve()),
                           '--manifest', str(args.manifest), '--output', str(args.output),
                           '--host', args.host, '--port', str(args.port),
                           '--sim-mode', args.sim_mode, '--trial-index', str(index)]
                child = subprocess.Popen(command)
                try:
                    if child.wait():
                        raise RuntimeError(f'Isaac trial process exited: {child.returncode}')
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                result = json.loads((args.output / trial['trial_id'] / 'result.json').read_text())
                if result['outcome'] not in ('success', 'failure'):
                    raise RuntimeError(f'Isaac trial did not complete: {result}')
                summary['results'][index] = result
            else:
                summary['results'][index] = run_trial(trial, args.output / trial['trial_id'], args.host,
                                                      args.port, args.sim_mode, args.isaac_session)
            write_json(args.output / 'summary.json', summary)
            if args.isaac_session == 'task':
                gc.collect()
        summary['status'] = 'complete'
    except BaseException as error:
        summary['status'] = 'error'
        result_path = args.output / trial['trial_id'] / 'result.json'
        summary['results'][index] = (json.loads(result_path.read_text()) if result_path.exists()
                                     else {'trial_id': trial['trial_id'], 'outcome': 'error', 'error': repr(error)})
        # A crash during Kit shutdown is infrastructure failure even if the
        # worker saved a policy result before exiting abnormally.
        summary['results'][index].update(outcome='error', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        summary['completed'] = sum(r['outcome'] in ('success', 'failure') for r in summary['results'])
        summary['successes'] = sum(r['outcome'] == 'success' for r in summary['results'])
        summary['unrun'] = [r['trial_id'] for r in summary['results'] if r['outcome'] == 'unrun']
        write_json(args.output / 'summary.json', summary)


if __name__ == '__main__':
    main()
