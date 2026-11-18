"""Run a standing two-arm lift with native SONIC and the frozen two-table scene."""

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import threading
import time

import gymnasium as gym
import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import zmq

import simple.envs  # noqa: F401 - register the task
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message, build_planner_message
from simple.core.action import ActionCmd
from pi05_common import decode_environment_config, validate_manifest, write_json
from run_reference_carry import SIDES, TASK, forces_on_target, wrist_goal

FPS = 50
IDLE, SLOW_WALK = 0, 1
NEUTRAL_HEAD = [0., 0., .40]
# SONIC's virtual wrist tracking points are offset from wrist_yaw_link.
WRIST_OFFSETS = ([.18, -.025, 0.], [.18, .025, 0.])


def planner_packet(base, wrists, forward, facing):
    """Convert world-frame wrist-link targets to the native three-point input."""
    if wrists is None:
        return build_planner_message(IDLE, (0., 0., 0.), facing)
    to_local = Rotation.from_quat(base[3:7], scalar_first=True).inv()
    positions, orientations = [], []
    for (position, rotation), offset in zip(wrists, WRIST_OFFSETS):
        positions.extend(to_local.apply(position + rotation.apply(offset) - base[:3]))
        orientations.extend((to_local * rotation).as_quat(scalar_first=True))
    return build_planner_message(
        SLOW_WALK if forward else IDLE, facing if forward else (0., 0., 0.), facing,
        speed=forward if forward else -1.,
        left_hand_position=[0.] * 7, right_hand_position=[0.] * 7,
        vr_3pt_position=positions + NEUTRAL_HEAD,
        vr_3pt_orientation=orientations + [1., 0., 0., 0.],
    )


def arm_contacts(name, side):
    return name.startswith(side + '_') and any(
        part in name for part in ('hand', 'wrist', 'forearm', 'elbow')
    )


def lift_pass(rows):
    window = [row for row in rows if row['phase'] == 'hold'][-25:]
    return len(window) == 25 and all(
        row['displacement_m'][2] >= .05
        and not any('table' in name for name in row['contacts_n'])
        and all(any(arm_contacts(name, side) for name in row['contacts_n']) for side in SIDES)
        and row['standing']
        for row in window
    )


def run(args, trial):
    deploy = args.deploy.resolve()
    binary = deploy / 'target/release/g1_deploy_onnx_ref'
    models = ['policy/release/model_decoder.onnx', 'policy/release/model_encoder.onnx',
              'planner/target_vel/V2/planner_sonic.onnx']
    for path in (binary, *(deploy / name for name in models)):
        if not path.is_file() or path.stat().st_size < 1024:
            raise FileNotFoundError(f'Missing SONIC executable or weights at {path}')

    random.seed(trial['sim_seed'])
    np.random.seed(trial['sim_seed'])
    config = SimLoopConfig().load_wbc_yaml()
    config.update(ENV_NAME='simple', USE_JOYSTICK=False)
    physics = {'mass': {'scale': args.mass, 'actor': 'target'}}
    env = gym.make(TASK, sim_mode='mujoco', headless=True, webrtc=False,
                   render_hz=FPS, physics_dt=config['SIMULATE_DT'], disable_env_checker=True,
                   sonic_config=config, physics=physics)
    raw = env.unwrapped
    raw.mujoco.upstream_object_gravity = True
    env.reset(seed=trial['sim_seed'],
              options={'state_dict': decode_environment_config(trial['environment_config'])})
    # The expert reads physics state and renders the third-person video after the rollout.
    # Skip the two unused wrist/head image readbacks in the real-time loop.
    camera_names = tuple(raw.mujoco.renderers)
    raw.mujoco.close()
    raw.mujoco.renderers = dict.fromkeys(camera_names)
    model, data, robot = raw.mjModel, raw.mjData, raw.task.robot
    bridge = UnitreeSdk2Bridge(config)

    def robot_state():
        return {name: value.copy() if isinstance(value, np.ndarray) else value
                for name, value in robot.prepare_obs().items()}

    proprio = robot_state()
    target = raw.mujoco.mj_objects['target'].id
    initial = data.body(target).xpos.copy()
    table = model.geom('table_geom')
    edge = float(data.geom(table.id).xpos[0] - table.size[0])
    top = float(data.geom(table.id).xpos[2] + table.size[2])
    wrists = tuple(model.body(f'{side}_wrist_yaw_link').id for side in SIDES)
    limits = robot.torque_limit[np.stack((robot.left_hand_index, robot.right_hand_index)) - 1]
    np.testing.assert_allclose(limits, [[2.45, .7, .7, .7, .7, .7, .7]] * 2)

    def current_wrists():
        return [(data.body(wrist).xpos.copy(),
                 Rotation.from_matrix(data.body(wrist).xmat.reshape(3, 3))) for wrist in wrists]

    def cradle(width, height, shift=(0., 0.)):
        x, y = initial[:2] + shift
        return [(pos, Rotation.from_matrix(mat)) for pos, mat in (
            wrist_goal([x, y + sign * width, top + height],
                       roll=-sign * 45, yaw=-sign * 12, reach=.075) for sign in (1, -1)
        )]

    def publish_state():
        while not heartbeat_stop.is_set():
            bridge.PublishLowState(proprio)
            heartbeat_stop.wait(.005)

    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.2, 132, -19
    camera.lookat[:] = [edge - .16, initial[1], .71]
    model.vis.global_.offwidth = max(640, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(480, model.vis.global_.offheight)
    rows, frames = [], []
    facing = (1., 0., 0.)
    heartbeat_stop = threading.Event()
    heartbeat = threading.Thread(target=publish_state, daemon=True)
    startup_support = 1.
    band_action = ActionCmd('elastic_band')
    robot.elastic_band.length = float(data.xpos[robot.band_attached_link, 2] - robot.elastic_band.point[2])

    def warmup_crane():
        robot.apply_action(band_action)
        data.xfrc_applied[robot.band_attached_link] *= startup_support

    with zmq.Context() as context, context.socket(zmq.PUB) as pub, (
        args.output / 'controller.log'
    ).open('w') as log:
        pub.bind(f'tcp://127.0.0.1:{args.port}')
        command = [str(binary), 'lo', models[0], 'reference/example/',
                   '--encoder-file', models[1], '--planner-file', models[2],
                   '--obs-config', 'policy/release/observation_config.yaml',
                   '--default-motion', 'squat_001__A359', '--input-type', 'zmq_manager',
                   '--zmq-host', '127.0.0.1', '--zmq-port', str(args.port),
                   '--output-type', 'all', '--zmq-out-port', str(args.port + 1),
                   '--logs-dir', str((args.output / 'controller').resolve()),
                   '--disable-crc-check']
        controller = subprocess.Popen(command, cwd=deploy, stdout=log, stderr=subprocess.STDOUT,
                                      env={**os.environ, 'SONIC_FORCE_UNITREE_DDS': '1'})
        heartbeat.start()
        try:
            nominal_wrists = last_goals = current_wrists()
            deadline = time.monotonic() + args.startup_timeout
            while max(bridge.low_cmd.motor_cmd[i].kp for i in range(29)) <= 5:
                if controller.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError('SONIC did not engage. See controller.log.')
                pub.send(build_command_message(start=True, stop=False, planner=True))
                pub.send(planner_packet(data.qpos[:7], None, 0., facing))
                time.sleep(.04)

            def tick(phase, goals, forward=0.):
                nonlocal proprio, last_goals
                if goals is not None:
                    last_goals = goals
                start = time.monotonic()
                if controller.poll() is not None:
                    raise RuntimeError('SONIC exited. See controller.log.')
                pub.send(planner_packet(data.qpos[:7], goals, forward, facing))
                action = ActionCmd('wbc_torque', low_cmd=bridge.low_cmd, use_sensor=bridge.use_sensor,
                                   left_hand_cmd=bridge.left_hand_cmd, right_hand_cmd=bridge.right_hand_cmd)
                _, _, terminated, _, _ = env.step(action)
                proprio = robot_state()
                pos = data.body(target).xpos.copy()
                contacts = forces_on_target(model, data, target)
                rows.append(dict(phase=phase, t_s=float(data.time), object_xyz_m=pos.tolist(),
                                 displacement_m=(pos - initial).tolist(),
                                 pelvis_xyz_m=data.qpos[:3].tolist(),
                                 pelvis_quat_wxyz=data.qpos[3:7].tolist(),
                                 navigation_speed_mps=forward,
                                 standing=bool(data.qpos[2] > .54),
                                 task_success=bool(terminated),
                                 contacts_n={key: force.tolist() for key, force in contacts.items()},
                                 wrists_xyz_m=[p.tolist() for p, _ in current_wrists()],
                                 wrist_targets_m=[p.tolist() for p, _ in (goals or [])]))
                if len(rows) % 2 == 0:
                    frames.append(data.qpos.copy())
                time.sleep(max(0., 1 / FPS - (time.monotonic() - start)))

            def move(phase, steps, goals, follow_box=False):
                origins = last_goals
                for step in range(steps):
                    ratio = (step + 1) / steps
                    fraction = ratio * ratio * (3 - 2 * ratio)
                    shift = np.zeros(3)
                    if follow_box:
                        shift[:2] = np.clip(data.body(target).xpos[:2] - initial[:2], -.04, .07)
                    targets = [(p0 + fraction * (p1 + shift - p0),
                                Rotation.from_rotvec(fraction * (r1 * r0.inv()).as_rotvec()) * r0)
                               for (p0, r0), (p1, r1) in zip(origins, goals)]
                    tick(phase, targets)
                    if not rows[-1]['standing']:
                        break
                print(phase, rows[-1]['displacement_m'], list(rows[-1]['contacts_n']), flush=True)
                return rows[-1]['standing']

            # Use the existing sim crane during SONIC's three-second motor ramp.
            # Remove its wrench before standing manipulation.
            raw.substep_callback = warmup_crane
            warmup_crane()
            for _ in range(185):
                tick('startup', None)
            last_goals = current_wrists()
            for step in range(60):
                startup_support = 1. - (step + 1) / 60
                tick('release_crane', last_goals)
            raw.substep_callback = None
            data.xfrc_applied[robot.band_attached_link] = 0.
            move('settle', 50, last_goals)
            if rows[-1]['standing'] and move('approach', 105, cradle(.145, .076)):
                if move('cup', 105, cradle(.045, .046)):
                    move('preload', 65, cradle(.033, .046), follow_box=True)
            if rows[-1]['standing'] and rows[-1]['phase'] == 'preload':
                lifted = cradle(.033, .175)
                if move('lift', 145, lifted, follow_box=True):
                    move('hold', 70, lifted, follow_box=True)
        finally:
            heartbeat_stop.set()
            heartbeat.join()
            controller.terminate()
            try:
                controller.wait(timeout=5)
            except subprocess.TimeoutExpired:
                controller.kill()
                controller.wait()
            write_json(args.output / 'trajectory.json', rows)
            result = dict(trial_id=trial['trial_id'], mass_scale=args.mass,
                          target_mass_kg=float(model.body_mass[target]), initial_xyz_m=initial.tolist(),
                          max_lift_m=max((row['displacement_m'][2] for row in rows), default=0.),
                          lift_and_hold_pass=lift_pass(rows), task_success=any(r['task_success'] for r in rows),
                          steps=len(rows), hand_torque_limits_nm=limits.tolist())
            write_json(args.output / 'result.json', result)
            if frames:
                with mujoco.Renderer(model, height=480, width=640) as renderer, imageio.get_writer(
                    args.output / 'third_person.mp4', fps=FPS // 2
                ) as video:
                    for qpos in frames:
                        data.qpos[:] = qpos
                        mujoco.mj_forward(model, data)
                        renderer.update_scene(data, camera=camera)
                        video.append_data(renderer.render())
            raw.mujoco.renderers.clear()
            env.close()
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--trial', required=True)
    parser.add_argument('--mass', type=float, choices=[1., 10.], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=5596)
    parser.add_argument('--startup-timeout', type=float, default=600., help='Seconds allowed for cold engine startup.')
    parser.add_argument('--deploy', type=Path, default=Path('third_party/GR00T-WholeBodyControl/gear_sonic_deploy'))
    args = parser.parse_args()
    trials = validate_manifest(json.loads(args.manifest.read_text()))
    trial = next((row for row in trials if row['trial_id'] == args.trial and row['env_id'] == TASK), None)
    if trial is None:
        parser.error('Select a frozen two-table carry trial.')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'source_trial.json', trial)
    run(args, trial)


if __name__ == '__main__':
    main()
