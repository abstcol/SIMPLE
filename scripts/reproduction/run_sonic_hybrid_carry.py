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
IDLE = 0
NEUTRAL_HEAD = [0., 0., .40]
# SONIC's virtual wrist tracking points are offset from wrist_yaw_link.
WRIST_OFFSETS = ([.18, -.025, 0.], [.18, .025, 0.])
# Native planner upper-body payload order, indexing the G1 motor array.
UPPER_BODY = [12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]


def planner_packet(base, wrists, hands, stowed=None):
    """Convert world-frame wrist-link targets to the native three-point input."""
    fingers = dict(left_hand_position=hands[0], right_hand_position=hands[1])
    if wrists is None:
        return build_planner_message(IDLE, (0., 0., 0.), (1., 0., 0.),
                                     upper_body_position=stowed, **fingers)
    to_local = Rotation.from_quat(base[3:7], scalar_first=True).inv()
    positions, orientations = [], []
    for (position, rotation), offset in zip(wrists, WRIST_OFFSETS):
        point = to_local.apply(position + rotation.apply(offset) - base[:3])
        # Keep the absolute target from running away when the base steps back.
        point[0] = min(point[0], .40)
        positions.extend(point)
        orientations.extend((to_local * rotation).as_quat(scalar_first=True))
    return build_planner_message(
        IDLE, (0., 0., 0.), (1., 0., 0.),
        **fingers,
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

    closed_hands = []
    for side in SIDES:
        cfg = getattr(robot.controller_cfg, f'{side}_eef_cfg')
        joints = getattr(robot, f'{side}_hand_index')
        targets = dict(zip(cfg.joint_names, cfg.close_qpos))
        values = [targets[model.joint(int(i)).name] for i in joints]
        data.qpos[joints + robot.qpos_offset - 1] = values
        closed_hands.append(values)
    open_hands = [[0.] * 7] * 2

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
    contact_impulse = {}
    pelvis = model.body('pelvis').id
    robot_bodies = {model.body(i).name for i in range(model.nbody)
                    if model.body_rootid[i] == pelvis}
    gates = {'startup': False, 'wrists': False}
    heartbeat_stop = threading.Event()
    heartbeat = threading.Thread(target=publish_state, daemon=True)
    startup_support = 1.
    band_action = ActionCmd('elastic_band')

    def warmup_crane():
        robot.apply_action(band_action)
        data.xfrc_applied[robot.band_attached_link] *= startup_support

    action = ActionCmd('wbc_torque', use_sensor=bridge.use_sensor)

    def pace_substep():
        nonlocal next_substep
        bridge.PublishLowState(robot.prepare_obs())
        for name, force in forces_on_target(model, data, target).items():
            contact_impulse[name] = contact_impulse.get(name, np.zeros(3)) + force * raw.mujoco.physics_dt
        if startup_support:
            warmup_crane()
        next_substep += raw.mujoco.physics_dt
        time.sleep(max(0., next_substep - time.monotonic()))
        action.parameters.update(low_cmd=bridge.low_cmd, left_hand_cmd=bridge.left_hand_cmd,
                                 right_hand_cmd=bridge.right_hand_cmd)

    next_substep = 0.

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
            # Start above the box so SONIC's motor ramp can run without
            # contacting it. Use the same lowering sequence as the flat-ground probe.
            robot.elastic_band.point[2] = 1.20
            data.qpos[2] = 1.15
            data.qvel[:6] = 0.
            mujoco.mj_forward(model, data)
            proprio = robot_state()
            deadline = time.monotonic() + args.startup_timeout
            while max(bridge.low_cmd.motor_cmd[i].kp for i in range(29)) <= 5:
                if controller.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError('SONIC did not engage. See controller.log.')
                pub.send(build_command_message(start=False, stop=False, planner=True))
                time.sleep(.04)

            # Finish the stock motor ramp before running physics near the box.
            time.sleep(3.1)
            motor_pose = np.array([bridge.low_cmd.motor_cmd[i].q for i in range(29)])
            # Keep the Dex3 pads behind the box while lowering and releasing.
            motor_pose[[15, 22]] = .43
            motor_pose[[16, 23]] = [.20, -.20]
            motor_pose[[18, 25]] = .40
            stowed = motor_pose[UPPER_BODY].tolist()
            data.qpos[robot.body_joint_index + robot.qpos_offset - 1] = motor_pose
            mujoco.mj_forward(model, data)
            proprio = robot_state()
            for _ in range(5):
                pub.send(build_command_message(start=True, stop=False, planner=True))
                pub.send(planner_packet(data.qpos[:7], None, closed_hands, stowed))
                time.sleep(.02)
            heartbeat_stop.set()
            heartbeat.join()
            last_goals = current_wrists()

            def tick(phase, goals):
                nonlocal next_substep, last_goals
                if goals is not None:
                    last_goals = goals
                next_substep = time.monotonic()
                if controller.poll() is not None:
                    raise RuntimeError('SONIC exited. See controller.log.')
                hands = closed_hands if phase in ('startup', 'release_crane', 'settle', 'clear', 'ready') else open_hands
                pub.send(planner_packet(data.qpos[:7], goals, hands, stowed))
                contact_impulse.clear()
                action.parameters.update(low_cmd=bridge.low_cmd, left_hand_cmd=bridge.left_hand_cmd,
                                         right_hand_cmd=bridge.right_hand_cmd)
                _, _, terminated, _, _ = env.step(action)
                pos = data.body(target).xpos.copy()
                contacts = forces_on_target(model, data, target)
                rows.append(dict(phase=phase, t_s=float(data.time), object_xyz_m=pos.tolist(),
                                 displacement_m=(pos - initial).tolist(),
                                 pelvis_xyz_m=data.qpos[:3].tolist(),
                                 pelvis_quat_wxyz=data.qpos[3:7].tolist(),
                                 standing=bool(data.qpos[2] > .54),
                                 task_success=bool(terminated),
                                 contacts_n={key: force.tolist() for key, force in contacts.items()},
                                 contact_impulse_ns={key: impulse.tolist() for key, impulse in contact_impulse.items()},
                                 robot_contacts=sorted(robot_bodies.intersection(contact_impulse)),
                                 wrists_xyz_m=[p.tolist() for p, _ in current_wrists()],
                                 wrist_targets_m=[p.tolist() for p, _ in (goals or [])]))
                if len(rows) % 2 == 0:
                    frames.append(data.qpos.copy())

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

            def quiet(window):
                poses = np.array([row['pelvis_xyz_m'] for row in window])
                return (len(poses) >= 20 and poses[:, 2].min() >= .70
                        and np.linalg.norm(np.ptp(poses, axis=0)) <= .035)

            def clean_scene():
                return all(not row['robot_contacts'] and np.linalg.norm(row['displacement_m']) <= .01
                           for row in rows)

            raw.substep_callback = pace_substep
            warmup_crane()
            anchor_x = float(robot.elastic_band.point[0])
            for step in range(305):
                # Move back while suspended so the hands clear the box at release.
                robot.elastic_band.point[0] = anchor_x - .30 * min(step / 80, 1.)
                robot.elastic_band.length = -.40 * float(np.clip((step - 80) / 175, 0., 1.))
                tick('startup', None)
            for step in range(75):
                startup_support = 1. - (step + 1) / 75
                tick('release_crane', None)
            data.xfrc_applied[robot.band_attached_link] = 0.
            for _ in range(100):
                tick('settle', None)
                if not rows[-1]['standing']:
                    break
            gates['startup'] = clean_scene() and quiet(rows[-25:])
            print('startup', gates['startup'], flush=True)
            last_goals = current_wrists()
            clearance = [(np.array([data.qpos[0] + .13, data.qpos[1] + sign * .23, top + .31]),
                          Rotation.identity()) for sign in (1, -1)]
            if gates['startup'] and move('clear', 80, clearance):
                move('ready', 35, clearance)
                window = rows[-20:]
                gates['wrists'] = (clean_scene() and quiet(window)
                                  and all(row['phase'] == 'ready'
                                          and np.max(np.linalg.norm(
                                              np.subtract(row['wrists_xyz_m'], row['wrist_targets_m']), axis=1)) <= .06
                                          for row in window))
            print('wrists', gates['wrists'], flush=True)
            if gates['wrists'] and args.stage == 'lift':
                approach = [('approach', 95, cradle(.24, .29)),
                            ('lower', 85, cradle(.22, .055)),
                            ('cup', 95, cradle(.045, .046)),
                            ('preload', 65, cradle(.033, .046))]
                for phase, count, goals in approach:
                    if not move(phase, count, goals, follow_box=phase in ('cup', 'preload')):
                        break
                if rows[-1]['standing'] and rows[-1]['phase'] == 'preload':
                    lifted = cradle(.023, .175)
                    if move('lift', 125, lifted, follow_box=True):
                        move('hold', 55, lifted, follow_box=True)
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
                          max_lift_m=max(0., max((row['displacement_m'][2] for row in rows), default=0.)),
                          lift_and_hold_pass=lift_pass(rows), task_success=any(r['task_success'] for r in rows),
                          steps=len(rows), gates=gates, hand_torque_limits_nm=limits.tolist())
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
    parser.add_argument('--stage', choices=('ready', 'lift'), default='lift')
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
