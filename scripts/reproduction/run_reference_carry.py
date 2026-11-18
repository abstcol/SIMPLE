"""Test a geometric support grasp using the original GR00T-WBC and a frozen trial."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import simple.envs  # noqa: F401 - register the task
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent
from simple.core.action import ActionCmd
from pi05_common import decode_environment_config, validate_manifest, write_json
from run_pi05_suite import clear_registry_instances, effective_threshold


TASK = 'simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0'
SIDES = ('left', 'right')


class WristIK:
    """Damped least-squares wrist IK on a scratch copy of the MuJoCo state."""

    def __init__(self, model, data, robot_model):
        self.model = model
        self.scratch = mujoco.MjData(model)
        self.scratch.qpos[:] = data.qpos
        self.jac_position = np.zeros((3, model.nv))
        self.jac_rotation = np.zeros((3, model.nv))
        self.arms = {}
        for side in SIDES:
            arm_joints = [
                model.joint(i).name for i in range(model.njnt)
                if model.joint(i).name.startswith(side + '_')
                and any(part in model.joint(i).name for part in ('shoulder', 'elbow', 'wrist'))
            ]
            if len(arm_joints) != 7:
                raise ValueError(f'Expected seven arm joints, got {arm_joints}')
            joints = [model.joint(name).id for name in arm_joints]
            self.arms[side] = (
                model.body(f'{side}_wrist_yaw_link').id,
                model.jnt_qposadr[joints], model.jnt_dofadr[joints],
                model.jnt_range[joints].copy(),
                [robot_model.joint_to_dof_index[name] for name in arm_joints],
            )

    def solve(self, side, state, target_pos, target_rotation, previous):
        body, qpos_ids, velocity_ids, limits, policy_ids = self.arms[side]
        scratch = self.scratch
        scratch.qpos[:] = state.qpos
        scratch.qpos[qpos_ids] = previous[policy_ids]
        for _ in range(16):
            mujoco.mj_kinematics(self.model, scratch)
            mujoco.mj_comPos(self.model, scratch)
            wrist = scratch.body(body)
            position_error = target_pos - wrist.xpos
            rotation_error = Rotation.from_matrix(target_rotation @ wrist.xmat.reshape(3, 3).T).as_rotvec()
            if np.linalg.norm(position_error) < .002 and np.linalg.norm(rotation_error) < .025:
                break
            mujoco.mj_jacBody(self.model, scratch, self.jac_position, self.jac_rotation, body)
            jacobian = np.vstack((self.jac_position[:, velocity_ids], .18 * self.jac_rotation[:, velocity_ids]))
            error = np.r_[position_error, .18 * rotation_error]
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(6) * .0025, error)
            scratch.qpos[qpos_ids] = np.clip(scratch.qpos[qpos_ids] + np.clip(delta, -.16, .16),
                                            limits[:, 0] + .005, limits[:, 1] - .005)
        return policy_ids, scratch.qpos[qpos_ids].copy()

    def press(self, side, state, goal, force, position_gains):
        """Translate a palm force into a bounded position-reference offset."""
        body, _, velocity_ids, limits, policy_ids = self.arms[side]
        wrist = state.body(body)
        palm = wrist.xpos + .085 * wrist.xmat.reshape(3, 3)[:, 0]
        mujoco.mj_jac(self.model, state, self.jac_position, None, palm, body)
        offset = (self.jac_position[:, velocity_ids].T @ force) / position_gains
        goal[policy_ids] = np.clip(goal[policy_ids] + np.clip(offset, -.17, .17),
                                   limits[:, 0] + .005, limits[:, 1] - .005)


def wrist_goal(point, roll=0., pitch=0., yaw=0., reach=.14):
    """Position a point on the open hand along the wrist's local x-axis."""
    rotation = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
    return np.asarray(point) - reach * rotation[:, 0], rotation


def forces_on_target(model, data, target):
    """Sum measured world-frame forces applied to the box by each body."""
    forces = {}
    wrench = np.zeros(6)
    for i, contact in enumerate(data.contact[:data.ncon]):
        a, b = model.geom_bodyid[[contact.geom1, contact.geom2]]
        if target not in (a, b):
            continue
        other = model.body(int(b if a == target else a)).name
        mujoco.mj_contactForce(model, data, i, wrench)
        world = contact.frame.reshape(3, 3).T @ wrench[:3]
        forces[other] = forces.get(other, np.zeros(3)) + (-world if a == target else world)
    return forces


def lateral_pressure(forces, side, sign):
    return -sign * sum(force[1] for name, force in forces.items()
                       if name.startswith(side + '_') and ('hand' in name or 'wrist' in name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--trial', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--approach', choices=('table-edge', 'bimanual-cradle', 'deep-cradle',
                                               'loaded-cradle', 'force-cradle', 'side-cradle', 'tip-and-scoop'),
                        default='tip-and-scoop')
    args = parser.parse_args()
    records = validate_manifest(json.loads(args.manifest.read_text()))
    trial = next((row for row in records if row['trial_id'] == args.trial), None)
    if trial is None or trial['env_id'] != TASK or trial['controller'] != 'decoupled':
        parser.error('Select a GR00T two-table carry trial from the evaluation manifest.')
    args.output.mkdir(parents=True, exist_ok=False)
    from simple.robots.registry import RobotRegistry
    from simple.tasks.registry import TaskRegistry
    import torch

    random.seed(trial['sim_seed'])
    np.random.seed(trial['sim_seed'])
    torch.manual_seed(trial['sim_seed'])
    clear_registry_instances(TaskRegistry, RobotRegistry)
    config = SimLoopConfig().load_wbc_yaml()
    config['ENV_NAME'] = 'simple'
    env = gym.make(TASK, sim_mode='mujoco', headless=True, webrtc=False, render_hz=50,
                   sonic_config=config, physics=trial.get('physics'))
    raw = env.unwrapped
    raw.mujoco.upstream_object_gravity = True
    env = gym.wrappers.TimeLimit(env, max_episode_steps=trial['max_steps'])
    assert math.isclose(effective_threshold(raw.task, TASK), trial['success_criteria'])
    obs, _ = env.reset(seed=trial['sim_seed'],
                       options={'state_dict': decode_environment_config(trial['environment_config'])})
    agent = SonicDecoupledWbcAgent(raw.task.robot, config)
    agent.reset()
    policy = agent._wbc_policy
    policy.lower_body_policy.use_policy_action = True
    dt = 4 * config['SIMULATE_DT']
    for _ in range(300):
        if raw.task.robot.stabilized:
            break
        start = time.monotonic()
        obs, *_ = raw.step(agent.get_stabilize_action(obs))
        raw.update_reward()
        time.sleep(max(0, dt - (time.monotonic() - start)))
    policy.lower_body_policy.gait_indices = torch.zeros((1,), dtype=torch.float32)

    model, data = raw.mjModel, raw.mjData
    robot_model = agent._dwbc_robot_model
    ik = WristIK(model, data, robot_model)
    upper = agent._upper_body_indices
    full_goal = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=raw.task.robot.prepare_obs()['body_q'],
        left_hand_actuated_joint_values=np.zeros(7), right_hand_actuated_joint_values=np.zeros(7))
    target = raw.mujoco.mj_objects['target'].id
    table = model.geom('table_geom')
    edge = float(data.geom(table.id).xpos[0] - table.size[0])
    top = float(data.geom(table.id).xpos[2] + table.size[2])
    initial = data.body(target).xpos.copy()
    x, y, z = initial
    # The thin dimension is aligned with the approach in the selected frozen scene.
    # Reach over the far edge and pull the box toward a palm below the near edge.
    left_wait = wrist_goal([edge + .005, y + .03, top - .03], roll=-90, reach=.12)
    right_above = wrist_goal([x + .075, y - .025, z + .15], pitch=65)
    right_behind = wrist_goal([x + .042, y - .025, z + .063], pitch=65)
    right_pull = wrist_goal([edge - .10, y - .025, z + .008], pitch=65)
    left_lift = wrist_goal([edge - .018, y + .025, top + .16], roll=-90, reach=.12)
    right_support = wrist_goal([edge - .005, y - .025, top + .22], pitch=45)
    stages = [
        ('approach', 95, (left_wait, right_above)),
        ('reach_far_side', 65, (left_wait, right_behind)),
        ('tip_toward_support', 100, (left_wait, right_pull)),
        ('lift', 100, (left_lift, right_support)),
        ('hold', 60, (left_lift, right_support)),
    ]
    side_entry = args.approach == 'side-cradle'
    scoop = args.approach == 'tip-and-scoop'
    force_guided = args.approach in ('force-cradle', 'side-cradle')
    loaded = args.approach == 'loaded-cradle' or force_guided
    if args.approach != 'table-edge':
        # Angle both palms upward and approach the lower corners from the sides.
        # The fingers are open. The arms move the palms inward and then upward.
        deep = args.approach != 'bimanual-cradle'

        def cradle(clearance, height, forward=0.):
            return tuple(wrist_goal([x + forward + (.055 if side_entry else 0.),
                                     y + sign * clearance, top + height],
                                    roll=-sign * 45, yaw=-sign * (5 if side_entry else 12 if deep else 22),
                                    reach=.075 if deep else .13)
                         for sign in (1, -1))

        stages = [('approach', 100, cradle(.175 if side_entry else .155, .07)),
                  ('cup_lower_corners', 105, cradle(.044 if deep else .058, .05)),
                  ('lift', 100, cradle(.038 if deep else .051, .18, -.01)),
                  ('hold', 60, cradle(.038 if deep else .051, .18, -.01))]
        if deep:
            # Keep the palms high and outside the box during the base approach.
            clear = tuple(wrist_goal([edge - .035, y + sign * .24, top + .28], reach=.10)
                          for sign in (1, -1))
            stages[:0] = [('clear_box', 80, clear), ('step_closer', 125, clear)]
        if loaded:
            cup = cradle(.044, .05)
            # Carry the commanded wrist pose across the phase boundary. The
            # measured pose while pressing encodes deflection and would unload
            # the contact if used as a fresh setpoint.
            stages[-2:] = [('preload', 80, cup),
                           ('lift', 200, tuple((p + [0, 0, .13], r) for p, r in cup)),
                           ('hold', 60, tuple((p + [0, 0, .13], r) for p, r in cup))]
    if scoop:
        left_clear = wrist_goal([edge - .018, y + .055, top + .038], roll=-66, pitch=18, reach=.15)
        right_near = wrist_goal([x - .092, y - .042, z + .069], roll=-10, pitch=8, reach=.157)
        right_tip = wrist_goal([x + .008, y - .042, z + .069], roll=-10, pitch=8, reach=.157)
        stages[2:] = [('get_into_position', 115, (left_clear, right_near)),
                      ('tip_box', 145, (left_clear, right_tip)),
                      ('slide_under', 120, (left_clear, right_tip)),
                      ('lift', 125, (left_clear, right_tip)),
                      ('hold', 60, (left_clear, right_tip))]
    model.vis.global_.offwidth = max(640, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(480, model.vis.global_.offheight)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.5, 132, -22
    camera.lookat[:] = [edge - .08, y, .66]
    rows = []
    ended = False
    previous_goals = None
    squeeze = np.zeros(2)
    lift_progress = 0
    loads = {}
    tip_goal = None
    scoop_goal = None
    hand_limits = raw.task.robot.torque_limit
    for joints in (raw.task.robot.left_hand_index, raw.task.robot.right_hand_index):
        np.testing.assert_allclose(hand_limits[joints - 1], [2.45, .7, .7, .7, .7, .7, .7])
    arm_kp = dict(zip(raw.task.robot.joint_names[:29], config['MOTOR_KP']))
    position_gains = {side: np.array([arm_kp[model.joint(i).name] for i in range(model.njnt)
                                     if model.jnt_qposadr[i] in ik.arms[side][1]]) for side in SIDES}
    try:
        with imageio.get_writer(args.output / 'overview.mp4', fps=25, codec='libx264') as video:
            for name, steps, goals in stages:
                # Interpolate the wrists from their actual poses at stage entry.
                origins = [(data.body(ik.arms[side][0]).xpos.copy(),
                            data.body(ik.arms[side][0]).xmat.reshape(3, 3).copy()) for side in SIDES]
                pressing = loaded and name in ('preload', 'lift', 'hold')
                if pressing:
                    origins = previous_goals
                if scoop and name == 'slide_under':
                    origins = tip_goal
                if scoop and name in ('lift', 'hold'):
                    origins = scoop_goal if name == 'lift' else previous_goals
                    goals = tuple((p + np.array([-.025, 0., .104]), r) for p, r in scoop_goal)
                stage_rows = []
                for step in range(steps):
                    fraction = min(1., (step + 1) / (steps * .80))
                    pressure = np.array([lateral_pressure(loads, side, sign)
                                         for side, sign in zip(SIDES, (1, -1))])
                    if pressing and not force_guided:
                        squeeze = np.clip(squeeze + .00018 * np.clip(7. - pressure, -4., 7.), 0., .037)
                    if loaded and name == 'lift':
                        if pressure.min() >= 3. or loads.get('table', np.zeros(3))[2] < 2.:
                            lift_progress += 1
                        fraction = min(1., lift_progress / 100.)
                    fraction = fraction**2 * (3 - 2 * fraction)
                    commanded_wrists = []
                    step_goals = goals
                    if scoop and name in ('tip_box', 'slide_under'):
                        box = data.body(target)
                        box_rotation = box.xmat.reshape(3, 3)
                        bottom = box.xpos + box_rotation @ [-.028, .04, -.113]
                        if name == 'tip_box' and tip_goal is None and bottom[2] >= top + .023 and step >= 20:
                            # Freeze the pusher command as soon as a finger can enter the gap.
                            tip_goal = tuple((p0 + fraction * (p1 - p0), r1)
                                             for (p0, _), (p1, r1) in zip(origins, goals))
                        if tip_goal is not None:
                            if name == 'tip_box':
                                step_goals = tip_goal
                                fraction = 1.
                            else:
                                point = bottom + [.012, -.002, -.008]
                                point[2] = max(top + .010, point[2])
                                step_goals = (wrist_goal(point, roll=-66, pitch=18, reach=.15), tip_goal[1])
                    for i, (side, (p0, r0), (p1, r1)) in enumerate(zip(SIDES, origins, step_goals)):
                        axis = Rotation.from_matrix(r1 @ r0.T).as_rotvec()
                        position = p0 + fraction * (p1 - p0)
                        if pressing:
                            position[1] += (-1 if side == 'left' else 1) * squeeze[i]
                        commanded_wrists.append(position.tolist())
                        ids, values = ik.solve(side, data, position,
                                               Rotation.from_rotvec(fraction * axis).as_matrix() @ r0,
                                               full_goal)
                        full_goal[ids] += np.clip(values - full_goal[ids], -.055, .055)
                    vx = .15 if name == 'step_closer' and data.qpos[0] < edge - .22 else 0.
                    joint_goal = full_goal.copy()
                    if force_guided and pressing:
                        ramp = min(1., (step + 1) / 50.) if name == 'preload' else 1.
                        lift = 4. if name in ('lift', 'hold') and lift_progress else 0.
                        for side, sign in zip(SIDES, (1, -1)):
                            ik.press(side, data, joint_goal, np.array([0., -sign * 16. * ramp, lift]),
                                     position_gains[side])
                    if scoop and name in ('lift', 'hold'):
                        ik.press('left', data, joint_goal, np.array([1., 0., 9.]), position_gains['left'])
                        ik.press('right', data, joint_goal, np.array([4., 0., 1.]), position_gains['right'])
                    now = time.monotonic()
                    policy.set_observation(agent._build_wbc_observation(raw.task.robot.prepare_obs()))
                    policy.set_goal({'target_upper_body_pose': joint_goal[upper],
                                     'navigate_cmd': np.array([vx, 0., 0., 0.]),
                                     'base_height_command': np.array([.74]),
                                     'target_time': now + 1 / agent._control_frequency,
                                     'interpolation_garbage_collection_time': now - 2 / agent._control_frequency,
                                     'timestamp': now})
                    q = policy.get_action(time=now)['q']
                    action = ActionCmd('decoupled_wbc', target_q=robot_model.get_body_actuated_joints(q),
                                       left_hand_q=robot_model.get_hand_actuated_joints(q, side='left'),
                                       right_hand_q=robot_model.get_hand_actuated_joints(q, side='right'))
                    obs, reward, terminated, truncated, info = env.step(action)
                    loads = forces_on_target(model, data, target)
                    state = {'stage': name, 'step': len(rows) + 1, 'target': info['target'].tolist(),
                             'pelvis': data.qpos[:7].tolist(), 'reward':float(reward),
                             'wrists': [data.body(ik.arms[s][0]).xpos.tolist() for s in SIDES],
                             'wrist_targets': commanded_wrists,
                             'target_contact_bodies': sorted(loads), 'target_contact_forces_n': loads,
                             'commands': action.parameters, 'lift_progress': lift_progress}
                    rows.append(state)
                    stage_rows.append(state)
                    if len(rows) % 2 == 0:
                        renderer.update_scene(data, camera=camera)
                        video.append_data(renderer.render())
                    if terminated or truncated or data.qpos[2] < .45:
                        ended = True
                        break
                    time.sleep(max(0, dt - (time.monotonic() - now)))
                print(name, 'target', np.round(stage_rows[-1]['target'][:3], 3).tolist(),
                      'wrists', np.round(stage_rows[-1]['wrists'], 3).tolist(),
                      'contacts', {body: np.round(force, 1).tolist() for body, force in loads.items()}, flush=True)
                previous_goals = goals
                if scoop and name == 'tip_box' and tip_goal is None:
                    print('The near bottom edge never opened a 23 mm gap.', flush=True)
                    break
                if scoop and name == 'slide_under':
                    if not any(any(body.startswith('left_') for body in row['target_contact_bodies'])
                               for row in stage_rows[-25:]):
                        print('No support-hand contact after insertion.', flush=True)
                        break
                    scoop_goal = step_goals
                if loaded and name == 'lift' and lift_progress < 100:
                    print('Lift paused because the hands did not maintain inward pressure.', flush=True)
                    break
                if ended:
                    break
    finally:
        renderer.close()
        env.close()
        write_json(args.output / 'trajectory.json', rows)
        write_json(args.output / 'trial.json', trial)
        z_values = [row['target'][2] for row in rows]
        hold = [row for row in rows if row['stage'] == 'hold']
        clear_of_table = all(any('hand' in body or 'wrist' in body for body in row['target_contact_bodies'])
                             and not any('table' in body for body in row['target_contact_bodies'])
                             for row in hold[-25:])
        print('initial', initial.tolist(), 'table edge/top', [edge, top], flush=True)
        result = {'trial_id':trial['trial_id'], 'approach':args.approach,
                  'target_mass_kg':float(model.body_mass[target]),
                  'hand_torque_limits_nm':hand_limits[raw.task.robot.right_hand_index - 1].tolist(),
                  'target_initial_height_m':float(z), 'max_lift_m':max(z_values, default=z)-z,
                  'lift_and_hold_pass':len(hold) == 60 and clear_of_table
                  and min(row['target'][2] for row in hold[-25:]) - z >= .05,
                  'steps':len(rows), 'task_success':bool(raw._success),
                  'robot_stabilized':bool(raw.task.robot.stabilized)}
        write_json(args.output / 'result.json', result)
        print(json.dumps(result, default=lambda value: value.item()), flush=True)


if __name__ == '__main__':
    main()
