"""Check two-arm support using simulator state, a frozen trial and GR00T-WBC or native SONIC."""

import argparse
from contextlib import ExitStack
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

    def wrist_poses(self, state, joint_goal):
        self.scratch.qpos[:] = state.qpos
        for body, qpos_ids, _, _, policy_ids in self.arms.values():
            self.scratch.qpos[qpos_ids] = joint_goal[policy_ids]
        mujoco.mj_kinematics(self.model, self.scratch)
        return [(self.scratch.body(arm[0]).xpos.copy(), self.scratch.body(arm[0]).xmat.reshape(3, 3).copy())
                for arm in self.arms.values()]


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


def foot_support(model, data, ground, feet):
    """Measure loaded foot contacts and tangential velocity at the contact point."""
    support = {}
    wrench = np.zeros(6)
    velocity = np.zeros(6)
    for i, contact in enumerate(data.contact[:data.ncon]):
        if ground not in (contact.geom1, contact.geom2):
            continue
        body = int(model.geom_bodyid[contact.geom2 if contact.geom1 == ground else contact.geom1])
        if body not in feet:
            continue
        mujoco.mj_contactForce(model, data, i, wrench)
        normal = float(wrench[0])
        if normal <= 5.:
            continue
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, velocity, 0)
        point_velocity = velocity[3:] + np.cross(velocity[:3], contact.pos - data.xpos[body])
        tangent_speed = float(np.linalg.norm(contact.frame.reshape(3, 3)[1:] @ point_velocity))
        foot = support.setdefault(feet[body], {'normal_n': 0., 'max_tangent_speed_mps': 0.,
                                              'max_tangent_to_normal': 0.,
                                              'sliding_friction': contact.friction[:2].tolist()})
        foot['normal_n'] += normal
        foot['max_tangent_speed_mps'] = max(foot['max_tangent_speed_mps'], tangent_speed)
        foot['max_tangent_to_normal'] = max(foot['max_tangent_to_normal'],
                                           float(np.linalg.norm(wrench[1:3]) / normal))
    return support


def held_above_table(rows, initial_height):
    window = [row for row in rows if row['stage'] == 'hold'][-25:]
    return len(window) == 25 and all(
        row['target'][2] >= initial_height + .05
        and 'table' not in row['target_contact_bodies']
        and all(any(body.startswith(side + '_') and ('hand' in body or 'wrist' in body)
                    for body in row['target_contact_bodies']) for side in SIDES)
        for row in window)


def placed_in_container(rows, container_name):
    window = rows[-20:]
    return len(window) == 20 and all(
        row['stage'] == 'settle' and container_name in row['target_contact_bodies']
        and not any(body.startswith(('left_', 'right_')) and ('hand' in body or 'wrist' in body)
                    for body in row['target_contact_bodies'])
        and np.linalg.norm(np.array(row['target'][:2]) - row['container'][:2]) < .10
        for row in window)


def yaw_rotation(qpos):
    w, x, y, z = qpos[3:7]
    return Rotation.from_euler('z', np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))).as_matrix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--trial', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--controller', choices=('gr00t', 'sonic', 'sonic-split'), default='gr00t')
    parser.add_argument('--carry', action='store_true', help='Continue to the second table after passing the lift check.')
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
    env = gym.make(TASK, sim_mode='mujoco', headless=True, webrtc=False, render_hz=50, disable_env_checker=True,
                   sonic_config=config, physics=trial.get('physics'))
    raw = env.unwrapped
    raw.mujoco.upstream_object_gravity = True
    env = gym.wrappers.TimeLimit(env, max_episode_steps=trial['max_steps'])
    assert math.isclose(effective_threshold(raw.task, TASK), trial['success_criteria'])
    obs, _ = env.reset(seed=trial['sim_seed'],
                       options={'state_dict': decode_environment_config(trial['environment_config'])})
    # The scripted expert only needs physics state. Render saved frames after
    # the run so the two real-time controllers see the same recording overhead.
    camera_names = tuple(raw.mujoco.renderers)
    raw.mujoco.close()
    raw.mujoco.renderers = dict.fromkeys(camera_names)
    agent = SonicDecoupledWbcAgent(raw.task.robot, config)
    agent.reset()
    native_start = args.controller == 'sonic-split'
    robot = raw.task.robot
    band = robot.elastic_band if native_start else None
    band_action = ActionCmd('elastic_band')
    policy = agent._wbc_policy
    policy.lower_body_policy.use_policy_action = True
    dt = 4 * config['SIMULATE_DT']
    model, data = raw.mjModel, raw.mjData
    ground = model.geom('ground').id
    feet = {model.body(f'{side}_ankle_roll_link').id: side for side in SIDES}
    target = raw.mujoco.mj_objects['target'].id
    spawn_box = data.body(target).xpos.copy()
    pelvis = model.body('pelvis').id
    robot_bodies = {model.body(i).name for i in range(model.nbody)
                    if model.body_rootid[i] == pelvis}
    early_contacts = set()
    substep_contacts = set()
    sonic = None
    sonic_action = None
    physics_deadline = 0.
    band_scale = 1.
    control_time_origin = time.monotonic()

    def apply_band():
        if band is not None and band.enable:
            robot.apply_action(band_action)
            data.xfrc_applied[robot.band_attached_link] *= band_scale

    def observe_substep():
        nonlocal physics_deadline
        substep_contacts.update(robot_bodies.intersection(forces_on_target(model, data, target)))
        if sonic is not None:
            if native_start:
                apply_band()
                sonic.cache_substep()
            else:
                sonic.cache_state()
                physics_deadline += config['SIMULATE_DT']
                time.sleep(max(0., physics_deadline - time.monotonic()))
                sonic.refresh_action(sonic_action)

    raw.substep_callback = observe_substep
    if not native_start:
        for _ in range(300):
            if robot.stabilized:
                break
            start = time.monotonic()
            obs, *_ = raw.step(agent.get_stabilize_action(obs))
            raw.update_reward()
            early_contacts.update(substep_contacts)
            time.sleep(max(0, dt - (time.monotonic() - start)))
    policy.lower_body_policy.gait_indices = torch.zeros((1,), dtype=torch.float32)

    robot_model = agent._dwbc_robot_model
    ik = WristIK(model, data, robot_model)
    upper = agent._upper_body_indices
    # Match the original WBC's motor ownership, including the waist, in MuJoCo order.
    lower_mask = np.zeros(robot_model.num_dofs, dtype=bool)
    lower_mask[robot_model.get_joint_group_indices('lower_body')] = True
    lower_motors = np.flatnonzero(robot_model.get_body_actuated_joints(lower_mask))
    grounded_height = float(data.qpos[2])
    if native_start:
        # Start from the asset's native pose under the band, as in the stand-alone simulator.
        lower_qpos = model.jnt_qposadr[robot.body_joint_index[lower_motors]]
        data.qpos[lower_qpos] = model.qpos0[lower_qpos]
        band.point[:] = [*data.qpos[:2], data.geom(ground).xpos[2] + 1.]
        data.qpos[2] = model.qpos0[2]
        data.qpos[3:7] = Rotation.from_matrix(yaw_rotation(data.qpos)).as_quat(scalar_first=True)
        mujoco.mj_forward(model, data)
    full_goal = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=raw.task.robot.prepare_obs()['body_q'],
        left_hand_actuated_joint_values=np.zeros(7), right_hand_actuated_joint_values=np.zeros(7))
    container = raw.mujoco.mj_objects['container'].id
    container_name = model.body(container).name
    table = model.geom('table_geom')
    edge = float(data.geom(table.id).xpos[0] - table.size[0])
    top = float(data.geom(table.id).xpos[2] + table.size[2])
    initial = data.body(target).xpos.copy()
    x, y, z = initial
    # Keep wrists clear during the base approach, then cup the lower corners.
    clear = tuple(wrist_goal([edge - .035, y + sign * .24, top + .28], reach=.10)
                  for sign in (1, -1))
    if native_start:
        # Start above the normal wrist-clearance pose, with the hands outside the box path.
        offset = [0., 0., data.qpos[2] - grounded_height]
        for side, (position, rotation) in zip(SIDES, clear):
            ids, angles = ik.solve(side, data, position + offset, rotation, full_goal)
            full_goal[ids] = angles
            data.qpos[ik.arms[side][1]] = angles
        mujoco.mj_forward(model, data)

    def cradle(clearance, height):
        return tuple(wrist_goal([x, y + sign * clearance, top + height],
                                roll=-sign * 45, yaw=-sign * 12, reach=.075)
                     for sign in (1, -1))

    cupped = cradle(.044, .05)
    raised = tuple((position + [0, 0, .13], rotation) for position, rotation in cupped)
    stages = [('clear_box', 80, clear), ('step_closer', 125, clear),
              ('approach', 100, cradle(.155, .07)), ('cup_lower_corners', 105, cupped),
              ('preload', 80, cupped), ('lift', 200, raised), ('hold', 60, raised)]
    model.vis.global_.offwidth = max(640, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(480, model.vis.global_.offheight)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.5, 40 if native_start else 132, -22
    camera.lookat[:] = [edge - .08, y, .66]
    rows, frames = [], []
    ended = False
    handover_pass = None
    standing_pass = wrist_clearance_pass = None
    official_success_step = None
    previous_goals = None
    lift_progress = 0
    loads = {}
    box_velocity = np.zeros(6)
    hand_limits = raw.task.robot.torque_limit
    for joints in (raw.task.robot.left_hand_index, raw.task.robot.right_hand_index):
        np.testing.assert_allclose(hand_limits[joints - 1], [2.45, .7, .7, .7, .7, .7, .7])
    arm_kp = dict(zip(raw.task.robot.joint_names[:29], config['MOTOR_KP']))
    position_gains = {side: np.array([arm_kp[model.joint(i).name] for i in range(model.njnt)
                                     if model.jnt_qposadr[i] in ik.arms[side][1]]) for side in SIDES}

    def advance(stage, joint_goal, navigation, wrist_targets, base_height=.74):
        nonlocal obs, loads, ended, official_success_step, sonic_action, physics_deadline
        now = control_time_origin + data.time if native_start else time.monotonic()
        q = joint_goal
        split = native_start
        if sonic is None or split:
            goal = {'target_upper_body_pose': joint_goal[upper], 'navigate_cmd': navigation,
                    'base_height_command': np.array([base_height]),
                    'target_time': now + 1 / agent._control_frequency,
                    'interpolation_garbage_collection_time': now - 2 / agent._control_frequency}
            if split:
                policy.upper_body_policy.set_goal(goal)
                q = joint_goal.copy()
                q[upper] = policy.upper_body_policy.get_action(now)['target_upper_body_pose']
            else:
                policy.set_observation(agent._build_wbc_observation(raw.task.robot.prepare_obs()))
                policy.set_goal(goal)
                q = policy.get_action(time=now)['q']
        commands = dict(target_q=robot_model.get_body_actuated_joints(q),
                        left_hand_q=robot_model.get_hand_actuated_joints(q, side='left'),
                        right_hand_q=robot_model.get_hand_actuated_joints(q, side='right'))
        action = ActionCmd('decoupled_wbc', **commands)
        if sonic is not None:
            if split:
                action.parameters['sonic_body_indices'] = lower_motors
            rotation = yaw_rotation(data.qpos)
            heading = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
            action = sonic_action = sonic.action(commands['target_q'], commands['left_hand_q'],
                                                 commands['right_hand_q'], navigation, base_height, heading,
                                                 split_action=action if split else None)
            if not split:
                physics_deadline = max(now - .005, physics_deadline)
        substep_contacts.clear()
        apply_band()
        step_env = raw if stage in ('crane', 'stand') else env
        obs, reward, terminated, truncated, info = step_env.step(action)
        loads = forces_on_target(model, data, target)
        state = {'stage': stage, 'step': len(rows) + 1, 'target': info['target'].tolist(),
                 'container': info['container'].tolist(), 'official_success': bool(terminated),
                 'pelvis': data.qpos[:7].tolist(), 'reward': float(reward),
                 'wrists': [data.body(ik.arms[s][0]).xpos.tolist() for s in SIDES],
                 'wrist_targets': wrist_targets, 'navigate_cmd': list(navigation),
                 'base_height_command': -1. if split else base_height,
                 'target_contact_bodies': sorted(loads), 'target_contact_forces_n': loads,
                 'commands': commands, 'lift_progress': lift_progress,
                 'robot_contacts_over_tick': sorted(substep_contacts),
                 'foot_support': foot_support(model, data, ground, feet),
                 'object_displacement_m': data.body(target).xpos - spawn_box}
        if band is not None:
            state['band_support_n'] = float(data.xfrc_applied[robot.band_attached_link, 2])
        if sonic is not None:
            state['sonic_motor_target_q'] = [float(m.q) for m in action['low_cmd'].motor_cmd[:29]]
            if split:
                state['sonic_ack_step'] = sonic.barrier.sim_step_seq
        rows.append(state)
        if len(rows) % 2 == 0:
            frames.append((stage, data.qpos.copy()))
        if terminated and official_success_step is None:
            official_success_step = len(rows)
        # Continue a diagnostic placement if the score fires on basket contact.
        # Preserve the first official success and honor the original TimeLimit.
        ended = (terminated and not args.carry) or truncated or data.qpos[2] < .45
        if not split:
            time.sleep(max(0, dt - (time.monotonic() - now)))
        return state

    def box_was_disturbed():
        early_contacts.update(substep_contacts)
        return bool(early_contacts) or np.linalg.norm(data.body(target).xpos - spawn_box) > .01

    def stand_with_sonic(resources):
        nonlocal sonic, band_scale
        from sonic_reference import SonicReference

        facing = yaw_rotation(data.qpos)
        yaw = float(np.arctan2(facing[1, 0], facing[0, 0]))
        idle = [0., 0., 0., yaw]
        sonic = resources.enter_context(SonicReference(robot, config, args.output, lockstep=True))
        # The official 50 Hz startup ramp takes 152 boundaries under support.
        for _ in range(152):
            advance('crane', full_goal, idle, None)
            if ended or box_was_disturbed():
                return False

        # Follow the native band descent, then wait for the feet to bear the
        # weight before removing its vertical and lateral support.
        weight = -model.opt.gravity[2] * model.body_subtreemass[pelvis]
        loaded = 0
        for step in range(round(9. / dt)):
            if step * dt < 3.5:
                band.length = -.20 * min(1., (step + 1) * dt / 3.5)
            elif not loaded:
                band.length = max(-.32, band.length - .035 * dt)
            state = advance('crane', full_goal, idle, None)
            if ended or box_was_disturbed():
                return False
            foot_load = sum(foot['normal_n'] for foot in state['foot_support'].values())
            supported = foot_load >= .75 * weight and state['band_support_n'] <= .25 * weight
            loaded = loaded + 1 if supported else 0
            if loaded >= 15:
                break
        else:
            print('The feet did not take enough load before crane release.', flush=True)
            return False
        release_steps = round(1.5 / dt)
        for step in range(release_steps):
            band_scale = 1. - (step + 1) / release_steps
            advance('crane', full_goal, idle, None)
            if ended or box_was_disturbed():
                return False
        band.enable = False
        data.xfrc_applied[robot.band_attached_link] = 0.
        for step in range(round(4. / dt)):
            advance('stand', full_goal, idle, None)
            if ended or box_was_disturbed():
                break
            if step >= 24:
                window = rows[-25:]
                poses = np.array([row['pelvis'][:3] for row in window])
                if (poses[:, 2].min() > .65 and np.linalg.norm(np.ptp(poses, axis=0)) < .04
                        and all(len(row['foot_support']) == 2 for row in window)):
                    return True
        return False

    def carry_to_container(joint_goal):
        box = data.body(target)
        basket = data.body(container)
        carry_height = .74
        transport_goal = joint_goal.copy()
        lost = 0
        for _ in range(365):
            heading = yaw_rotation(data.qpos)
            facing_basket = heading[0, 0] < -.97
            if facing_basket:
                # GR00T raises its base before crossing the rim. With SONIC's
                # nominal walking height, use the same wrist IK to lift the box.
                clearance = float(np.clip((-data.qpos[0] - 1.30) / .28, 0., 1.))
                if not native_start:
                    carry_height = .74 + .065 * clearance
                error = basket.xpos[:2] - box.xpos[:2]
                if np.linalg.norm(error) < .045 and box.xpos[2] > .86:
                    break
                if native_start and clearance > 0.:
                    for side, (position, rotation) in zip(SIDES, ik.wrist_poses(data, joint_goal)):
                        ids, angles = ik.solve(side, data, position + [0., 0., .09 * clearance],
                                               rotation, transport_goal)
                        transport_goal[ids] += np.clip(angles - transport_goal[ids], -.055, .055)
                world_velocity = np.clip(2.5 * error, [-.40, -.18], [.18, .18])
                if error[0] < -.045:
                    world_velocity[0] = min(world_velocity[0], -.12)
            else:
                world_velocity = [-.19, 0.] if data.qpos[0] > edge - .38 else [0., 0.]
            local_velocity = heading[:2, :2].T @ world_velocity
            advance('carry', transport_goal, [*local_velocity, 1., -np.pi], None, carry_height)
            lost = lost + 1 if not any('hand' in body or 'wrist' in body for body in loads) else 0
            if ended or lost >= 10:
                print('Carry stopped after contact loss or episode termination.', flush=True)
                return
        else:
            print('The box did not reach the release pose over the container.', flush=True)
            return

        heading = yaw_rotation(data.qpos)
        origins = ik.wrist_poses(data, transport_goal)
        release_steps = 36
        for step in range(trial['max_steps']):
            progress = min(1., (step + 1) / release_steps)
            smooth = progress**2 * (3. - 2. * progress)
            targets = []
            for side, sign, (p, r) in zip(SIDES, (1, -1), origins):
                # First clear the sides. Then draw the hands back from the rim.
                retreat = .05 * max(0., (smooth - .4) / .6)
                goal = p + heading @ [-retreat, sign * .09 * smooth, .02 * smooth]
                targets.append(goal.tolist())
                indices, q = ik.solve(side, data, goal, r, transport_goal)
                transport_goal[indices] += np.clip(q - transport_goal[indices], -.055, .055)
            advance('release' if step < release_steps else 'settle', transport_goal,
                    [0., 0., 0., -np.pi], targets, carry_height)
            if ended or placed_in_container(rows, container_name):
                return

    try:
        with ExitStack() as resources:
            if early_contacts or np.linalg.norm(initial - spawn_box) > .01:
                raise RuntimeError(f'Stabilization disturbed the box: {sorted(early_contacts)}')
            if native_start:
                standing_pass = stand_with_sonic(resources)
                print('SONIC unassisted standing', standing_pass, 'contacts', sorted(early_contacts), flush=True)
                if not standing_pass:
                    return
            for name, steps, goals in stages:
                origins = [(data.body(ik.arms[side][0]).xpos.copy(),
                            data.body(ik.arms[side][0]).xmat.reshape(3, 3).copy()) for side in SIDES]
                pressing = name in ('preload', 'lift', 'hold')
                if pressing:
                    # Continue from the commanded pose to preserve arm preload.
                    origins = previous_goals
                stage_rows = []
                for step in range(steps):
                    fraction = min(1., (step + 1) / (steps * .80))
                    if name == 'lift':
                        pressure = [lateral_pressure(loads, side, sign) for side, sign in zip(SIDES, (1, -1))]
                        if min(pressure) >= 3. or loads.get('table', np.zeros(3))[2] < 2.:
                            lift_progress += 1
                        fraction = min(1., lift_progress / 100.)
                    fraction = fraction**2 * (3 - 2 * fraction)
                    commanded_wrists = []
                    forward_bias = 0.
                    shift = np.zeros(3)
                    if name in ('lift', 'hold'):
                        # Keep the wrists beside the actual box while damping
                        # its tendency to move away along the table.
                        displacement = data.body(target).xpos - initial
                        shift[:2] = np.clip(displacement[:2], [-.035, -.035], [.09, .035])
                        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, target, box_velocity, 0)
                        forward_bias = float(np.clip(-80. * displacement[0] - 12. * box_velocity[3], -8., 8.))
                    for side, (p0, r0), (p1, r1) in zip(SIDES, origins, goals):
                        axis = Rotation.from_matrix(r1 @ r0.T).as_rotvec()
                        position = p0 + fraction * (p1 - p0) + shift
                        commanded_wrists.append(position.tolist())
                        ids, values = ik.solve(side, data, position,
                                               Rotation.from_rotvec(fraction * axis).as_matrix() @ r0,
                                               full_goal)
                        full_goal[ids] += np.clip(values - full_goal[ids], -.055, .055)
                    vx = .15 if name == 'step_closer' and data.qpos[0] < edge - .22 else 0.
                    joint_goal = full_goal.copy()
                    if pressing:
                        ramp = min(1., (step + 1) / 50.) if name == 'preload' else 1.
                        lift = 4. if name in ('lift', 'hold') and lift_progress else 0.
                        for side, sign in zip(SIDES, (1, -1)):
                            ik.press(side, data, joint_goal, np.array([forward_bias, -sign * 16. * ramp, lift]),
                                     position_gains[side])
                    state = advance(name, joint_goal, [vx, 0., 0., 0.], commanded_wrists)
                    stage_rows.append(state)
                    if ended:
                        break
                print(name, 'target', np.round(stage_rows[-1]['target'][:3], 3).tolist(),
                      'wrists', np.round(stage_rows[-1]['wrists'], 3).tolist(),
                      'contacts', {body: np.round(force, 1).tolist() for body, force in loads.items()}, flush=True)
                previous_goals = goals
                if name in ('clear_box', 'step_closer'):
                    early_contacts.update(c for row in stage_rows for c in row['robot_contacts_over_tick'])
                    if early_contacts or np.linalg.norm(data.body(target).xpos - spawn_box) > .01:
                        print('Early robot contact or box motion before approach.', flush=True)
                        break
                if native_start and name == 'clear_box':
                    window = stage_rows[-25:]
                    errors = np.linalg.norm(np.array([row['wrists'] for row in window])
                                            - np.array([position for position, _ in goals]), axis=2)
                    wrist_clearance_pass = not ended and len(window) == 25 and errors.max() <= .06
                    print('SONIC wrist clearance', wrist_clearance_pass,
                          'max error (m)', errors.max(), flush=True)
                    if not wrist_clearance_pass:
                        break
                if args.controller == 'sonic' and name == 'clear_box' and not ended:
                    from sonic_reference import SonicReference
                    sonic = resources.enter_context(SonicReference(raw.task.robot, config, args.output))
                    heading = yaw_rotation(data.qpos)
                    sonic.engage(float(np.arctan2(heading[1, 0], heading[0, 0])))
                    wrist_reference = [position.tolist() for position, _ in goals]
                    handover_pass = False
                    for handover_step in range(125):
                        advance('handover', full_goal, [0., 0., 0., 0.], wrist_reference)
                        early_contacts.update(substep_contacts)
                        if ended or early_contacts or np.linalg.norm(data.body(target).xpos - spawn_box) > .01:
                            break
                        if handover_step >= 24:
                            window = rows[-25:]
                            poses = np.array([row['pelvis'][:3] for row in window])
                            wrist_error = np.linalg.norm(np.array([row['wrists'] for row in window]) - wrist_reference, axis=2)
                            if (poses[:, 2].min() > .65 and np.linalg.norm(np.ptp(poses, axis=0)) < .04
                                    and wrist_error.max() <= .06):
                                handover_pass = True
                                break
                    travel = np.ptp(np.array([row['pelvis'][:3] for row in rows[-25:]]), axis=0)
                    print('SONIC handover', bool(handover_pass), 'base motion', travel.tolist(),
                          'contacts', sorted(early_contacts), flush=True)
                    if not handover_pass:
                        return
                if name == 'lift' and lift_progress < 100:
                    print('Lift paused because the hands did not maintain inward pressure.', flush=True)
                    break
                if ended:
                    break
            if args.carry and not ended and held_above_table(rows, z):
                print('Lift passed. Starting transport and placement.', flush=True)
                carry_to_container(joint_goal)
    finally:
        write_json(args.output / 'trajectory.json', rows)
        write_json(args.output / 'trial.json', trial)
        z_values = [row['target'][2] for row in rows]
        print('initial', initial.tolist(), 'table edge/top', [edge, top], flush=True)
        result = {'trial_id':trial['trial_id'], 'approach':'tracking-cradle',
                  'target_mass_kg':float(model.body_mass[target]),
                  'controller': args.controller, 'startup_wbc': 'sonic' if native_start else 'gr00t',
                  'sonic_lockstep_steps': sonic.barrier.sim_step_seq if native_start and sonic else None,
                  'handover_pass': handover_pass, 'standing_pass': standing_pass,
                  'wrist_clearance_pass': wrist_clearance_pass,
                  'sonic_joints': [raw.task.robot.joint_names[i] for i in lower_motors]
                  if args.controller == 'sonic-split' else None,
                  'early_robot_contacts': sorted(early_contacts),
                  'hand_torque_limits_nm':hand_limits[raw.task.robot.right_hand_index - 1].tolist(),
                  'target_initial_height_m':float(z), 'max_lift_m':max(z_values, default=z)-z,
                  'lift_and_hold_pass':held_above_table(rows, z),
                  'carry_requested':args.carry, 'phases':list(dict.fromkeys(row['stage'] for row in rows)),
                  'steps':len(rows), 'task_success':official_success_step is not None,
                  'first_official_success_step':official_success_step,
                  'placement_pass':placed_in_container(rows, container_name),
                  'robot_stabilized':bool(raw.task.robot.stabilized)}
        write_json(args.output / 'result.json', result)
        print(json.dumps(result, default=lambda value: value.item()), flush=True)
        if frames:
            with mujoco.Renderer(model, height=480, width=640) as renderer, imageio.get_writer(
                args.output / 'overview.mp4', fps=25, codec='libx264'
            ) as video:
                for stage, qpos in frames:
                    data.qpos[:] = qpos
                    mujoco.mj_forward(model, data)
                    if stage in ('carry', 'release', 'settle'):
                        camera.lookat[:] = [qpos[0], qpos[1], .68]
                    renderer.update_scene(data, camera=camera)
                    video.append_data(renderer.render())
        raw.mujoco.renderers.clear()
        env.close()


if __name__ == '__main__':
    main()
