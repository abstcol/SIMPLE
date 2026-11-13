"""G1 hand-gain ablation and read-only, control-rate contact diagnostics."""
import numpy as np

HAND_NAMES = [f'{side}_hand_{finger}_{index}_joint' for side in ('left', 'right')
              for finger, count in (('thumb', 3), ('index', 2), ('middle', 2)) for index in range(count)]
PRE_NOV9_COMMIT = 'b75c6786f87b4a7f43b52b0ac5b83a1d9ea62e11'


def configure_hands(robot, profile):
    if profile not in ('current', 'pre_nov9') or robot.uid != 'g1_wholebody':
        raise ValueError('Hand-gain profiles require the AMO G1 whole-body robot')
    if list(robot.hand_names) != HAND_NAMES:
        raise ValueError('Unexpected G1 hand joint ordering')
    indices = [robot.joints_names.index(name) for name in HAND_NAMES]
    if profile == 'pre_nov9':
        # Exact hand entries from PRE_NOV9_COMMIT, not a historical robot rollback.
        robot.stiffness[indices] = [80, 40, 40, 60, 40, 40, 40] * 2
        robot.torque_limits[indices] = [5, 5, 3, 5, 3, 5, 3] * 2
    return dict(profile=profile, historical_reference_commit=PRE_NOV9_COMMIT, joint_names=HAND_NAMES,
                stiffness=robot.stiffness[indices].copy(), damping=robot.damping[indices].copy(),
                software_torque_limits=robot.torque_limits[indices].copy(),
                applied_after_stabilization=True)


class HandDiagnostics:
    """Samples the last physics substep at 50 Hz; short contacts can be missed."""

    def __init__(self, robot, model, data, target_body):
        self.robot, self.model, self.data = robot, model, data
        self.names = list(robot.hand_names)
        self.indices = [robot.joints_names.index(n) for n in self.names]
        joints = [robot.joints[n].id for n in self.names]
        self.qadr, self.dadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        self.aids = np.array([robot.actuators[n].id for n in self.names])
        self.hand_side = np.zeros(model.nbody, dtype=int)
        self.is_target = np.zeros(model.nbody, dtype=bool)
        for body in range(model.nbody):
            parent = body
            while parent:
                name = model.body(parent).name or ''
                if name.startswith('left_hand_'):
                    self.hand_side[body] = 1
                elif name.startswith('right_hand_'):
                    self.hand_side[body] = 2
                if parent == target_body:
                    self.is_target[body] = True
                parent = model.body_parentid[parent]
        self.rows = []

    def metadata(self):
        m, ids = self.model, self.aids
        return dict(joint_names=self.names, sample_rate_hz=50,
                    sampling='end of control step; transient contacts may be missed',
                    actuator_ctrl_limited=m.actuator_ctrllimited[ids].copy(),
                    actuator_ctrl_range=m.actuator_ctrlrange[ids].copy(),
                    actuator_force_limited=m.actuator_forcelimited[ids].copy(),
                    actuator_force_range=m.actuator_forcerange[ids].copy(),
                    actuator_gear=m.actuator_gear[ids].copy())

    def sample(self, step):
        import mujoco
        m, d = self.model, self.data
        counts, normal = np.zeros(2, dtype=int), np.zeros(2)
        for index in range(d.ncon):
            contact = d.contact[index]
            a, b = m.geom_bodyid[[contact.geom1, contact.geom2]]
            side = self.hand_side[b] if self.is_target[a] else self.hand_side[a] if self.is_target[b] else 0
            if side:
                force = np.zeros(6)
                mujoco.mj_contactForce(m, d, index, force)
                counts[side-1] += 1
                normal[side-1] += max(0., force[0])
        ctrl = d.ctrl[self.aids].copy()
        row = dict(step=step, time=d.time, target=np.array([self.robot.target_hand_qpos[n] for n in self.names]),
                   qpos=d.qpos[self.qadr].copy(), qvel=d.qvel[self.dadr].copy(), ctrl=ctrl,
                   actuator_force=d.actuator_force[self.aids].copy(),
                   joint_actuator_force=d.qfrc_actuator[self.dadr].copy(),
                   software_saturated=np.abs(ctrl) >= self.robot.torque_limits[self.indices] - 1e-6,
                   target_contact_count=counts, target_contact_normal_force=normal)
        if any(not np.isfinite(value).all() for value in row.values()):
            raise FloatingPointError('Nonfinite hand diagnostic sample')
        self.rows.append(row)

    def save(self, path):
        if self.rows:
            np.savez_compressed(path, **{key: np.stack([r[key] for r in self.rows]) for key in self.rows[0]})
