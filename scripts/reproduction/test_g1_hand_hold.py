from types import MethodType, SimpleNamespace

import mujoco
import numpy as np
import pytest

from simple.core.action import ActionCmd
from simple.robots.g1_wholebody import G1Wholebody
from simple.robots.controllers.combo import WholeBodyEEFController


@pytest.mark.parametrize('replay', [False, True])
@pytest.mark.parametrize('refresh_hand_pd', [False, True])
def test_hand_feedback_cadence_and_replay_torque(monkeypatch, replay, refresh_hand_pd):
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep="0.002" gravity="0 0 0"/>
      <worldbody><body><joint name="finger" type="hinge"/>
        <inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>
        <geom type="sphere" size=".1"/></body></worldbody>
      <actuator><motor name="finger" joint="finger"/></actuator>
    </mujoco>''')
    data = mujoco.MjData(model)
    robot = SimpleNamespace(
        controller=WholeBodyEEFController(G1Wholebody.controller_cfg),
        amo_policy=SimpleNamespace(get_action=lambda *args: (np.empty(0), [])),
        joints={'finger': data.joint('finger')}, actuators={'finger': data.actuator('finger')},
        joints_names=['finger'], mjdata=data, mjmodel=model,
        stiffness=[5.], damping=[1.], torque_limits=[.7],
        target_hand_qpos={'finger': .05}, target_waist_qpos=None,
        refresh_hand_pd=refresh_hand_pd,
        pd_target=np.empty(0), output_joint_names=[], count=0,
        command=[0., 0., 0., -.3, 0., 0., 0., 0.],
    )
    robot.pd_control = MethodType(G1Wholebody.pd_control, robot)
    G1Wholebody.apply_action(robot, ActionCmd(
        'loco_command', command=[0., 0., 0., 0., 0., 0., 0., 0.],
        motion_type='height_adjust', keep_waist_pose=True))
    data.ctrl[0] = .123  # Last grasp torque must not be frozen during the lift.
    controls = []
    physics_step = mujoco.mj_step

    def observe(model, data, **kwargs):
        controls.append(float(data.ctrl[0]))
        physics_step(model, data, **kwargs)

    monkeypatch.setattr(mujoco, 'mj_step', observe)
    G1Wholebody.step(robot, robot.command, replay=replay)
    assert len(controls) == 10
    assert data.qpos[0] > 0
    if replay or not refresh_hand_pd:
        assert controls == [.123] * 10
    else:
        assert controls[0] == pytest.approx(.25)
        assert controls[-1] < controls[0]
        assert robot.target_hand_qpos == {'finger': .05}
        assert max(abs(x) for x in controls) <= .7
