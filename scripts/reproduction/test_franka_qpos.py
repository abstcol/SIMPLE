from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from simple.robots.franka import FrankaMixin


def test_joint_dictionary_returns_scalar_positions_in_joint_order():
    robot = SimpleNamespace(joints={
        'joint2': SimpleNamespace(name='panda/joint2', qpos=np.array([-1.3])),
        'joint1': SimpleNamespace(name='joint1', qpos=np.array([0.25])),
        'finger_joint1': SimpleNamespace(name='finger_joint1', qpos=np.array([0.04])),
    })
    values = FrankaMixin.get_robot_qpos(robot)
    assert list(values) == ['joint2', 'joint1', 'finger_joint1']
    assert list(values.values()) == [-1.3, 0.25, 0.04]
    assert all(isinstance(value, float) for value in values.values())


@pytest.mark.parametrize('finger_prefix', ['finger_actuator', 'finger_joint'])
def test_control_setup_supports_both_asset_actuator_names(finger_prefix):
    joint_names = [f'joint{i}' for i in range(1, 8)] + ['finger_joint1', 'finger_joint2']
    actuator_names = [f'actuator{i}' for i in range(1, 8)] + [f'{finger_prefix}{i}' for i in (1, 2)]
    bodies = ''.join(f'<body><joint name="{name}"/><geom size=".01"/></body>' for name in joint_names)
    actuators = ''.join(f'<motor name="{act}" joint="{joint}"/>'
                        for act, joint in zip(actuator_names, joint_names))
    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><worldbody>{bodies}</worldbody><actuator>{actuators}</actuator></mujoco>')
    data = mujoco.MjData(model)
    initialized = []
    robot = SimpleNamespace(
        controller=SimpleNamespace(set_initial_qpos=lambda a, j: initialized.append((a, j))),
        joint_limits={},
    )
    joints, controls = FrankaMixin.setup_control(robot, data, model)
    assert list(joints) == joint_names
    assert [act.name for act in controls.values()] == actuator_names
    assert initialized == [(controls, joints)]
    assert set(robot.joint_limits) == set(joint_names)
