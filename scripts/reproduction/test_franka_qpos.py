from types import SimpleNamespace

import numpy as np

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
