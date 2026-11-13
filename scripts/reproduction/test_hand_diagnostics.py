from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from hand_diagnostics import HAND_NAMES, configure_hands, HandDiagnostics


@pytest.mark.parametrize('profile', ['current', 'pre_nov9'])
def test_profile_changes_only_named_hand_gains(profile):
    robot = SimpleNamespace(uid='g1_wholebody', hand_names=HAND_NAMES,
                            joints_names=['leg']+HAND_NAMES+['arm'],
                            stiffness=np.full(16, 2.5), torque_limits=np.full(16, .7),
                            damping=np.ones(16))
    result = configure_hands(robot, profile)
    np.testing.assert_array_equal(robot.stiffness[[0,-1]], [2.5,2.5])
    np.testing.assert_array_equal(robot.torque_limits[[0,-1]], [.7,.7])
    np.testing.assert_array_equal(robot.damping, np.ones(16))
    expected = [80,40,40,60,40,40,40]*2 if profile == 'pre_nov9' else [2.5]*14
    np.testing.assert_array_equal(result['stiffness'], expected)
    expected = [5,5,3,5,3,5,3]*2 if profile == 'pre_nov9' else [.7]*14
    np.testing.assert_array_equal(result['software_torque_limits'], expected)


@pytest.mark.parametrize('reverse', [False, True])
def test_contacts_and_actuator_clipping_are_read_without_advancing_physics(reverse, tmp_path):
    hand = '<body name="left_hand_palm"><joint name="finger"/><geom type="sphere" size=".02" mass="1"/></body>'
    target = '<body name="target" pos=".03 0 0"><freejoint/><geom type="sphere" size=".02" mass="1"/></body>'
    model = mujoco.MjModel.from_xml_string('<mujoco><option gravity="0 0 0"/><worldbody>'
        + (target+hand if reverse else hand+target)
        + '</worldbody><actuator><motor name="finger" joint="finger" forcelimited="true" forcerange="-.2 .2"/></actuator></mujoco>')
    data = mujoco.MjData(model)
    data.ctrl[0] = .3
    mujoco.mj_forward(model, data)
    robot = SimpleNamespace(hand_names=['finger'], joints_names=['finger'],
        joints={'finger':data.joint('finger')}, actuators={'finger':data.actuator('finger')},
        target_hand_qpos={'finger':.05}, torque_limits=np.array([.3]))
    recorder = HandDiagnostics(robot, model, data, model.body('target').id)
    qpos, qvel, time = data.qpos.copy(), data.qvel.copy(), data.time
    recorder.sample(1)
    row = recorder.rows[0]
    assert row['target_contact_count'].tolist() == [1,0]
    assert row['target_contact_normal_force'][0] > 0
    assert row['software_saturated'].tolist() == [True]
    assert row['actuator_force'][0] == pytest.approx(.2)
    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(data.qvel, qvel)
    assert data.time == time
    recorder.save(tmp_path/'hands.npz')
    assert np.load(tmp_path/'hands.npz')['qpos'].shape == (1,1)
