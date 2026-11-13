from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from simple.engines.mujoco import MujocoSimulator


@pytest.mark.parametrize('need_gravity', [False, True])
@pytest.mark.parametrize('is_container', [False, True])
@pytest.mark.parametrize('upstream', [False, True])
def test_object_acceleration_matches_selected_gravity_mode(need_gravity, is_container, upstream):
    mass = 1.0 if is_container else .1
    gravity = -9.81 if need_gravity else 0
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option gravity="0 0 {gravity}" timestep=".002"/>
      <worldbody><body name="object" pos="0 0 1"><freejoint/>
        <geom type="sphere" size=".02" mass="{mass}"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    MujocoSimulator._apply_object_gravity(
        SimpleNamespace(need_gravity=need_gravity, upstream_object_gravity=upstream),
        data.body('object'), is_container)
    if need_gravity and not upstream:
        np.testing.assert_array_equal(data.body('object').xfrc_applied, np.zeros(6))
    mujoco.mj_step(model, data)
    assert data.qvel[2] / model.opt.timestep == pytest.approx(-9.81 * (2 if need_gravity and upstream else 1))
