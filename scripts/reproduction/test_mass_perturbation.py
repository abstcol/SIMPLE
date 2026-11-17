"""Real MuJoCo builds: mass/inertia scaling, unchanged contacts, and reset isolation."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from simple.assets.primitive import Box, Cube
from simple.core.actor import ObjectActor
from simple.core.layout import Layout
from simple.dr.physics import MassPerturbation
from simple.engines.mujoco import MujocoSimulator
from simple.robots.protocols import Controllable


class NoControl(Controllable):
    def setup_control(self, *args, **kwargs):
        return {}, {}


@pytest.mark.parametrize('config', [dict(enabled=1), dict(mass_scale=True), dict(mass_scale=0),
    dict(mass_scale=-1), dict(mass_scale=float('inf')), dict(mass_scale=float('nan'))])
def test_invalid_physics_rejected(config):
    with pytest.raises(ValueError):
        MassPerturbation(**config)


@pytest.mark.parametrize('mesh', [False, True])
def test_build_and_repeated_reset_preserve_everything_except_target_mass_and_inertia(tmp_path, mesh):
    if mesh:
        files = []
        for i in range(2):
            path = tmp_path / f'part{i}.obj'
            x = i * .05
            path.write_text(f'v {x} 0 0\nv {x+.02} 0 0\nv {x} .03 0\nv {x} 0 .04\n'
                            'f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n')
            files.append(str(path))
        asset = SimpleNamespace(uid='target_asset', collision_meshes_mujoco=files)
    else:
        asset = Cube(size=[.02, .03, .04])
    layout = Layout()
    layout.add_object('target', asset)
    layout.add_object('distractor', SimpleNamespace(uid='other', collision_meshes_mujoco=files)
                      if mesh else Cube(size=.03))
    # Primitive UIDs are identical by default; use distinct body/mesh names.
    if not mesh:
        layout.actors['distractor'].asset.uid = 'other'
        layout.actors['distractor'].asset.label = 'other'
    layout.add_primitive('table', Box([1, 1, .1], [0, 0, -.05], [1, 0, 0, 0]))
    task = SimpleNamespace(layout=layout, metadata={'need_gravity': True}, robot=NoControl())
    sim = MujocoSimulator(task)
    sim.upstream_object_gravity = True

    def build(config):
        sim.mass_perturbation = config
        sim.update_layout(sonic_config={})  # No robot assets, renderer or GPU needed.
        m, d = sim.mjModel, sim.mjData
        ids = [sim.mj_objects[k].id for k in ('target', 'distractor')]
        values = {key: getattr(m, key).copy() for key in (
            'body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'body_pos', 'qpos0',
            'geom_friction', 'geom_solref', 'geom_solimp', 'geom_condim', 'geom_priority', 'geom_size')}
        force = d.body(ids[0]).xfrc_applied.copy()
        np.testing.assert_array_equal(force, [0, 0, -.9810000000000001, 0, 0, 0])
        return values, ids

    baseline, ids = build(MassPerturbation())
    assert baseline['body_mass'][ids[0]] == pytest.approx(.1)
    for config in (MassPerturbation(False, 2), MassPerturbation(True, 1),
                   MassPerturbation(True, 2), MassPerturbation(True, 10), MassPerturbation(True, 20),
                   MassPerturbation(), MassPerturbation(True, 1)):
        current, current_ids = build(config)
        assert current_ids == ids
        factor = config.effective_scale
        for field, original in baseline.items():
            expected = original.copy()
            if field in ('body_mass', 'body_inertia'):
                expected[ids[0]] *= factor
                np.testing.assert_allclose(current[field], expected, rtol=1e-12, atol=1e-15)
            else:
                np.testing.assert_array_equal(current[field], expected)


def test_enabled_requires_a_target_before_scene_build():
    sim = MujocoSimulator(SimpleNamespace(metadata={}, layout=Layout()), physics={'enabled': True})
    with pytest.raises(ValueError, match='target'):
        sim.update_layout()
