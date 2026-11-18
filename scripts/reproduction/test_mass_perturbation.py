"""Real MuJoCo builds: mass/inertia scaling, unchanged contacts, and reset isolation."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from simple.assets.primitive import Box, Cube
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
    sim = MujocoSimulator(task, physics={'mass': {'scale': 2}})
    sim.upstream_object_gravity = True

    def build(config=None):
        if config is not None:
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

    configured, _ = build()
    baseline, ids = build(MassPerturbation())
    assert baseline['body_mass'][ids[0]] == pytest.approx(.1)
    for config in (MassPerturbation(False, 2), MassPerturbation(True, 1),
                   MassPerturbation(True, 2), MassPerturbation(True, 10), MassPerturbation(True, 20),
                   MassPerturbation(), MassPerturbation(True, 1)):
        current, current_ids = build(config)
        assert current_ids == ids
        factor = config.effective_scale
        for field, original in baseline.items():
            if factor == 2:
                np.testing.assert_array_equal(configured[field], current[field])
            expected = original.copy()
            if field in ('body_mass', 'body_inertia'):
                expected[ids[0]] *= factor
                np.testing.assert_allclose(current[field], expected, rtol=1e-12, atol=1e-15)
            elif mesh and field in ('body_ipos', 'body_iquat'):
                # Recompiling scaled mesh densities introduces floating-point
                # roundoff in derived COM/principal axes, not a geometry change.
                np.testing.assert_allclose(current[field], expected, rtol=1e-12, atol=1e-15)
            else:
                np.testing.assert_array_equal(current[field], expected)


def test_enabled_requires_a_target_before_scene_build():
    sim = MujocoSimulator(SimpleNamespace(metadata={}, layout=Layout()), physics={'enabled': True})
    with pytest.raises(ValueError, match='target'):
        sim.update_layout()


def test_articulated_mass_scales_only_the_chair_and_rebuilds_from_nominal(tmp_path):
    asset_path = tmp_path / 'chair.xml'
    asset_path.write_text("""
        <mujoco>
          <worldbody>
            <body name="articulate_base" pos="0 0 0.2">
              <freejoint name="articulate_floating_base"/>
              <geom name="seat" type="box" size=".1 .1 .04" mass="2"/>
              <body name="wheel" pos=".05 0 -.1">
                <joint name="articulate_joint_0" axis="0 1 0"/>
                <geom type="sphere" size=".04" mass=".5"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
    """)
    layout = Layout()
    layout.add_object('target', Cube(size=.04))
    layout.add_articulated_object(
        'articulated', SimpleNamespace(uid='chair', mjcf_path=str(asset_path), articulate_init_joint_qpos=None))
    task = SimpleNamespace(layout=layout, metadata={'need_gravity': True}, robot=NoControl())
    sim = MujocoSimulator(task)
    sim.update_layout(sonic_config={})
    baseline = sim.mjModel
    root = baseline.body('articulate_base').id
    chair = baseline.body_rootid == root
    assert np.count_nonzero(chair) == 2
    assert baseline.body_subtreemass[root] == pytest.approx(2.5)

    for scale in (1, 5, 10, 1):
        sim.mass_perturbation = MassPerturbation(True, scale, actor='articulated')
        sim.update_layout(sonic_config={})
        current = sim.mjModel
        assert current.body_subtreemass[root] == pytest.approx(2.5 * scale)
        np.testing.assert_array_equal(current.body_mass[chair], baseline.body_mass[chair] * scale)
        np.testing.assert_array_equal(current.body_inertia[chair], baseline.body_inertia[chair] * scale)
        np.testing.assert_array_equal(current.body_mass[~chair], baseline.body_mass[~chair])
        np.testing.assert_array_equal(current.body_inertia[~chair], baseline.body_inertia[~chair])
        for field in ('body_ipos', 'body_iquat', 'geom_friction', 'geom_size', 'jnt_dofadr', 'dof_damping'):
            np.testing.assert_array_equal(getattr(current, field), getattr(baseline, field))
