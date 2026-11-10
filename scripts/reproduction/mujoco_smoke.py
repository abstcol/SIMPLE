"""Test MuJoCo physics and EGL, without claiming a SIMPLE task result."""
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image

out = Path(sys.argv[1])
model = mujoco.MjModel.from_xml_string('''
<mujoco><option timestep="0.002"/>
<worldbody><light pos="0 0 3"/>
<geom type="plane" size="2 2 .1" rgba=".3 .4 .5 1"/>
<body pos="0 0 1"><freejoint/>
<geom type="sphere" size=".1" mass="1" rgba=".8 .2 .1 1"/></body>
</worldbody></mujoco>''')
data = mujoco.MjData(model)
for _ in range(500):
    mujoco.mj_step(model, data)
assert np.isfinite(data.qpos).all(), data.qpos
assert 0.08 < data.qpos[2] < 0.12, data.qpos
physics = {"mujoco_version": mujoco.__version__, "steps": 500,
           "simulated_seconds": data.time, "resting_height": float(data.qpos[2]),
           "scope": "synthetic falling sphere, NOT a SIMPLE robot task"}
(out / "mujoco_physics.json").write_text(json.dumps(physics, indent=2) + "\n")
print(json.dumps(physics), flush=True)
camera = mujoco.MjvCamera()
camera.lookat[:] = [0, 0, .2]
camera.distance = 2
camera.azimuth = 90
camera.elevation = -25
with mujoco.Renderer(model, height=240, width=320) as renderer:
    renderer.update_scene(data, camera=camera)
    frame = renderer.render()
    assert frame.shape == (240, 320, 3)
    assert float(frame.std()) > 1, "Empty render"
    Image.fromarray(frame).save(out / "mujoco_smoke.png")
    print(json.dumps({"render": "passed", "pixel_std": float(frame.std())}))
