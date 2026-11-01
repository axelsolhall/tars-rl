"""Viz tests on a tiny stand-in body (one hinge + one slide), independent of the TARS body."""

from __future__ import annotations

import imageio.v2 as imageio
import mujoco
import numpy as np
import pytest

from tars_rl.viz.render import frame_indices, render_mp4
from tars_rl.viz.sample import scripted_sine
from tars_rl.viz.trajectory import Trajectory

STAND_IN = """
<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="0 0 0.05"/>
    <body name="central" pos="0 0 0.2">
      <freejoint/>
      <geom type="box" size=".05 .05 .2" mass="1"/>
      <body name="left_arm" pos="0 .1 0">
        <joint name="left_hinge" type="hinge" axis="0 1 0" range="-2 2"/>
        <joint name="left_slide" type="slide" axis="0 0 1" range="-.05 .05"/>
        <geom type="box" size=".05 .02 .2" mass=".3"/>
      </body>
    </body>
  </worldbody>
  <contact><exclude body1="central" body2="left_arm"/></contact>
  <actuator>
    <position name="left_hinge_act" joint="left_hinge" kp="8" ctrlrange="-2 2"/>
    <position name="left_slide_act" joint="left_slide" kp="3000" ctrlrange="-.05 .05"/>
  </actuator>
</mujoco>
"""


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(STAND_IN)


def test_trajectory_roundtrip(tmp_path):
    t = Trajectory(qpos=np.random.default_rng(0).normal(size=(7, 9)), dt=0.01, body_spec_version=1)
    path = t.save(tmp_path / "t.npz")
    back = Trajectory.load(path)
    np.testing.assert_array_equal(back.qpos, t.qpos)
    assert (back.dt, back.body_spec_version, back.version) == (0.01, 1, t.version)


def test_trajectory_rejects_other_version(tmp_path):
    path = tmp_path / "t.npz"
    np.savez(path, version=99, qpos=np.zeros((2, 3)), dt=0.01, body_spec_version=1)
    with pytest.raises(ValueError, match="version"):
        Trajectory.load(path)


def test_frame_indices_resample():
    t = Trajectory(qpos=np.zeros((400, 3)), dt=0.01, body_spec_version=1)  # 4 s
    idx = frame_indices(t, fps=30)
    assert len(idx) == 120
    assert idx[0] == 0 and idx[-1] <= 399
    assert np.all(np.diff(idx) > 0)


def test_scripted_sine_moves_hinge_and_slide(model):
    t = scripted_sine(model, body_spec_version=1, seconds=2.0)
    assert t.qpos.shape == (200, model.nq)
    hinge, slide = t.qpos[:, 7], t.qpos[:, 8]
    assert np.ptp(hinge) > 0.3
    assert np.ptp(slide) > 0.02


def test_render_mp4_frame_count(model, tmp_path):
    t = scripted_sine(model, body_spec_version=1, seconds=1.0)
    out = tmp_path / "s.mp4"
    s = render_mp4(model, t, out, fps=20, width=160, height=120)
    assert s.frames == s.expected_frames == 20
    with imageio.get_reader(out) as r:
        frames = [f for f in r]
    assert len(frames) == 20
    assert frames[0].shape == (120, 160, 3)
    assert frames[0].std() > 1  # not a blank image


def test_render_rejects_wrong_nq(model, tmp_path):
    t = Trajectory(qpos=np.zeros((5, model.nq + 1)), dt=0.01, body_spec_version=1)
    with pytest.raises(ValueError, match="nq"):
        render_mp4(model, t, tmp_path / "x.mp4")
