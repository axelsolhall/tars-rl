"""Viz tests on a tiny stand-in body (one hinge + one slide), independent of the TARS body."""

from __future__ import annotations

import imageio.v2 as imageio
import mujoco
import numpy as np
import pytest

from tars_rl.env.rollout import FlagRollout
from tars_rl.viz.flag import (
    arena_model,
    central_xy,
    flag_distances,
    framing_camera,
    render_flag_mp4,
)
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


# ---- flag arena (T16) ------------------------------------------------------------------------

FLAG_STAND_IN = STAND_IN.replace(
    "  </worldbody>",
    """    <light pos="0 0 5" dir="0 0 -1" directional="true"/>
    <body name="flag" mocap="true" pos="0 0 0">
      <geom type="box" size=".05 .05 .3" pos="0 0 .3" rgba="1 0 0 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>""",
)


@pytest.fixture(scope="module")
def flag_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(FLAG_STAND_IN)


def _red_mask(frame: np.ndarray) -> np.ndarray:
    r, g, b = (frame[..., i].astype(int) for i in range(3))
    return (r > 90) & (r > 2 * g) & (r > 2 * b)


def _red(frame: np.ndarray) -> int:
    return int(_red_mask(frame).sum())


def _red_col(frame: np.ndarray) -> float:
    return float(np.nonzero(_red_mask(frame).any(axis=0))[0].mean())


def test_flag_distances_match_central_xy(flag_model):
    t = scripted_sine(flag_model, body_spec_version=1, seconds=0.5)
    flag = np.array([2.0, -1.0, 0.0])
    d = flag_distances(flag_model, t.qpos, flag)
    xy = central_xy(flag_model, t.qpos)
    assert d.shape == (t.n_frames,)
    np.testing.assert_allclose(d, np.hypot(xy[:, 0] - 2.0, xy[:, 1] + 1.0))
    assert d[0] == pytest.approx(np.hypot(2.0, 1.0), abs=1e-6)  # body starts at the origin


@pytest.mark.parametrize("flag_xy", [(3.0, 0.0), (0.0, -4.0), (-2.0, 2.5)])
def test_render_flag_mp4_shows_flag_and_body(flag_model, tmp_path, flag_xy):
    """The flag is drawn where the rollout says, and stays in view next to the body."""
    t = scripted_sine(flag_model, body_spec_version=1, seconds=0.5)
    r = FlagRollout(t.qpos, t.dt, 1, np.array([*flag_xy, 0.0]))
    out = tmp_path / "f.mp4"
    s = render_flag_mp4(flag_model, r, out, fps=20, width=160, height=120)
    assert s.frames == s.expected_frames == 10
    with imageio.get_reader(out) as rd:
        frames = [f for f in rd]
    for f in (frames[0], frames[-1]):
        assert _red(f) > 5  # flag visible
        assert f.std() > 1

    # Without the flag position the mocap body sits at the origin, inside the body: the red
    # pixels move, which shows render_flag_mp4 really placed it.
    cam = framing_camera(flag_model, r.qpos, r.flag_pos, 160, 120)
    s2 = render_mp4(flag_model, t, tmp_path / "g.mp4", fps=20, width=160, height=120, camera=cam)
    with imageio.get_reader(tmp_path / "g.mp4") as rd:
        g0 = next(iter(rd))
    assert s2.frames == 10
    assert _red(g0) > 0
    assert abs(_red_col(frames[0]) - _red_col(g0)) > 5


def test_render_flag_rejects_model_without_flag(model, tmp_path):
    t = scripted_sine(model, body_spec_version=1, seconds=0.2)
    r = FlagRollout(t.qpos, t.dt, 1, np.zeros(3))
    with pytest.raises(ValueError, match="flag"):
        render_flag_mp4(model, r, tmp_path / "x.mp4")


def test_arena_model_has_flag_and_matches_rollout_nq():
    m, version = arena_model()
    assert m.nmocap == 1
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "flag") >= 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "central") >= 0
    assert version >= 1
