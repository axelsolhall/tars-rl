import math

import mujoco
import numpy as np
import pytest

from tars_rl.body import eval as body_eval
from tars_rl.body.mjcf import ASSETS, build_mjcf, load_model, load_spec
from tars_rl.body.spec import BodySpec, default_spec


def test_spec_json_roundtrip():
    spec = default_spec()
    assert BodySpec.from_json(spec.to_json()) == spec


def test_committed_assets_match_generator():
    spec = default_spec()
    assert (ASSETS / "tars_v1.json").read_text() == spec.to_json()
    assert (ASSETS / "tars_v1.xml").read_text() == build_mjcf(spec)


def test_wrong_version_rejected():
    text = default_spec().to_json().replace('"version": 1', '"version": 99')
    with pytest.raises(ValueError):
        BodySpec.from_json(text)


def test_defaults_follow_assumptions():
    spec = default_spec()
    assert 0.30 <= spec.central.height <= 0.40  # A7 printable scale
    assert spec.slide_range == pytest.approx((-0.25 * spec.arm.height, 0.25 * spec.arm.height))
    assert spec.actuator_names == [
        "left_hinge_act", "right_hinge_act", "left_slide_act", "right_slide_act"
    ]


def test_model_matches_spec():
    spec = load_spec()
    m = load_model(spec)
    assert m.nu == 4
    for i, (jn, an) in enumerate(zip(spec.joint_names, spec.actuator_names, strict=True)):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        assert j >= 0
        assert mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) == an
        assert m.actuator_trnid[i, 0] == j
        assert tuple(m.jnt_range[j]) == pytest.approx(spec.joint_range(jn))
    assert m.body_subtreemass[1] == pytest.approx(spec.total_mass)


def test_hinges_share_axis_through_centre():
    m = load_model()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    centre = d.xpos[1]
    for s in ("left", "right"):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_hinge")
        assert np.allclose(d.xaxis[j], [0, 1, 0])
        assert np.allclose(d.xanchor[j][[0, 2]], centre[[0, 2]])  # on the y line through centre


def test_slide_moves_arm_along_its_length_when_swung():
    spec = load_spec()
    m = load_model(spec)
    d = mujoco.MjData(m)
    b = body_eval.Body(spec, m)
    d.qpos[b.qadr("left_hinge")] = math.pi / 2
    mujoco.mj_forward(m, d)
    arm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_arm")
    p0 = d.xpos[arm].copy()
    d.qpos[b.qadr("left_slide")] = 0.05
    mujoco.mj_forward(m, d)
    move = d.xpos[arm] - p0
    assert np.allclose(np.abs(move), np.abs(0.05 * d.xmat[arm].reshape(3, 3)[:, 2]), atol=1e-9)
    assert abs(move[2]) < 1e-9  # arm horizontal, so the slide is horizontal too


def test_powered_stands_short():
    spec = load_spec()
    r = body_eval.stand(body_eval.Body(spec, load_model(spec)), load_model(spec), 2.0)
    assert not r["fell"]
    assert r["max_tilt_deg"] < 2.0


def test_mjx_loads_and_steps():
    from mujoco import mjx

    m = load_model()
    mx = mjx.put_model(m)
    dx = mjx.step(mx, mjx.make_data(mx))
    assert np.all(np.isfinite(np.asarray(dx.qpos)))
