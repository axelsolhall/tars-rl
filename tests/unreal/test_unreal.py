import math

import numpy as np
import pytest

from tars_rl.body.mjcf import load_spec
from tars_rl.unreal import export
from tars_rl.unreal.compare import gap, run_metrics
from tars_rl.unreal.poses import (
    CSV_COLUMNS,
    Poses,
    from_unreal_csv,
    joints_from_poses,
    mj_to_ue,
    mujoco_reference,
    qrotate,
    to_unreal_csv,
    ue_to_mj,
)
from tars_rl.unreal.sequence import STAND_SECONDS, Sequence, default_sequence


@pytest.fixture(scope="module")
def spec():
    return load_spec()


@pytest.fixture(scope="module")
def run(spec):
    seq = default_sequence(spec)
    poses, q = mujoco_reference(spec, seq)
    return seq, poses, q


def test_sequence_json_roundtrip(spec):
    seq = default_sequence(spec)
    assert Sequence.from_json(seq.to_json()) == seq
    assert seq.times[1] == STAND_SECONDS and not np.any(seq.at(np.linspace(0, 10, 50)))


def test_sequence_check_rejects_bad_targets(spec):
    seq = default_sequence(spec)
    bad = Sequence(seq.version, seq.actuator_names, (0.0, 1.0), ((0.0,) * 4, (0.0, 0, 1.0, 0)))
    with pytest.raises(ValueError, match="outside"):
        bad.check(spec)
    renamed = Sequence(seq.version, ("a", "b", "c", "d"), seq.times, seq.targets)
    with pytest.raises(ValueError, match="actuators"):
        renamed.check(spec)


def test_committed_unreal_files_are_fresh(spec):
    for name, text in export.export_files(spec, default_sequence(spec)).items():
        assert (export.UNREAL_DATA / name).read_text() == text, (
            f"{name} is stale: run python -m tars_rl.unreal.export")


def test_mujoco_stands_through_whole_sequence(spec, run):
    seq, poses, _ = run
    r = run_metrics(spec, seq, poses)
    assert not r["fell"] and r["stand_s"] == pytest.approx(seq.duration)
    assert r["max_tilt_stand_deg"] < 1.0
    for e in r["tracking"].values():
        assert e["corr"] > 0.9


def test_joints_from_poses_match_qpos(spec, run):
    _, poses, q = run
    np.testing.assert_allclose(joints_from_poses(spec, poses), q, atol=1e-5)


def test_frame_conversion_roundtrip_and_mirror():
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(5, 3))
    quat = rng.normal(size=(5, 4))
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    p2, q2 = ue_to_mj(*mj_to_ue(pos, quat))
    np.testing.assert_allclose(p2, pos, atol=1e-12)
    np.testing.assert_allclose(q2, quat, atol=1e-12)
    # A yaw of +30 deg in MuJoCo (x towards +y_mj = left) turns x towards -y in Unreal.
    a = math.radians(30)
    q_mj = np.array([math.cos(a / 2), 0, 0, math.sin(a / 2)])
    _, q_ue = mj_to_ue(np.zeros(3), q_mj)
    x, y, z, w = q_ue
    x_ue = qrotate(np.array([w, x, y, z]), np.array([1.0, 0, 0]))
    np.testing.assert_allclose(x_ue, [math.cos(a), -math.sin(a), 0], atol=1e-12)


def test_unreal_csv_roundtrip_gives_zero_gap(spec, run, tmp_path):
    seq, poses, _ = run
    path = tmp_path / "unreal_poses.csv"
    to_unreal_csv(poses, path)
    assert path.read_text().splitlines()[0].split(",") == CSV_COLUMNS
    ue = from_unreal_csv(path)
    g = gap(poses, ue)
    assert g["seconds_compared"] == pytest.approx(seq.duration)
    for b in ("central", "left_arm", "right_arm"):
        assert g[b]["pos_max_m"] < 1e-5 and g[b]["rot_max_deg"] < 0.01
    assert run_metrics(spec, seq, ue)["stand_s"] == pytest.approx(seq.duration)


def test_gap_sees_an_offset_and_resamples(run):
    _, poses, _ = run
    t = poses.t[::3] + 0.001  # another engine's clock: coarser and shifted
    shifted = poses.resample(t)
    shifted = Poses(shifted.t, shifted.pos + np.array([0.01, 0, 0]), shifted.quat)
    g = gap(poses, shifted)
    assert g["central"]["pos_mean_m"] == pytest.approx(0.01, abs=1e-3)
