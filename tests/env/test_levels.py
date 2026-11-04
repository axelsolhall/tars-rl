import math

import mujoco
import numpy as np
import pytest

from tars_rl.env.arena import build_arena_mjcf, load_body_spec
from tars_rl.env.levels import LEVELS, add_to_mjcf, make_level, yaw_pitch_quat
from tars_rl.unreal.engines import MujocoEngine


def test_flat_is_the_training_arena():
    lv = make_level("flat")
    assert lv.boxes == () and lv.bounds is None
    spec = load_body_spec()
    assert add_to_mjcf(build_arena_mjcf(spec), lv) == build_arena_mjcf(spec)


@pytest.mark.parametrize("name", sorted(LEVELS))
def test_levels_are_reproducible_and_compile_in_mujoco(name):
    a, b = make_level(name, 3), make_level(name, 3)
    assert a.boxes == b.boxes
    m = mujoco.MjModel.from_xml_string(add_to_mjcf(build_arena_mjcf(load_body_spec()), a))
    assert m.ngeom >= len(a.boxes)
    assert a.ground_z(a.spawn_xy) == 0.0, "TARS starts on the bare floor"


@pytest.mark.parametrize("name", ["bumpy", "steps", "slopes", "apartment"])
def test_tars_stands_still_at_its_start_on_every_level(name):
    spec = load_body_spec()
    eng = MujocoEngine(spec, 0.02)
    lv = make_level(name)
    eng.load(lv)
    s = eng.reset(np.asarray(lv.spawn_xy), np.array([5.0, 5.0, 0.0]))
    for _ in range(100):  # 2 s at neutral
        s = eng.step(np.zeros(4))
    assert s.cos_tilt() > 0.99


def test_ramps_climb_continuously_onto_their_plateau():
    lv = make_level("slopes")
    plateau = lv.boxes[1]
    c = np.asarray(plateau.centre[:2])
    d = plateau.rot()[:2, 0] / np.linalg.norm(plateau.rot()[:2, 0])
    z = np.array([lv.ground_z(c + t * d) for t in np.linspace(-2.5, 2.5, 201)])
    assert np.abs(np.diff(z)).max() < 0.01, "no step anywhere along a mound"
    assert z.max() == pytest.approx(plateau.centre[2] + plateau.size[2] / 2, abs=1e-6)


def test_flags_need_a_clear_straight_walk():
    lv = make_level("apartment")
    assert lv.flag_ok((-2.6, -0.8), (-3.5, -1.2))  # open floor in the living room
    assert not lv.flag_ok((-2.0, -1.5), (2.0, -1.5)), "straight through the interior wall"
    assert lv.flag_ok((-0.6, 0.85), (0.6, 0.85)), "through the door"
    assert not lv.flag_ok((-2.6, -0.8), (10.0, 0.0)), "outside the flat"


def test_a_table_top_is_neither_floor_nor_wall():
    lv = make_level("apartment")
    top = next(b for b in lv.boxes if b.kind == "furniture" and b.bottom_z > 0.7)
    assert not top.solid
    assert lv.ground_z(top.centre[:2]) == 0.0, "TARS under the table walks on the floor"


def test_yaw_pitch_quat_turns_then_tilts():
    from tars_rl.env.levels import quat_to_mat

    r = quat_to_mat(np.asarray(yaw_pitch_quat(math.pi / 2, -0.1)))
    assert np.allclose(r[:, 0], [0.0, math.cos(0.1), math.sin(0.1)], atol=1e-9)
