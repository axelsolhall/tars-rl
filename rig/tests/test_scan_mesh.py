"""ChangeGrid rules on synthetic rays (logic checks, not tuning): a camera at the origin looks
along +x at a wall at 2 m; a box face at 1 m sits in front of part of it."""

import numpy as np
from tars_scan.scan_mesh import ChangeGrid, ScanParams, look_weight
from tars_scan.scan_tune import SPACE, stepped

# the rules, not the tuned defaults: settings pinned here
P = ScanParams(remove_tau_s=0.0, sight_margin_m=0.12, remove_level=0.3, ever_free_level=1.0)
LO = np.array([-0.5, -1.0, -1.0])
DIMS = np.array([150, 100, 100])  # 2 cm voxels: 3 x 2 x 2 m


def rays(n=50, spread=0.2):
    y = np.linspace(-spread, spread, n)
    d = np.stack([np.ones(n), y, np.zeros(n)], 1)
    return d / np.linalg.norm(d, axis=1)[:, None]


def frame(g, dw, dist_x, t):
    """One keyframe: every ray measures a surface at x = dist_x. Returns (blocked, removed)."""
    o = np.zeros(3)
    r = dist_x / dw[:, 0]
    z = np.full(len(r), dist_x)
    w = look_weight(z, P)
    blocked, passed = g.step(o, dw, r, z, w, t)
    keep = ~blocked
    X = o + dw[keep] * r[keep, None]
    offs = np.arange(-P.trunc_m, P.trunc_m + 1e-9, P.voxel_m)
    S = (o + dw[keep, None, :] * (r[keep, None] - offs[None, :])[..., None]).reshape(-1, 3)
    sdf = np.tile(offs / P.trunc_m, keep.sum())
    moving = g.hits(X, w[keep], S, sdf, np.repeat(w[keep], len(offs)))
    return blocked, g.removals(passed), moving


def certain(g, x):
    i, _ = g.idx(np.array([[x, 0.0, 0.0]]))
    return float(g.cert[i[0]])


def test_seen_through_surface_is_removed_and_then_counts():
    g, dw = ChangeGrid(LO, DIMS, P, 275 * 0.142), rays()
    for t in range(10):  # the box at 1 m, seen for 10 keyframes (2.7 s): certain
        frame(g, dw, 1.0, t)
    assert certain(g, 1.0) > P.block_level
    blocked_first, _, _ = frame(g, dw, 2.0, 10)  # box gone: rays now reach the wall at 2 m
    assert blocked_first.all()  # first judged as seen-through junk...
    for t in range(11, 30):
        b, _, _ = frame(g, dw, 2.0, t)
    assert certain(g, 1.0) == 0  # ...but the box was removed by looking through it
    assert not b.any() and certain(g, 2.0) > 0  # and the wall behind now counts


def test_hidden_surface_stays():
    g, dw = ChangeGrid(LO, DIMS, P, 275 * 0.142), rays()
    for t in range(10):
        frame(g, dw, 2.0, t)  # the wall
    wall = certain(g, 2.0)
    for t in range(10, 40):
        frame(g, dw, 1.0, t)  # a box placed in front: the wall is hidden, not seen through
    assert certain(g, 2.0) == wall


def test_hit_wins_over_pass_in_same_frame():
    g, dw = ChangeGrid(LO, DIMS, P, 275 * 0.142), rays(n=100)
    for t in range(10):
        frame(g, dw, 2.0, t)
    for t in range(10, 40):  # half the rays: wrong matches 0.5 m behind the wall, every frame
        o, r_ok = np.zeros(3), 2.0 / dw[:, 0]
        r = np.where(np.arange(len(dw)) % 2 == 0, r_ok, r_ok + 0.5)
        z = r * dw[:, 0]
        w = look_weight(z, P)
        blocked, passed = g.step(o, dw, r, z, w, t)
        k = ~blocked
        X = o + dw[k] * r[k, None]
        g.hits(X, w[k], X, np.zeros(k.sum()), w[k])
        g.removals(passed)
    assert certain(g, 2.0) > P.block_level  # the wall survives the stray rays through it


def test_new_thing_in_empty_space_is_moving():
    g, dw = ChangeGrid(LO, DIMS, P, 275 * 0.142), rays()
    # empty space is as certain as the depth that saw through it: a wall at 1.5 m adds 0.2 per
    # look, so 30 keyframes (8 s) make the space in front ever-free (ever_free_level 1 s)
    for t in range(30):
        frame(g, dw, 1.5, t)
    _, _, moving = frame(g, dw, 0.8, 30)  # something appears at 0.8 m
    assert moving.mean() > 0.9


def test_tuner_steps_stay_in_range():
    p = {"power": 8.0, "draw_level": 0.5, "dropoff": True}
    assert stepped(p, "power", +1, 1.0) is None  # at the top of its range
    assert stepped(p, "power", -1, 1.0)["power"] == 7.0
    assert stepped(p, "draw_level", +1, 2.0)["draw_level"] == 1.0
    assert stepped(p, "dropoff", +1, 1)["dropoff"] is False
    assert set(SPACE) >= {"power", "remove_level", "draw_level"}


def test_drifted_wall_is_not_moving():
    g, dw = ChangeGrid(LO, DIMS, P, 275 * 0.142), rays()
    for t in range(30):
        frame(g, dw, 1.5, t)  # the wall, certain, and the space in front ever-free
    _, _, moving = frame(g, dw, 1.42, 30)  # the same wall seen 8 cm nearer (pose drift)
    assert moving.mean() < 0.1  # beside a solid surface: drift, not a moving thing
