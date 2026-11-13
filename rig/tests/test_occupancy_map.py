import numpy as np
from tars_scan.occupancy_map import Grid, Params, filter_depth, inject, ray_updates

# the behaviour tests pin the parameters they were written for: tuning defaults cannot break them
P = Params(
    voxel_m=0.03,
    ray_every=4,
    max_depth_m=2.5,
    l_miss=-0.4,
    l_occ=2.0,
    lr_max_diff=-1.0,
    l_miss_occ=0.0,
    min_neighbors=0,
    carve_stride=1,
    align_every=0,
    subvoxel=False,
)


def _grid():
    return Grid(np.array([-1.0, -1.0, -1.0]), np.array([3.0, 1.0, 1.0]), P)


def _wall_rays(n=200):
    """Rays from the origin fanning out to a wall at x = 2 m."""
    rng = np.random.default_rng(0)
    d = np.c_[np.ones(n), rng.uniform(-0.3, 0.3, (n, 2))]
    d /= np.linalg.norm(d, axis=1)[:, None]
    return d, 2.0 / d[:, 0]


def _fuse(grid, r_of_frame, frames, t0=0.0, dt=0.2):
    d, r_wall = _wall_rays()
    for k in range(frames):
        r = r_of_frame(d, r_wall)
        grid.update(*ray_updates(grid, np.zeros(3), d, r, np.full(len(r), 0.01)), t0 + k * dt)
    return t0 + (frames - 1) * dt


def _occupied_x(grid, now):
    return grid.centers(grid.occupied(now))[:, 0]


def test_wall_needs_several_frames():
    g = _grid()
    _fuse(g, lambda d, r: r, 2)
    assert len(g.occupied(0.2)) == 0  # two hits are not enough (l_occ = 2.0)
    now = _fuse(g, lambda d, r: r, 3, t0=0.4)
    x = _occupied_x(g, now)
    assert len(x) and np.all(np.abs(x - 2.0) < 0.05)  # only the wall, nothing in front of it


def test_person_walking_through_disappears():
    g = _grid()
    box = (np.array([0.9, -0.5, -0.5]), np.array([1.1, 0.5, 0.5]))
    _fuse(g, lambda d, r: r, 10)
    now = _fuse(g, lambda d, r: inject(np.zeros(3), d, r, [box]), 6, t0=2.0)
    assert np.any(np.abs(_occupied_x(g, now) - 1.0) < 0.1)  # there while it stood there
    now = _fuse(g, lambda d, r: r, 15, t0=3.2)
    x = _occupied_x(g, now)
    assert not np.any(np.abs(x - 1.0) < 0.1) and np.any(np.abs(x - 2.0) < 0.05)


def test_unseen_uncertain_voxels_fade_faster_than_walls():
    g = _grid()
    _fuse(g, lambda d, r: r, 40)  # wall: 40 hits
    weak = g.index(np.array([[0.5, 0.0, 0.0]]))
    for k in range(3):
        g.update(weak, np.array([], np.int64), np.ones(1), None, None, 8.0 + k * 0.2)
    wall = np.array([np.argmax(g.hits)])
    later = 8.4 + 60.0
    assert g.logodds(later, wall)[0] > P.l_occ  # a wall a minute out of view stays
    assert g.logodds(later, weak)[0] < P.l_occ  # three sightings a minute ago do not


def test_edge_pixels_are_dropped():
    z = np.full((20, 20), 1.0)
    z[:, 10:] = 2.0
    f = filter_depth(z, P)
    assert np.isnan(f[:, 9:11]).all() and np.isfinite(f[:, :8]).all()


def test_board_in_map_measures_offset_and_ghosts():
    from tars_scan.occupancy_map import board_in_map

    g = _grid()
    T = np.eye(4)
    T[:3, 3] = [1.0, 0.0, 0.0]
    T[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]  # board plane facing the camera, x = 1 m
    yy, zz = np.meshgrid(np.arange(-0.2, 0.2, 0.03), np.arange(-0.2, 0.2, 0.03))
    board = np.c_[np.full(yy.size, 1.01), yy.ravel(), zz.ravel()]  # 1 cm behind the plane
    ghost = board.copy()
    ghost[:, 0] = 1.07
    mid = ghost[(np.abs(ghost[:, 1]) < 0.1) & (np.abs(ghost[:, 2]) < 0.1)][:10]  # inside the area
    idx = np.unique(g.index(np.r_[board, mid]))
    r = board_in_map(g, idx, T, (np.array([-0.15, -0.15]), np.array([0.15, 0.15])))
    assert abs(abs(r["offset_mm"]) - 10) < 25 and 0 < r["ghost_share"] < 0.2 and r["coverage"] > 0.5


def _frames_to_clear_door(p: Params) -> int:
    """A door seen closed for 50 frames, then open (the wall behind shows): frames until it goes."""
    g = Grid(np.array([-1.0, -1.0, -1.0]), np.array([3.0, 1.0, 1.0]), p)
    d, r_wall = _wall_rays()
    door = (np.array([0.97, -0.5, -0.5]), np.array([1.03, 0.5, 0.5]))
    for k in range(50):
        r = inject(np.zeros(3), d, r_wall, [door])
        g.update(*ray_updates(g, np.zeros(3), d, r, np.full(len(r), 0.01)), k * 0.2)
    for k in range(1, 60):
        g.update(*ray_updates(g, np.zeros(3), d, r_wall, np.full(len(r_wall), 0.01)), 10 + k * 0.2)
        if not np.any(np.abs(_occupied_x(g, 10 + k * 0.2) - 1.0) < 0.1):
            return k
    return 99


def test_change_evidence_clears_an_opened_door_faster():
    from dataclasses import replace

    slow = _frames_to_clear_door(P)
    fast = _frames_to_clear_door(replace(P, l_miss_occ=-1.5, reset_hits_when_free=True))
    assert fast < slow and fast <= 3


def test_subvoxel_surface_sits_on_the_wall_not_the_voxel_centre():
    from dataclasses import replace

    g = Grid(
        np.array([-1.0, -1.0, -1.0]),
        np.array([3.0, 1.0, 1.0]),
        replace(P, voxel_m=0.05, subvoxel=True),
    )
    d, r = _wall_rays()
    r = r * 2.01 / 2.0  # wall at x = 2.01: voxel centres there are at 2.025, 1.6 cm off
    for k in range(10):
        g.update(*ray_updates(g, np.zeros(3), d, r, np.full(len(r), 0.01)), k * 0.2)
    occ = g.occupied(1.8)
    assert len(occ) and np.all(np.abs(g.centers(occ)[:, 0] - 2.01) > 0.01)
    assert np.all(np.abs(g.surface(occ)[:, 0] - 2.01) < 0.002)


def test_through_gate_still_clears_an_opened_door():
    from dataclasses import replace

    gated = replace(P, l_miss_occ=-1.5, subvoxel=True, miss_occ_through=0.7)
    assert _frames_to_clear_door(gated) < _frames_to_clear_door(P)


def test_time_gate_spares_a_voxel_that_is_still_being_hit():
    """A voxel hit by one ray and passed through by another in the same frames (a partly filled
    voxel) stays with the time gate; without it the strong miss wipes it."""
    from dataclasses import replace

    def survives(p):
        g = Grid(np.array([-1.0, -1.0, -1.0]), np.array([3.0, 1.0, 1.0]), p)
        v = g.index(np.array([[1.0, 0.0, 0.0]]))
        for k in range(30):
            now = k * 0.2
            g.update(v, np.array([], np.int64), np.ones(1), None, None, now)
            if k >= 10:  # from here on another ray also passes through its empty part each frame
                g.update(np.array([], np.int64), v, np.ones(0), None, None, now + 0.1)
        return len(g.occupied(now + 0.1)) == 1

    strong = replace(P, l_miss_occ=-1.5)
    assert not survives(strong) and survives(replace(strong, miss_occ_after_s=1.0))


def test_min_neighbors_drops_a_lone_voxel_but_keeps_the_wall():
    from dataclasses import replace

    g = Grid(np.array([-1.0, -1.0, -1.0]), np.array([3.0, 1.0, 1.0]), replace(P, min_neighbors=3))
    _fuse(g, lambda d, r: r, 10)
    lone = g.index(np.array([[0.5, 0.0, 0.0]]))
    for k in range(5):
        g.update(lone, np.array([], np.int64), np.ones(1), None, None, 2.0 + k * 0.2)
    occ = g.occupied(2.8)
    assert lone[0] not in occ and np.all(np.abs(g.centers(occ)[:, 0] - 2.0) < 0.05) and len(occ)


def test_far_rays_clear_a_door_that_opened_onto_open_space():
    """Behind the opened door nothing is within max range: only far-ray carving can clear it."""
    from dataclasses import replace

    p = replace(P, carve_far=True)
    z = np.array([[np.nan, 5.0, 1.0]], np.float32)
    assert np.isposinf(filter_depth(np.repeat(np.repeat(z, 5, 0), 5, 1), p)[2, 7])  # far, not edge
    g = Grid(np.array([-1.0, -1.0, -1.0]), np.array([3.0, 1.0, 1.0]), p)
    d, r_wall = _wall_rays()
    door = (np.array([0.97, -0.5, -0.5]), np.array([1.03, 0.5, 0.5]))
    sig = np.full(len(d), 0.01)
    for k in range(50):
        g.update(
            *ray_updates(g, np.zeros(3), d, inject(np.zeros(3), d, r_wall, [door]), sig), k * 0.2
        )
    assert np.any(np.abs(_occupied_x(g, 9.8) - 1.0) < 0.1)
    r_far = p.max_depth_m / d[:, 0]  # opened onto open space: every ray reads "far"
    for k in range(10):
        upd = ray_updates(g, np.zeros(3), d, r_far, sig, np.ones(len(d), bool))
        g.update(*upd, 10 + k * 0.2)
    assert not np.any(np.abs(_occupied_x(g, 11.8) - 1.0) < 0.1)


def test_floor_height_finds_the_lowest_big_layer_not_the_table():
    from tars_scan.occupancy_map import floor_height

    rng = np.random.default_rng(0)
    floor = rng.normal(-1.0, 0.01, 3000)
    table = rng.normal(-0.6, 0.01, 4000)  # more voxels than the floor, but higher
    walls = rng.uniform(-1.0, 0.8, 3000)
    z = np.concatenate([floor, table, walls])
    assert abs(floor_height(z, 0.0) - -1.0) < 0.05
