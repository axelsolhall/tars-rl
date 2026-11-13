import numpy as np
from tars_scan.map_live_view import (
    EDGE_REJECTED,
    FAR,
    KEPT,
    LR_REJECTED,
    map_layers,
    pixel_classes,
)
from tars_scan.occupancy_map import Grid, Params

nan = np.nan


def test_pixel_classes_say_why_a_pixel_was_dropped():
    z_one = np.array([[1.0, 1.0, 1.0, 3.0, nan]])  # one-sided stereo
    z_raw = np.array([[1.0, nan, 1.0, 3.0, nan]])  # after the left-right check
    z = np.array([[1.0, nan, nan, nan, nan]])  # after the edge filter
    got = pixel_classes(z_one, z_raw, z, max_depth=2.0)
    assert got.tolist() == [[KEPT, LR_REJECTED, EDGE_REJECTED, FAR, 0]]
    assert pixel_classes(None, z_raw, z, 2.0)[0, 1] == 0  # no check: nothing to blame on it


def test_map_layers_split_obstacles_floaters_not_yet_and_cleared():
    p = Params(min_neighbors=2, subvoxel=False)
    g = Grid(np.zeros(3), np.ones(3) * 0.5, p)  # 10^3 voxels
    wall = g.index(np.array([[0.125, 0.125 + 0.05 * k, 0.225] for k in range(4)]))
    lone = g.index(np.array([[0.375, 0.375, 0.375]]))
    weak = g.index(np.array([[0.025, 0.025, 0.025]]))
    g.L[wall], g.L[lone], g.L[weak] = 3.0, 3.0, 1.0
    lay = map_layers(g, 0.0, np.zeros(0, np.int64))
    assert sorted(lay["obstacles"]) == sorted(wall[1:3])  # wall ends have one neighbour only
    assert set(lone) <= set(lay["floaters"])
    assert lay["not_yet"].tolist() == weak.tolist()
    g.L[wall[1]] = -1.0  # a ray saw through it
    assert map_layers(g, 0.0, lay["obstacles"])["cleared"].tolist() == [wall[1]]


def test_upright_K_matches_a_camera_turned_180_degrees():
    from tars_scan.map_live_view import upright_K

    K = np.array([[130.0, 0, 150.0], [0, 130.0, 110.0], [0, 0, 1]])
    c = np.array([0.3, -0.2, 1.5])  # a point in the camera frame
    u, v, _ = K @ c / c[2]
    up, vp, _ = upright_K(K, 320, 240, True) @ (np.diag([-1.0, -1, 1]) @ c) / c[2]
    assert np.allclose([up, vp], [319 - u, 239 - v])
