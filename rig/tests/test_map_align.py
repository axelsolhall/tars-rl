import numpy as np
from tars_scan.map_align import AlignParams, MapAligner
from tars_scan.occupancy_map import Grid, Params


def _corner(n=4000, seed=0):
    """Points on three perpendicular walls (a room corner): constrains all six DOF."""
    rng = np.random.default_rng(seed)
    a, b = rng.uniform(0.0, 1.0, (2, n))
    k = rng.integers(0, 3, n)
    pts = np.zeros((n, 3))
    for axis in range(3):
        m = k == axis
        others = [i for i in range(3) if i != axis]
        pts[m, others[0]], pts[m, others[1]] = a[m], b[m]
    return pts + np.array([1.0, 1.0, 1.0])


def test_aligner_pulls_a_shifted_scan_back_onto_the_map():
    p = Params(voxel_m=0.05, subvoxel=True, min_neighbors=0)
    g = Grid(np.zeros(3), np.full(3, 3.0), p)
    wall = _corner()
    for k in range(10):  # the map: the corner seen ten times
        g.update(g.index(wall), np.array([], np.int64), np.ones(len(wall)), wall, None, k * 0.2)
    shift = np.array([0.03, -0.02, 0.015])
    al = MapAligner(AlignParams(gain=1.0, max_step_m=0.05, max_dist_m=0.15, min_matches=200))
    scan = _corner(seed=1) + shift  # the same corner seen from a pose that drifted
    for k in range(10):
        al.step(g, 2.0, k, scan @ al.R.T + al.t)
    residual = np.linalg.norm((scan @ al.R.T + al.t).mean(0) - (scan - shift).mean(0))
    assert al.stats["accepted"] > 0 and residual < 0.008
