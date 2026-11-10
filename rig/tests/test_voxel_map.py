import numpy as np
from scipy.spatial.transform import Rotation
from tars_scan.voxel_map import frame_points


def test_frame_points_undo_rectification():
    # A flat depth of 1 m in the rectified view, with rectification turned 10 deg about y: the
    # points come back in camera 0 coordinates, i.e. turned back by the rectifying rotation.
    K = np.array([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]])
    R1 = Rotation.from_euler("y", 10, degrees=True).as_matrix()
    rect = {"P1": np.c_[K, np.zeros(3)], "R1": R1}
    z = np.full((80, 100), 1.0)
    pts = frame_points(rect, z, every=1)
    centre = pts[40 * 100 + 50]  # principal point: straight ahead in the rectified view
    assert np.allclose(R1 @ centre, [0, 0, 1.0])
    assert np.allclose((pts @ R1.T)[:, 2], 1.0)  # rectified depth is 1 m everywhere
