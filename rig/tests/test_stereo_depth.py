import numpy as np
from tars_scan.stereo_depth import board_truth


def test_board_truth_fronto_parallel_plane():
    # Camera looking straight at the board from 0.5 m: every board pixel has plane depth 0.5 m.
    K = np.array([[200.0, 0, 100], [0, 200.0, 80], [0, 0, 1]])
    rect = {"R1": np.eye(3), "P1": np.c_[K, np.zeros(3)]}
    R_bc = np.eye(3)  # board frame = camera frame rotation
    p_bc = np.array([0.1, 0.1, -0.5])  # camera 0.5 m in front of the board's z = 0 plane
    corners = np.array([[0.05, 0.05, 0], [0.15, 0.05, 0], [0.15, 0.15, 0], [0.05, 0.15, 0]])
    mask, z = board_truth(rect, R_bc, p_bc, corners, (160, 200))
    assert mask.sum() > 1000
    assert np.allclose(z[mask], 0.5)
    # The board's centre (0.1, 0.1) is straight ahead: principal point inside the mask.
    assert mask[80, 100]
