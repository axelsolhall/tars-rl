"""tars-scan: what happens to a rig session on the Mac, one module per step.

export (session -> EuRoC layout that Basalt, OpenVINS, ORB-SLAM3 and Kalibr read), select (rank
frames by how much they are worth and pick a calibration subset, most valuable first), and their
evals. Reads only the session contract (tars_rig.contracts); never runs on the Pi.
"""
