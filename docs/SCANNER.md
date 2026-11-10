# The 3D scanner: research, what was measured, and the plan

The rig: Raspberry Pi 5, two Waveshare IMX219 (120°, rolling shutter, software-synced), an
MPU6050. The goal: a map that stays accurate over tens of seconds, built with as little compute as
possible, so it can later run on TARS. This file covers three things: what the 2025–2026 work says
for this exact kind of rig, how Axel's two ideas ("item memories" and "most valuable first") line
up with that work, and what was built and measured on 2026-11-08.

## 1. The state of the art for this rig (searched 2026-11-08)

**The closest benchmark** is the Hilti × Trimble SLAM Challenge 2026. It uses a handheld rig with
dual fisheye rolling-shutter cameras and an IMU, which is our sensor set. Winners were announced at
ICRA 2026, 5 June ([repo](https://github.com/Hilti-Research/hilti-trimble-slam-challenge-2026),
[leaderboard](https://hilti-trimble-challenge.com/leaderboard-2026)).

The winner, ACDC-VSLAM ([report](https://hilti-trimble-challenge.com/assets/2026/leaderboard_slam/1/report.pdf)),
is a classical keyframe visual-inertial SLAM built on OKVIS2-X
([T-RO 2025](https://arxiv.org/abs/2510.04612), [code](https://github.com/ethz-mrl/OKVIS2-X)).
What it added:

- **Spherical equal-area feature bucketing.** Each keypoint is lifted to its viewing ray. The
  sphere of rays is cut into equal-area cells, and each cell keeps only its strongest keypoint.
  Pixel grids over-sample the centre of a fisheye image, and this avoids that.
- **Reweighted bundle adjustment.** Long-lived tracks and near points count more. Short tracks and
  far, low-parallax points count less.
- **Line features** for white walls and other textureless areas.
- **Adaptive loop closure.** For each loop it picks the correction model (SE(3), Sim(3) or 4-DoF)
  that the evidence supports.

Other top entries used OpenVINS-style filters and VINS. None of the reports I read models the
rolling shutter explicitly; good timestamps and robust weighting carried them.

**Feed-forward 3D models** are the other big 2025–2026 line: MASt3R-SLAM
([CVPR 2025](https://www.semanticscholar.org/paper/MASt3R-SLAM:-Real-Time-Dense-SLAM-with-3D-Priors-Murai-Dexheimer/0022d6667d72a07ddcc29c0f2e22e1068df6d58c), 15 fps) and
VGGT-SLAM 2.0 ([2026](https://arxiv.org/html/2601.19887), about 7 fps). They give dense maps
from a single uncalibrated camera, but they need a large GPU and use no IMU. They don't suit
TARS's Pi. They could make a dense map offline on the Mac later.

Supporting pieces:

- **Features.** XFeat runs in real time on a laptop CPU and is up to 5× faster than comparable
  learned features ([CVPR 2024](https://arxiv.org/abs/2404.19174v1)). LightGlue matching is
  heavier.
- **Place recognition.** MegaLoc ([CVPR 2025](https://arxiv.org/abs/2502.17237v1)) is the current
  general retrieval model for loop closure and relocalisation.
- **Rolling-shutter line delay.** A Kalibr fork estimates it together with the camera-IMU
  calibration ([Huai et al.](https://arxiv.org/abs/2108.07200)). Basalt-based tools still call
  this experimental ([OpenImuCameraCalibrator](https://github.com/urbste/OpenImuCameraCalibrator/)).
  The rig now reads the line time straight from the sensor instead (section 3).
- **Software sync.** Raspberry Pi expects frame starts within "several tens of microseconds" for
  identical cameras on one Pi ([docs PR](https://github.com/raspberrypi/documentation/pull/4049/files)).
  `check.json` measures it per session.
- **Efficiency.** Among real-time stereo VIO systems, Basalt was the most efficiently
  parallelised, with the lowest per-frame time in the HybVIO comparison. HybVIO itself ran on a
  Raspberry Pi ([WACV 2022](https://arxiv.org/pdf/2106.11857)).

## 2. Axel's two ideas, against that work

**"Item memories": see the same item from several angles to sharpen the map, and restart from an
item seen before when unsure.** Keyframe SLAM already does this with points:

- Every landmark keeps all its observations, and bundle adjustment uses all of them, so more
  angles mean a more accurate landmark.
- Relocalisation looks up the known place that best matches the current view and resumes from it.
  ACDC's reweighting is the "trust items seen many times" half, made explicit.

The object-level version of the idea is OpenReLoc ([2026](https://arxiv.org/html/2606.24767)):

- **The map.** Objects with CLIP descriptors from five views each, joined by a scene graph.
- **The result.** On ScanNet it relocalises within 50 cm in 64–89% of cases, against 8–25% for the
  point-based GoReloc.
- **The catch.** It takes about 5 s per frame on an RTX 4090 including a GPT call, and it is
  accurate to about 10–20 cm, not millimetres.

So objects are a strong *recovery* memory but too coarse to make the map more accurate. The plan:
points for accuracy, retrieval for recovery, and an object layer later, where it also serves TARS's
"find a toy" goal.

**"Some parts of the image deserve more attention; go from highest value to lowest."** This is
well supported, with "value" meaning how much a feature pins down the pose:

- **Good Feature Selection / Good Feature Matching** (Zhao & Vela,
  [2019](https://arxiv.org/pdf/1905.07807), [T-RO 2020](https://arxiv.org/abs/2001.00714)). Pick
  the subset of features that maximises log det of the pose information. A lazier-greedy pass keeps
  that cheap. The result is the same accuracy with lower latency.
- **"Balancing the Budget"** ([Zhang et al.](https://arxiv.org/abs/2109.05975)). Applied to
  multi-camera VIO with a fixed feature budget, it keeps accuracy and cuts back-end time.

Saliency in the visual sense, as in Attention-SLAM ([2020](https://arxiv.org/abs/2009.06886)), has
weaker evidence. The geometric value is the one to build on.

## 3. What was built and measured (2026-11-08)

**Rig v2** (session schema 2, `rig/src/tars_rig/contracts.py`):

- **One frame file per camera.** Frames go into `frames.mjpeg` with byte offsets in `frames.csv`.
  This replaces 36,000 small files per camera per 20 minutes and cuts SD-card metadata writes.
  `rsync` copies one file, and `ffplay` plays it.
- **Capture is decoupled from encoding.** A bounded queue sits between them, so a slow encode or a
  write stall costs a counted drop instead of stalling libcamera.
- **`check.json` reports what the Pi spent:** encoder drops, encode time, KB per frame and MB/s
  written.
- **Rolling-shutter line time** is read from the sensor's V4L2 controls and stored per camera.
- **Calibration sessions record the IMU too.** It is used for frame values and for camera-IMU
  calibration.

**`tars_scan.export`** writes a session in the EuRoC layout, which Basalt, OpenVINS, ORB-SLAM3 and
Kalibr's bag tools read:

- **Camera timestamps** are at mid-exposure of the middle row.
- **IMU timestamps** come from the refitted clock, one fit per stretch between FIFO overflows.
- **Stereo pairs** are matched by nearest frame, and the offsets are reported.
- **Reduced size** (1/2, 1/4, 1/8) is decoded in the JPEG's DCT domain, for cheaper SLAM runs.

**`tars_scan.select` + `select_eval`** test "most valuable first" on calibration:

- **Value.** Each frame's value is its predicted smear in pixels, from the gyro and the exposure
  plus rolling-shutter readout. No image decoding is needed for this.
- **Search.** Board search runs on frames in value order.
- **Pick.** A greedy Max-logDet step picks the subset that best constrains the lens.
- **Test.** The simulation is a 98 s move-pause film at the real resolution, with exposure,
  readout, gyro noise and bias, and a 9×6-corner board. Results are in section 4.

## 4. Calibration selection results

Simulated 98 s move-pause film, 3 seeds (`PYTHONPATH=rig/src .venv/bin/python -m
tars_scan.select_eval`). The lens error is the RMS over image points within 60° of the axis: how
far the fitted lens puts each ray from the true lens. "Searched" counts frames that need a board
search, which takes about 67 ms each at full size on the Mac.

| Method | Views | Searched | Lens error, median (max) | Fit time |
|---|---|---|---|---|
| all detected frames (the brute-force baseline) | 1415 | 2940 | 0.243 (0.48) px | 38.5 s |
| random 160 | 160 | 2940 | 0.682 (2.10) px | 0.6 s |
| the 20 stillest frames (one per pause) | 20 | 21 | 0.964 (1.99) px | 0.1 s |
| still + Max-logDet 20 | 20 | 21 | 0.704 (1.96) px | 0.1 s |
| value + weighted Max-logDet 20 | 20 | 530 | 0.626 (0.90) px | 0.3 s |
| value + weighted Max-logDet 80 | 80 | 530 | 0.294 (0.62) px | 0.8 s |
| **value + weighted Max-logDet 160** | 160 | 530 | **0.245 (0.47) px** | 1.4 s |

What it says:

- **Most valuable first works when "value" means information.** With 160 views it matches every
  detected frame. It needs 18% of the board searches (about 70 s instead of about 6.5 min for both
  cameras) and a 27× faster fit. Random 160 is almost 3× worse.
- **Stillness alone is not enough.** Only still frames gives about 21 board poses. That leaves the
  principal point about 2 px off: the error was flat across the whole image. Slightly moving
  frames add poses. Weighting them by their predicted noise keeps their smear from hurting.
- **Caveat.** This is a simulation with my noise model (A156). The real calibration's held-out
  error, and then SLAM on the held-out part of the room, decide.

End-to-end check on rendered images (`rig/tests/test_calib.py`): a checkerboard is rendered through
two known fisheye lenses 60 mm apart, written by the Pi's own `FrameWriter`, then calibrated. All 24
views were found in both cameras. Focal length came back within 0.6 px (0.15%), the principal point
within 0.1 px, held-out reprojection error was 0.09–0.11 px, and the baseline was 60.0 mm.

## 5. Next steps, and the decisions that are Axel's

1. **First SLAM baseline:** pick the system the rest is measured against (see the open questions
   in chat).
2. **A public dataset with ground truth** to check the export → calibrate → SLAM chain before our
   own data exists. TUM-VI is stereo fisheye + IMU; the TUM rolling-shutter set has GS and RS side
   by side. Needs a download go.
3. **On our data:** calibrate (intrinsics, stereo, camera-IMU, line delay checked against the
   sensor value), then a SLAM sweep with the held-out half of the room for scoring.
4. **Feature budget for TARS:** equal-area bucketing plus Max-logDet selection inside the chosen
   SLAM. Accuracy vs CPU, measured at 1, 1/2 and 1/4 resolution.

## 6. First real data and first SLAM runs (2026-11-09)

**Calibration film** `20261109-181853-calibrate`: three scripted phases (60 s move-pause, 45 s
tilts, 45 s sweep), board 11x8 squares of 25.03 mm, frames about 4-6x underexposed in a dim room.

- **Lens and stereo** (`tars_scan.calib`): held-out 0.37-0.41 px. The right lens sits 138.9 mm
  from the left (Axel's ruler: 138 mm). The "3.7 mm depth" is camera 0 yawed 1.5 deg, not a lens
  offset. Value-first (566 frames searched, 1 min) vs every frame (4499, 25 min): same held-out
  error, but the brute-force baseline came out 146.9 mm, because frames taken while moving carry
  rolling-shutter distortion. Weakness: the board only reached ~32 deg off-axis (lens covers ~60).
- **Camera-IMU** (`tars_scan.camimu`, A159): gyro vs board-derived rotation rate over 5-frame
  windows, correlation 0.985. IMU clock = camera clock + 6.7 ms (halves 6.3 / 7.0 ms); the IMU is
  turned 135 deg about camera 0's y axis (halves agree within 0.3 deg); gyro bias matches the still
  desk measurement; lever arm ~7 cm (weakest estimate). Synthetic test in `rig/tests`.
- **Ground truth** (`tars_scan.slam_eval`, A160): camera 0 pose from the board in 3332 frames. The
  detector sometimes numbers the corners from the far end (180 deg flips, up to 32 cm jumps); poses
  are chained by temporal continuity, after which the largest frame-to-frame step is 4 deg / 26 mm.

**Basalt** (maintained mateosss fork, Hilti 2026 config, `tars_scan.slam_run`), ATE after rigid
alignment, RPE over 1 s. Runs are deterministic on one machine; the same binary on the Pi differed
by ~15% ATE, so differences below ~20% on one film are not meaningful.

| Variant | ATE mm | RPE 1 s | Time for the 147 s film |
|---|---|---|---|
| half res (820x616) | 444 | 116 mm, 2.7 deg | 41 s Mac VM |
| half res, visual only | 523 | 176 mm, 8.7 deg | 40 s |
| half res, 33 deg mask | 405 | 118 mm, 2.7 deg | |
| half res, IMU noise x4 | 285 | 187 mm, 1.4 deg | |
| **half res + rolling-shutter correction** (`tars_scan.rs_correct`) | **175** | 87 mm, 1.1 deg | |
| same, on the Pi 5 | 202 | 83 mm, 1.1 deg | **58 s on the Pi 5 (2.5x real time), 47 C** |
| full res (1640x1232) | 124 | 73 mm, 1.3 deg | 128 s Mac VM |
| full res + rolling-shutter correction | 119 | 71 mm, 1.2 deg | 135 s Mac VM |
| ORB-SLAM3 stereo-inertial, half and full res (FAST 8/4, 2000 features) | failed | hundreds of map resets: 20-30 points per new map | |

What it says:
- **Real-time on TARS's computer is possible**: half-resolution Basalt runs 2.5x faster than real
  time on the Pi 5, including PNG decoding from the SD card.
- **Rolling-shutter correction is the first rig-specific gain**: 2.5x lower ATE at half
  resolution. On the board, it lowers the corner fit error only when turning fast (>1 rad/s: 0.42
  -> 0.37 px) and leaves still frames unchanged, as it should.
- **At full resolution the correction is within noise** (124 -> 119 mm); it matters where tracking
  is fragile, which is the half resolution the Pi can afford live. Full res is ~3x the compute.
- **ORB-SLAM3 could not track this film**: too few ORB corners in dark, gain-boosted frames (Basalt
  tracks patches, which copes). Retest on a lit room sweep before ruling it out; a config mismatch
  is not excluded yet.
- **The film is a hard, small test**: dark, ~1 m of motion around one board. A room sweep with
  better light is the next data; the ranking needs more than one film.

## 7. All candidates on the same film, and the first custom gain (2026-11-10)

Same film, half resolution + rolling-shutter correction, same score. RPE pairs now take the frame
nearest to 1 s later (before: first frame after, which dropped every pair on frame-skipped runs).

| System / variant | ATE mm | RPE 1 s | Speed |
|---|---|---|---|
| **Basalt + stereo depth hypotheses** (our patch, 9 depths) | **11** | **5 mm, 0.6 deg** | Pi 5: 84 s for 147 s (1.75x real time) |
| same, 6 depths | 11 | 6 mm, 0.7 deg | **Pi 5: 75 s (1.97x real time)** |
| Basalt stock (depth guess 2 m) | 175 (Pi: 170) | 85 mm, 1.1 deg | Pi 5: 60 s (2.45x) |
| Basalt stock, depth guess hand-set to the board's 0.5 m | 6 | 5 mm, 0.6 deg | (tuned to this film; 0.3 m: 46, 0.8 m: 78) |
| OKVIS2-X 30 fps, no time limit (BRISK, loop closure) | 65 | 10 mm, 0.6 deg | Mac 0.2x, Pi 5 0.14x real time |
| OKVIS2-X 30 fps, real-time mode | 191 | 11 mm, 0.8 deg | Mac 0.3x |
| OKVIS2-X 10 or 15 fps, or gyro-gated frame skipping | lost in the 84 deg/s turn (~1042 s) | 15-19 mm before it | Pi 5 10 fps 300 kp: 1.0x |
| OpenVINS (4 variants) | lost at ~1042 s (102 before) | | |
| ORB-SLAM3 | failed (§6) | | |

What it says:
- **Basalt's poor score was a scale error, not tracking.** Its 1 s steps were 1.52x too long (3.3x
  at the start). A new point is matched left -> right by KLT starting from one depth guess (2 m);
  the board is 0.5 m away and repeats, so KLT settles on a neighbouring corner along the epipolar
  line, which the epipolar check cannot catch, and the average-depth guess keeps the wrong depth.
- **Our patch** (`rig/slam_patches/basalt-stereo-hypotheses.patch`, loopdesk A165/A166) starts the
  stereo match from several depths, keeps the lowest patch residual and drops points where a
  different position matches within 1.6x. Scale is right from the first second and the result no
  longer depends on the depth guess (0.3 / 0.8 / 2 m all give 11.4-11.5 mm). At full resolution
  the same depths are too far apart in disparity (28 mm); half resolution is what the Pi runs.
- **Prefilters gave no robust gain**: OKVIS2 + DoG 68 vs 65 mm, + CLAHE worse; Basalt band-pass
  1-8 px 143 vs 175 mm but neighbouring bands 178-202 (sensitivity, not a gain). Phase correlation
  tracked board corners worse than KLT (`tars_scan.track_eval`, report in
  `reports/Phase matching for visual SLAM.md`).
- **OKVIS2 needs every frame** to survive fast turns, which the Pi cannot do in real time
  (stage costs per frame on the Mac: optimise 75 ms, match 60 ms incl. 21 ms loop-closure queries).
- **One film, one board**: the patch must be confirmed on a room film (start and end at the board
  to measure drift).

**Dense stereo depth for obstacles** (`tars_scan.stereo_depth`): fisheye pair rectified to a
virtual pinhole (OpenCV's fisheye stereoRectify gives a degenerate focal length for these lenses),
semi-global matching, depth inside the board compared with the board plane from ground truth
(165 frames, board 0.4-0.6 m away).

| Variant | Board pixels with depth | Error median / p90 | Pi 5 per stereo pair |
|---|---|---|---|
| 820x616 rectified, 128 disparities | 84% | 1.7 / 8.5 mm | 248 ms |
| same + band-pass 0.7-4 px | 95% | 1.3 / 3.3 mm | 278 ms |
| **410x308, 64 disparities, + band-pass 0.5-4 px** | **99%** | **1.7 / 4.1 mm** | **15 ms** |
| 410x308, no band-pass | 94% | 2.0 / 5.0 mm | 12 ms |

- **Band-pass helps dense matching, robustly**: every band tried (0.7-4 to 2-16 px) raised the
  matched share from 84% to 90-96% and cut the p90 error from 8.5 to 3.2-4.2 mm. This is where the
  high-frequency idea pays off: in dark, gain-boosted frames it removes the slow brightness
  differences between the two cameras that block matching breaks on.
- **The Pi can afford depth at 10+ Hz next to Basalt** (15 ms per pair). Expected error grows with
  distance squared: ~7 cm at 3 m for the small variant (0.15 px disparity noise).
- The board is the easiest possible target (texture everywhere, 0.5 m): plain walls and floors
  will match worse (loopdesk A167). Needs the room film.

**Fused obstacle map** (`tars_scan.voxel_map`): small-rectification stereo depth (band-pass) every
6th frame, placed with a pose source, 2 cm voxels kept once 3 frames hit them, scored on the board
(SLAM world aligned to the board path as for ATE). Picture: `runs/slam/voxel_map_board.png`.

| Poses | Board RMS in the map | Ghost voxels (2-10 cm off) | Board covered |
|---|---|---|---|
| ground truth (board PnP) | 1.7 mm | 0% | 83% |
| **Basalt + stereo hypotheses** | **5.4 mm** | **0%** | 87% |
| OKVIS2 30 fps | 18.8 mm | 30% | 94% |
| Basalt stock | 55.2 mm | 79% | 83% |

The pose ranking carries straight into the map: with stock Basalt the board is smeared into a
~10 cm slab; with the patch it is one clean surface. Not done yet: free space (ray casting), a
2D cost map for driving, loop closure (Basalt has none), and C++ / on-Pi fusion speed.
