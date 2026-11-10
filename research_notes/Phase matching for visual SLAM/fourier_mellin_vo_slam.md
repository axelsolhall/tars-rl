# Fourier-Mellin / spectral (whole-image frequency-domain) visual odometry and SLAM

Scope note: all numbers below are literature results on the named dataset, sensor and conditions. None of them predicts
performance on the TARS rig (2x rolling-shutter IMX219 fisheye stereo, MPU6500, Pi 5, indoor, 0.3-5 m depth).

Source-access note: the MDPI page of the eFMT journal paper (Remote Sensing 13(5):1000, 2021) and the EUDL/Springer pages
of Lincoln & Gonzalez returned HTTP 403 to the fetcher. eFMT numbers below therefore come from Qingwen Xu's ShanghaiTech PhD
thesis (English PDF, read in full), whose Chapter 4 presents eFMT with the same experiments; o-eFMT numbers come from the
arXiv paper 2307.10015.

Correction to the brief: arXiv 2106.12738 is not an FMT visual-odometry paper from ShanghaiTech. It is Wan, Shao & Li,
"Planetary UAV localization based on Multi-modal Registration with Pre-existing Digital Terrain Model" (see Q1).

## Q1. Fourier-Mellin VO (Kazik & Goktogan, Goecke, Checchin radar, Xu/Schwertfeger iFMI / eFMT / o-eFMT, arXiv 2106.12738): what they report and on which data

### Takeaway
Classic FMT VO estimates only 4-DoF motion (x, y, zoom-as-z, yaw about the optical axis). It assumes one scene plane
parallel to the image plane, so it is used with downward-looking cameras (rovers, UAVs) or 2D radar scans. eFMT and o-eFMT
drop the single-depth assumption but stay 4-DoF and need a down-looking camera. Their reported wins over ORB-SLAM3, SVO and
DSO come from small, feature-poor aerial or toy datasets in which the feature/direct baselines often fail outright.

### Cited Findings
**Classic FMT and its limits**
- Classic FMT registration gives a 4-DoF camera motion (3-DoF translation, with zoom standing for motion along the optical
  axis, plus yaw). It "only works when the camera does not roll or pitch" and is "limited to single-depth environments"
  because it assumes one unique zoom and translation — [Xu PhD thesis, ShanghaiTech, §1.3–1.4, pp. 14–15](https://robotics.shanghaitech.edu.cn/sites/default/files/files/Qingwen_Xu_PhD_Thesis_English.pdf)
- The thesis calls FMT "a 2.5D method" that can only estimate 4-DoF, useful "for UAVs or UUVs with a down-looking camera or
  an imaging sonar under a flat world assumption" — [Xu thesis, pp. 15–16](https://robotics.shanghaitech.edu.cn/sites/default/files/files/Qingwen_Xu_PhD_Thesis_English.pdf)
- Claimed strengths: FMT uses the full spectrum (small patterns to whole-image structure), so it "is working quite well
  under challenging conditions, i.e., feature-deprived environments, under poor visibility conditions, with dynamics in the
  scene" and was "shown to be more accurate and faster than SIFT in certain environments" — [Xu thesis, §1.4, p. 15](https://robotics.shanghaitech.edu.cn/sites/default/files/files/Qingwen_Xu_PhD_Thesis_English.pdf)
- Using FMT densely as an optical-flow substitute (registering many patches) "is computationally very expensive" — [Xu thesis, p. 16](https://robotics.shanghaitech.edu.cn/sites/default/files/files/Qingwen_Xu_PhD_Thesis_English.pdf)

**Kazik & Göktoğan (2011)**
- Citation: T. Kazik and A. H. Göktoğan, "Visual Odometry Based on the Fourier-Mellin Transform for a Rover Using a
  Monocular Ground-Facing Camera," IEEE ICM 2011, pp. 469–474 — [imreg_fmt docs reference list](https://sthoduka.github.io/imreg_fmt/docs/fourier-mellin-transform)
- Later work describes it as FMI-based VO for a ground robot, "more accurate than optical flow based method", and notes that
  FMI/iFMI VO is "only applicable to 2D motion" — [search summary citing Xu et al. 2021 / arXiv 1811.05306 context](https://arxiv.org/pdf/1811.05306v4.pdf)

**Goecke et al. (2007)**
- R. Goecke, A. Asthana, N. Pettersson, L. Petersson, "Visual vehicle egomotion estimation using the Fourier-Mellin
  Transform," IEEE Intelligent Vehicles Symposium 2007, pp. 450–455, DOI 10.1109/ivs.2007.4290156. A single camera on a car;
  FMT registers video frames to get camera rotation and translation; described as global (every pixel contributes),
  accurate and efficient; compared with two optical-flow variants on one real-world video from a moving vehicle — [ANU research portal](https://researchportalplus.anu.edu.au/en/publications/visual-vehicle-egomotion-estimation-using-the-fourier-mellin-tran/)
- Scaramuzza et al. (ICRA 2009) cite it as FMT registration of perspective images for VO — [Scaramuzza ICRA09 PDF](https://rpg.ifi.uzh.ch/docs/ICRA09_scaramuzza.pdf)

**Checchin et al. radar (2009)**
- P. Checchin, F. Gérossier, C. Blanc, R. Chapuis, L. Trassoudaine, "Radar Scan Matching SLAM Using the Fourier-Mellin
  Transform," FSR 2009 (Springer STAR vol. 62), pp. 151–161 — [DBLP](https://dblp.org/rec/conf/fsr/ChecchinGBCT09.html);
  [Springer PDF](https://link.springer.com/content/pdf/10.1007/978-3-642-13408-1_14.pdf)
- FMT registers consecutive FMCW scanning-radar scans (2D, rotation + translation) in an urban setting; cited by later radar
  odometry surveys as one of the first FMCW-radar mapping/localisation systems — [Burnett et al. RA-L 2021](https://dynsyslab.org/wp-content/papercite-data/pdf/burnett-ral21.pdf);
  [Fast-MbyM, arXiv 2203.00459](https://ar5iv.labs.arxiv.org/html/2203.00459)
- Successors: PhaRaO (Park et al., KAIST) applies FMT to Cartesian and log-polar radar images for direct radar odometry; ROS
  code (GTSAM port) at hyesu-jang/PhaRaO_gtsam — [KNU record](https://research.knu.ac.kr/en/publications/pharao-direct-radar-odometry-using-phase-correlation/);
  [GitHub PhaRaO_gtsam](https://github.com/hyesu-jang/PhaRaO_gtsam)

**Xu, Schwertfeger et al.: iFMI on omni-cameras (ICIP 2019)**
- Extended iFMI on sub-image patches of omni images to estimate 3D orientation. Data: smartphone + ~15 USD omni lens (office,
  lawn, 500 images each) and MPI-omni urban driving (200 frames); INS ground truth; only orientation evaluated — [arXiv 1811.05306](https://ar5iv.labs.arxiv.org/html/1811.05306)
- Rotation RMSE (rad), mean of roll/pitch/yaw: iFMI 0.080 ± 0.025, ORB 0.219 ± 0.071, AKAZE 0.155 ± 0.028; time per frame
  0.12 s / 0.11 s / 0.75 s — [arXiv 1811.05306, Table I](https://ar5iv.labs.arxiv.org/html/1811.05306)

**Xu thesis Ch. 2: "3D Spectral VO" (sub-image FMT + five-point RANSAC, 6-DoF up to scale)**
- Omni image → panorama → regular grid of square sub-images; FMT on each pair gives "concordant points"; recursive
  subdivision when the peak-to-noise ratio says registration failed (e.g. a hand plus wall plus table in one window);
  Stewenius five-point in OpenGV inside RANSAC gives the 6-DoF transform (translation up to scale). FMT should not go below
  ~64×64 px sub-images — [Xu thesis §2.3–2.4, pp. 26–27](https://robotics.shanghaitech.edu.cn/sites/default/files/files/Qingwen_Xu_PhD_Thesis_English.pdf)
- Sub-image size matters: 32×32 sub-images give ~5° error at 5° inter-frame rotation and ~10° at 10°; 64×64 fails on yaw at
  10°; 110×110 copes with 10° (errors ~0.05–0.5°). Stride barely changes accuracy but drives runtime (0.11 s at stride 110 vs
  0.8–1.4 s at stride 10) — [Xu thesis Table 2.1, p. 32](https://robotics.shanghaitech.edu.cn/sites/default/files/files/Qingwen_Xu_PhD_Thesis_English.pdf)
- Rotation error over office/lawn/CVLIBS (rad, mean): 3D Spectral VO 0.088 ± 0.068, optical flow 0.136 ± 0.120 (failed on
  28% of frames, failures excluded), ORB 0.227 ± 0.174, AKAZE 0.163 ± 0.115. SVO2 only worked on the higher-quality CVLIBS
  data — [Xu thesis Table 2.2, pp. 40–41](https://robotics.shanghaitech.edu.cn/sites/default/files/files/Qingwen_Xu_PhD_Thesis_English.pdf)
- Translation was evaluated only as a direction-angle error (RMSAE), per frame pair, on one 318-image, 25.27 m outdoor OVMIS
  sequence, where spectral VO was best — [Xu thesis Fig. 2.14, p. 41](https://robotics.shanghaitech.edu.cn/sites/default/files/files/Qingwen_Xu_PhD_Thesis_English.pdf)
- Blur test (synthetic Gaussian, 0–20 px, 5° rotations): "AKAZE … shows the most robust performance"; spectral VO is close to
  AKAZE and better than ORB and optical flow; ORB and spectral VO are ~7× faster than AKAZE — [Xu thesis §2.6.4, pp. 42–43](https://robotics.shanghaitech.edu.cn/sites/default/files/files/Qingwen_Xu_PhD_Thesis_English.pdf)

**Kuang, Xu, Long, Schwertfeger: sinusoid fitting (IROS 2019)**
- iFMI shifts in sliding windows of an unwrapped omni panorama; sinusoids fitted (LM + Pseudo-Huber) encode rotation and
  translation. OnePlus 5 + Kogeto Dot lens on a tracked robot, phone IMU as ground truth, i7-6700 — [arXiv 1910.00882](https://ar5iv.labs.arxiv.org/html/1910.00882)
- Overall rotation RMSE (rad): proposed 0.074, AKAZE 0.177, ORB 0.300; time 0.40 / 0.75 / 0.11 s. Translation was not evaluated
  experimentally; the model assumes every scene point is equidistant from the camera, which the authors call "easily
  violated" — [arXiv 1910.00882](https://ar5iv.labs.arxiv.org/html/1910.00882)

**eFMT (Xu, Kuang, Kneip, Schwertfeger, Remote Sensing 2021)**
- Multiple depths produce collinear energy in the phase-shift diagram. Instead of one peak, eFMT takes the sector with the
  maximum summed energy as the translation direction, stores a 1D "energy vector", and keeps scale consistent over three
  frames by pattern matching (brute-force search over scale s = 0.1:0.002:10) — [Xu thesis §4.2–4.3, pp. 84–86](https://robotics.shanghaitech.edu.cn/sites/default/files/files/Qingwen_Xu_PhD_Thesis_English.pdf);
  [search summary of the MDPI paper](https://mdpi.com/2072-4292/13/5/1000)
- Toy example: two wooden boards at different depths, translation along x only, tracking-system ground truth. ATE mean / max /
  median (mm): eFMT 2.1 / 6.0 / 1.8; FMT 17.1 / 54.7 / 10.1; SVO 17.0 / 38.0 / 17.5; ORB-SLAM3 13.0 / 21.6 / 13.6; DSO failed.
  Scale was aligned to ground truth by manual re-scaling — [Xu thesis Table 4.2, p. 91](https://robotics.shanghaitech.edu.cn/sites/default/files/files/Qingwen_Xu_PhD_Thesis_English.pdf)
- UAV dataset: DJI Matrice-300 RTK, down-looking camera, ~80 m altitude, 2 m/s, images at 0.5 Hz, 350 frames over ~1,400 m.
  SVO and DSO fail to track; ORB-SLAM3 fails several times. Up to ORB-SLAM3's first failure, "the performance of eFMT is on a
  par with ORB-SLAM3 and both of them are better than FMT" (no table; error-vs-distance plot only). Dataset:
  robotics.shanghaitech.edu.cn/static/datasets/eFMT/ShanghaiTech_Campus.zip — [Xu thesis §4.4.2, pp. 91–95](https://robotics.shanghaitech.edu.cn/sites/default/files/files/Qingwen_Xu_PhD_Thesis_English.pdf)
- Runtime: ~0.7 s/frame at 512×512 and ~0.2 s/frame at 256×256, single-threaded C++ on an i7-4790 @ 3.6 GHz, no GPU (~2× slower
  than FMT) — [Xu thesis §4.4.3, p. 95](https://robotics.shanghaitech.edu.cn/sites/default/files/files/Qingwen_Xu_PhD_Thesis_English.pdf)
- Author conclusion: "the disadvantage of the eFMT-based VO is obvious that it can be only used for 4-DoF motion estimation,
  the same as FMT"; accumulated error grows over long runs (no back end) — [Xu thesis §5.1 p. 100, §4.4.2 p. 95](https://robotics.shanghaitech.edu.cn/sites/default/files/files/Qingwen_Xu_PhD_Thesis_English.pdf)

**o-eFMT (Jiang, Li, Cao, Schwertfeger, IROS 2023, arXiv 2307.10015)**
- Same 4-DoF / down-looking limits ("can only estimate camera motions with 4 Degrees of Freedom"); adds back-end optimisation
  over a three-frame loop — [arXiv 2307.10015](https://ar5iv.labs.arxiv.org/html/2307.10015)
- Real UAV data (ShanghaiTech campus, DJI M300 RTK, images cropped to 512×512), ATE mean (m): o-eFMT 15.7, eFMT 27.5, FMT 79.8,
  ORB-SLAM3 165.3, DSO failed. Blender simulation (3 depth planes): Circle 491 m: o-eFMT 0.58, eFMT 34.2, FMT 64.3, DSO 25.6;
  Analemma 880 m: o-eFMT 5.2, eFMT 51.4, FMT 98.7, DSO 27.8; ORB-SLAM3 failed on both. Scale is aligned by the ratio of
  trajectory lengths — [arXiv 2307.10015](https://ar5iv.labs.arxiv.org/html/2307.10015)
- Runtime per registration (i7-7700 @ 3.6 GHz): o-eFMT 0.143 s, eFMT 0.407 s — [arXiv 2307.10015](https://ar5iv.labs.arxiv.org/html/2307.10015)
- Stated limits: small motion between three consecutive frames is assumed; zoom combined with a change of dominant plane can
  give wrong motion length; low-overlap frames still cause errors; no loop closure — [arXiv 2307.10015](https://ar5iv.labs.arxiv.org/html/2307.10015)

**arXiv 2106.12738 (actual content)**
- Wan, Shao & Li: a downward-camera planetary drone matched to a pre-existing DEM via the cross-power spectrum, plus local BA
  for 6-DoF; synthetic dataset of 40,200 images in nine planetary scenes; over a 33.8 km flight, mean localisation error
  0.45 m vs 1.31 m for ORB-SLAM, ~12 Hz; no code link — [arXiv 2106.12738](https://arxiv.org/abs/2106.12738)

### Inferences
- In every FMT VO result above, the camera looks down at a roughly planar scene (rover ground, UAV at 80 m, planetary DEM), is a
  2D radar, or is a 360° omni camera where only rotation was measured. None evaluates a forward-looking perspective or fisheye
  camera doing 6-DoF motion with metric translation indoors.
- The large eFMT/o-eFMT margins over ORB-SLAM3/DSO come from scenes chosen to be feature-poor (roofs, grass, wooden boards),
  with very low frame rates (0.5 Hz) or synthetic data and scale aligned after the fact. They show robustness where
  features fail; they do not show better accuracy on ordinary textured scenes.

### Gaps
- Kazik & Göktoğan 2011 abstract and numbers: IEEE Xplore was not reachable; no drift % was found.
- Goecke 2007 quantitative results: only the abstract was readable. A search snippet attributed a 17° RMS rotation error to the
  FMT method on simulated data in a US patent (Efficient egomotion estimation using patch-based projected correlation,
  US 10776928); this was not verified and should not be relied on.
- Checchin 2009 accuracy numbers: full text not read.
- The eFMT Remote Sensing paper itself (MDPI) was blocked (403); numbers are from the thesis chapter that presents eFMT.
- No KITTI-style drift % exists for any FMT camera VO in the sources found.

## Q2. Lincoln & Gonzalez phase-correlation SLAM, other spectral SLAM back ends, and learned phase correlation (DPCN / DPCN++)

### Takeaway
Spectral SLAM with RGB-D exists in two lines: Lincoln & Gonzalez (3D / log-spherical phase correlation of volumes, 2016–17,
no public accuracy numbers or code found) and the NTU/CMU kernel-cross-correlator (KCC) line (non-iterative RGB-D(-inertial)
odometry, NIDEVO, GroundSLAM). Learned phase correlation (DPCN, DPCN++) targets map-to-map / BEV / point-cloud / medical
registration on large GPUs, not frame-to-frame camera VO.

### Cited Findings
**Lincoln & Gonzalez**
- L. Lincoln, R. Gonzalez, "Fourier Volume Registration based Dense 3D Mapping," EAI Endorsed Trans. Context-aware Systems
  and Applications, Sep 2016; Springer ICTCC version "Dense 3D Mapping Using Volume Registration" — [DOAJ record](https://doaj.org/article/cdd959f69b8b46eba40c7427c955fc68);
  [EUDL](https://eudl.eu/doi/10.4108/eai.12-9-2016.151675); [Springer DOI 10.1007/978-3-319-46909-6_3](https://eudl.eu/doi/10.1007/978-3-319-46909-6_3)
- Method (from abstract-level sources): a "closed form solution" for SLAM using 3D phase correlation instead of iterative
  feature matching. RGB-D frames (Asus Xtion PRO LIVE, 640×480) are projected to 3D; a log-spherical transform + 3D phase
  correlation gives rotation and scale, then a second phase correlation gives translation. A "spherical-map transform" lets 2D
  phase correlation replace the 3D one, ~3× faster than their original method. Claimed "high noise robustness, even in the
  presence of moving objects" — [UiTM open-access record of the EAI paper](https://openaccess.library.uitm.edu.my/Record/doaj-cdd959f69b8b46eba40c7427c955fc68);
  [EUDL PDF link](https://eudl.eu/pdf/10.4108/eai.12-9-2016.151675)

**Other spectral 3D registration / SLAM**
- Fourier-Mellin-SOFT (FMS, Bülow & Birk): full 7-DoF (6-DoF + scale) spectral registration of 3D scans/volumes using an
  SO(3) Fourier transform; "very competitive due to its very high robustness and its fast computation speed", but needs full 3D
  scans as input — [Xu thesis §1.4, p. 16](https://robotics.shanghaitech.edu.cn/sites/default/files/files/Qingwen_Xu_PhD_Thesis_English.pdf)
- PHASER (ETH ASL): correlation-based (spherical Fourier) point-cloud registration for RGB-D and LiDAR; code
  github.com/ethz-asl/phaser, C++, GPL-3.0, 132 stars, last push 2022-09-13 — [arXiv 2102.02767](https://ar5iv.labs.arxiv.org/html/2102.02767);
  [GitHub API metadata via gh, 2026-11-10](https://github.com/ethz-asl/phaser)
- NIDEVO (Yang, Xu, Yuan, Xie, 2025): RGB-D VO. Rotation comes from matched plane normals (≥2 non-parallel planes); translation from
  a KCC (FFT) on orthographically projected colour images, plus mean depth difference. ICL-NUIM RMSE (m) vs ORB-SLAM2: seq0
  0.0220 vs 0.0240; seq1 0.0413 vs 0.0321 (ORB-SLAM2 better); seq3 0.0235 vs 0.0870. 71 Hz on an i5-12500H vs ORB-SLAM2 59 Hz.
  Weaker than ORB-SLAM2 when few planes are visible; needs dense depth; no code link found — [arXiv 2507.18886](https://arxiv.org/html/2507.18886v1)
- GroundSLAM (formerly "Non-iterative SLAM", arXiv 1710.05502 v1 2017 → v4 2025): KCC pose tracking + loop detection + map reuse
  with a downward camera on ground textures; ~131k-image ground-texture dataset; code github.com/sair-lab/GroundSLAM — [arXiv 1710.05502](https://arxiv.org/abs/1710.05502)
- Earlier version of that line: inertial fusion + KCC matching point clouds in the frequency domain at O(n log n) for RGB-D-inertial
  odometry, with an iterative back end for loop closure — [search summary of arXiv 1710.05502 v1](https://ar5iv.labs.arxiv.org/html/1710.05502)

**Learned phase correlation**
- DPCN (Chen, Xu, Wang, Xiong, CoRL 2020): differentiable phase correlation with learnable feature extractors for heterogeneous
  matching (e.g. sensor-to-map); code github.com/jessychen1016/DPCN (Jupyter, no licence file, last push 2020-12-09) — [CoRL 2020 entry](https://mlanthology.org/corl/2020/chen2020corl-deep/);
  [GitHub](https://github.com/jessychen1016/DPCN)
- DPCN++ (arXiv 2206.05707): UNet features + differentiable phase correlation; 2D for gravity-aligned BEV images (4-DoF), 3D up to
  7-DoF via spherical/SO(3) correlation. 2D simulation: 100% on four accuracy metrics vs classical PC 69.1/45.7/72.3/97.6;
  Aero-Ground dataset >89.2% at 1 m; 3DMatch success 96.2 (Home 1) vs TEASER 92.9. ~22 ms (2D) / ~37 ms (3DMatch) on
  i9-12900K + 2× RTX 3090; memory is the bottleneck; degrades at small overlap; the 2D form assumes pitch, roll and height are
  pre-aligned; no visual-odometry or perspective-camera experiments — [arXiv 2206.05707](https://ar5iv.labs.arxiv.org/html/2206.05707)

### Inferences
- No published spectral SLAM system was found that runs on monocular or stereo perspective/fisheye images in 6-DoF with a
  full back end and public benchmark numbers (EuRoC/TUM-VI). The 6-DoF spectral systems all take 3D input (RGB-D volumes,
  point clouds) or get rotation from somewhere else (planes, IMU).
- DPCN/DPCN++ need GPU-class hardware and training data, so they do not fit a Pi 5 front end.

### Gaps
- Lincoln & Gonzalez: no ATE/RPE numbers, dataset names (e.g. TUM RGB-D) or code found; publisher pages returned 403.
- The title and exact IMU use of arXiv 1710.05502 v1 were not confirmed from the v1 PDF.
- No "DeepFMT" paper under that name was found in the searches made.

## Q3. Do these methods work for a perspective/fisheye camera moving in 6-DoF in a cluttered room at 0.3-5 m with strong parallax?

### Takeaway
Whole-image FMT/eFMT: no. Their own authors state they need no roll/pitch (4-DoF) and a dominant plane facing the camera
(eFMT: several fronto-parallel depths). The only route the literature shows to 6-DoF with general cameras is FMT on many
sub-images used as a robust patch matcher, feeding a geometric solver (five-point RANSAC), or sinusoid fitting on 360° panoramas.
That makes it a front-end replacement for feature/patch tracking, not a different kind of VO.

### Cited Findings
- FMT "requires that the captured device doesn't roll or pitch and that the environment is planar and parallel to the imaging
  plane"; sub-image extraction is "popular in addressing the 3D motion problem" — [Xu thesis §1.4, p. 15](https://robotics.shanghaitech.edu.cn/sites/default/files/files/Qingwen_Xu_PhD_Thesis_English.pdf)
- eFMT handles multiple and slanted depths (slanted roofs appear as line segments in the translation PSD) but stays 4-DoF — [Xu thesis pp. 95, 100](https://robotics.shanghaitech.edu.cn/sites/default/files/files/Qingwen_Xu_PhD_Thesis_English.pdf)
- o-eFMT: "Roll and pitch are not supported"; assumes the depth distribution stays roughly fixed over three frames (small
  motion) — [arXiv 2307.10015](https://ar5iv.labs.arxiv.org/html/2307.10015)
- Sub-image FMT handled an indoor office with mixed depths and a moving hand by recursive subdivision, triggered on ~5% of
  pairs in that test; it worked down to a minimum sub-image of 32 px, but 32 px windows failed at ≥5° inter-frame rotation — [Xu thesis §2.5.3–2.5.4, pp. 31–34](https://robotics.shanghaitech.edu.cn/sites/default/files/files/Qingwen_Xu_PhD_Thesis_English.pdf)
- Sinusoid-fitting translation assumes equidistant scene points, "easily violated"; only rotation was tested — [arXiv 1910.00882](https://ar5iv.labs.arxiv.org/html/1910.00882)
- Phase-correlation peaks: when frames overlap enough, non-overlapping regions only add spread-out noise to the PSD — [Xu thesis p. 14](https://robotics.shanghaitech.edu.cn/sites/default/files/files/Qingwen_Xu_PhD_Thesis_English.pdf)
- FMT accuracy trades off below ~64×64 px input — [Xu thesis p. 27](https://robotics.shanghaitech.edu.cn/sites/default/files/files/Qingwen_Xu_PhD_Thesis_English.pdf)
- The 2D DPCN++ formulation assumes pitch, roll and height are already aligned (BEV) — [arXiv 2206.05707](https://ar5iv.labs.arxiv.org/html/2206.05707)

### Inferences
- Which assumptions break on the TARS rig. (a) Rotation about axes other than the optical axis: a forward fisheye on a 50 cm
  robot pitches and rolls, and yaw of a forward camera is horizontal image shift, not in-plane rotation, so whole-image FMT
  mislabels it. (b) Single or fronto-parallel depth: a cluttered room at 0.3–5 m has a depth ratio of ~17× and strong
  parallax, so whole-image peaks smear (eFMT's line model expects fronto-parallel planes and zoom/translation along one line).
  (c) Fisheye distortion breaks the global shift model unless the image is undistorted/rectified or cut into small patches.
  (d) Small, uniform motion across three frames (o-eFMT).
- What could transfer: sub-image phase correlation (≥64×64 px windows on rectified pinhole views of the fisheye) as a
  blur-tolerant patch matcher feeding stereo triangulation + RANSAC, or as Basalt-style patch tracking. Xu's blur test suggests
  it is about as blur-robust as AKAZE and better than ORB; that was synthetic Gaussian blur on omni images, not rolling-shutter
  motion blur.
- Large windows (110 px) are needed for robustness to ≥10° inter-frame rotation, but large windows also mix depths more at close
  range. That tension is untested in the literature for indoor 0.3–5 m scenes.

### Gaps
- No paper found evaluates FMT/phase-correlation VO on EuRoC, TUM-VI or another indoor 6-DoF handheld/robot benchmark against
  Basalt, OKVIS2, VINS or ORB-SLAM3.
- No study found of rolling-shutter effects on FMT registration.

## Q4. Can FMT/eFMT be combined with an IMU (gyro for rotation, phase correlation for translation)?

### Takeaway
Yes in principle, and it is the standard design for downward-camera flow sensors and some radar odometry. For eFMT it is
proposed only as future work. Xu warns that derotating with the IMU tilt does not by itself make eFMT 6-DoF, because overlap
after warping may be limited. No camera-based FMT+IMU VO with published results was found.

### Cited Findings
- eFMT authors: "with the help of an IMU, eFMT will be able to estimate the 6-DoF motion" (future work); in the outlook, fusing
  an IMU for camera tilt "does not mean that eFMT can estimate 6-DoF motion with these sensors because there may be limited
  overlap after transforming the image with tilt angles" — [Xu thesis pp. 97, 100](https://robotics.shanghaitech.edu.cn/sites/default/files/files/Qingwen_Xu_PhD_Thesis_English.pdf)
- Downward-camera practice: the IMU supplies rotation, the image is de-rotated, then 2D translation is estimated under a planar
  floor assumption with a height sensor (e.g. Honegger et al. ICRA 2013 gyro + sonar optical-flow module) — [TUM Autonomous Navigation lecture 7 part 2](https://autonav.in.tum.de/handouts/lecture_7_part_2.pdf)
- FD-RIO (radar, arXiv 2505.07694): rotation and translation both from phase correlation (polar-warped FFT magnitudes, then
  translation), fused with raw IMU in a Kalman filter. Navtech radar on MulRan/Boreas: mean 1.68% / 0.35°/100 m (MulRan), 1.98%
  / 0.02°/100 m (Boreas); 14.3 Hz radar processing on an i7-13700 single thread (7.5 Hz on an i7-6700); SE(2) only, gyro bias
  tuned per dataset — [arXiv 2505.07694](https://arxiv.org/html/2505.07694v1)
- An RGB-D-inertial odometry combined inertial fusion with KCC frequency-domain matching of point clouds — [arXiv 1710.05502 (v1 line)](https://arxiv.org/abs/1710.05502)
- Correlation Flow (Wang, Ji, Nguyen, Xie, ICRA 2018): kernel cross-correlator optical flow for a downward monocular camera on a
  micro-UAV; claimed robust to motion blur, also estimates altitude velocity and yaw rate; ROS code mentioned but no URL on the
  arXiv page; no numbers on the abstract page — [arXiv 1802.07078](https://arxiv.org/abs/1802.07078)

### Inferences
- For a forward-looking stereo fisheye, "gyro rotation then phase-correlate translation" only works patch-wise: after gyro
  derotation the remaining image motion of a translating camera depends on per-pixel depth (parallax), so one global
  translation peak does not exist indoors. Stereo depth per patch would be needed to turn patch shifts into metric
  translation, which is what Basalt/OKVIS2 already do with their own patch or feature tracks.
- The MPU6500 gyro could supply the rotation prior that removes the in-plane rotation and scale search (the costly log-polar
  step), leaving plain 2D phase correlation per patch. This is cheaper than full FMT; it is an inference, not a measured result.

### Gaps
- No published camera VO was found that fuses an IMU with FMT/eFMT and reports accuracy.

## Q5. Open-source code: repositories, language, licence, maintenance

### Takeaway
Usable code is thin. o-eFMT (C++, GPL-3.0, 9 stars, last push 2023) is the only public FMT camera-VO code found; it is
pinhole-only and 4-DoF. Generic FMT image-registration libraries exist (imreg_fmt, GPL-3.0; LogPolarFFTTemplateMatcher). No
public eFMT (2021) or "3D Spectral VO" repository was found. All licences found are GPL-3.0 or unstated.

### Cited Findings
(GitHub metadata pulled with `gh api` on 2026-11-10)
- STAR-Center/o-eFMT: C++, GPL-3.0, 9 stars, last push 2023-03-02; deps fftw3, OpenCV 4.2, Eigen, Ceres; offline `image_main
  <images dir>` with a PinHole cam_param.yaml; derived from the Gohlke/Týč Python imreg code — [GitHub](https://github.com/STAR-Center/o-eFMT)
- STAR-Center/eFMT: not found (404) — [GitHub search](https://github.com/STAR-Center)
- sthoduka/imreg_fmt: C++ FMT image registration, GPL-3.0, 116 stars, last push 2026-11-06 — [GitHub](https://github.com/sthoduka/imreg_fmt);
  companion sthoduka/fmt_motion_detection (C++, FMT camera-motion compensation) — [GitHub](https://github.com/sthoduka/fmt_motion_detection)
- Smorodov/LogPolarFFTTemplateMatcher: OpenCV FMT implementation, C++, 181 stars — [GitHub](https://github.com/Smorodov/LogPolarFFTTemplateMatcher)
- jessychen1016/DPCN: Jupyter/PyTorch, no licence, last push 2020-12-09 — [GitHub](https://github.com/jessychen1016/DPCN)
- ethz-asl/phaser: C++, GPL-3.0, point clouds — [GitHub](https://github.com/ethz-asl/phaser)
- hyesu-jang/PhaRaO_gtsam: ROS, radar — [GitHub](https://github.com/hyesu-jang/PhaRaO_gtsam)
- sair-lab/GroundSLAM: downward ground-texture KCC SLAM — [arXiv 1710.05502](https://arxiv.org/abs/1710.05502)
- Datasets: eFMT ShanghaiTech campus UAV set (robotics.shanghaitech.edu.cn/static/datasets/eFMT/ShanghaiTech_Campus.zip) — [Xu thesis p. 87 footnote](https://robotics.shanghaitech.edu.cn/sites/default/files/files/Qingwen_Xu_PhD_Thesis_English.pdf)

### Inferences
- GPL-3.0 on o-eFMT/imreg_fmt matters only if TARS code is distributed. A per-patch phase-correlation matcher is short (OpenCV
  `phaseCorrelate`, FFTW) and could be written fresh.
- Cost on a Pi 5 (inference): eFMT at 256×256 took ~0.2 s/frame on a 3.6 GHz desktop i7, and the sub-image spectral VO
  ~0.11 s/frame at stride 110. A Cortex-A76 Pi 5 is slower per core, so whole-image eFMT at camera rate (≥20 Hz) on the Pi is
  unlikely without heavy downscaling. Per-patch 64×64 phase correlation (two FFTs + one inverse) is cheap and parallelisable.
  Not measured.

### Gaps
- Licence of the eFMT Remote Sensing code, if any, is unknown (no repository found).
- No ARM / Raspberry Pi runtime measurement exists in any source found.
