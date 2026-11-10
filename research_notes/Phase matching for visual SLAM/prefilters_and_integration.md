# Image prefilters for feature tracking under noise, blur and illumination change, and integration points in Basalt / OKVIS2

Scope note: literature numbers below come from other sensors and conditions (named with each number). They are not predictions for the TARS rig (2x rolling-shutter IMX219 fisheye, Pi 5, ~4x underexposed + gain, blur on turns). Code references were read from shallow clones on 2026-11-10 at these commits:
- Basalt upstream `gitlab.com/VladyslavUsenko/basalt` @ `0f3b2b52` (2026-03-22); headers submodule `basalt-headers` @ `b7f915b1` (2026-03-17)
- Basalt Monado fork `gitlab.freedesktop.org/mateosss/basalt` @ `df6e970c` (2026-07-24) (note: it is NOT on github.com/mateosss — that URL 404s)
- OKVIS2 `github.com/smartroboticslab/okvis2` @ `a2ea0068` (2025-12-11)
- OpenVINS `github.com/rpng/open_vins` @ `69488123` (2025-11-30)

---

## Q1. Measured effects of prefilters / robust residuals (DoG/LoG, whitening, CLAHE/HE, gamma, census/rank, gradient magnitude, normalised residuals) on tracking

### Takeaway
The only systematic, numeric comparison found (Park, Schöps, Pollefeys, ICRA 2017) shows that on real sequences with illumination changes, census (4.23 cm/s drift) and gradient magnitude (4.33) beat raw intensity (5.56) and a local-mean-subtraction (high-pass) residual (4.90), while ZNCC was worse than raw (6.39); census costs ~10x the runtime. Basalt's own residual is a mean-scaled SSD (gain-invariant only, not offset-invariant). No paper was found that measures a DoG band-pass or spectral whitening in front of KLT, nor one that measures CLAHE/histogram equalisation in front of KLT with tracking-error numbers.

### Cited Findings

**Park, Schöps, Pollefeys 2017 "Illumination change robustness in direct visual SLAM" (ICRA 2017, pp. 4523–4530)** — [PDF (MSR copy)](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/09/park2017icra.pdf), [DBLP](https://dblp.org/rec/conf/icra/ParkSP17)
- Evaluated 10 direct-alignment costs: BCA (raw brightness constancy), GMedian (global median bias), GAffine (global affine gain/bias), ZNCC, MI, GradM (gradient magnitude), Grad (gradient vector), LMean (subtract local mean of an 11x11 patch around each pixel), DF (descriptor fields), Census. Also SIFT/COLMAP as a feature baseline. — [Park 2017](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/09/park2017icra.pdf)
- Setup: frame-to-frame dense RGB-D alignment, pyramid down to 40x30, LM until mean pixel update < 1e-2 or 20 iterations; Huber parameter tuned per method on TUM RGB-D (BCA 10.0, GMedian 6.5, GAffine 5.2, GradM 16.9, Grad 7.3, LMean 2.9, DF 300). Sobel filter used for GradM. — [Park 2017](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/09/park2017icra.pdf)
- Table IV, translational drift (cm/s), mean per group:

  | Method | TUM RGB-D (5 seqs, little illum. change) | ICL-NUIM synthetic w/ illum. change (mean) | ICL-NUIM "flash" (camera-mounted light) | Real Kinect w/ illum. change (global/local/flash mean) |
  |---|---|---|---|---|
  | BCA (raw) | 3.14 | 0.21 | 0.55 | 5.56 |
  | GMedian | 2.98 | 0.23 | 0.67 | 4.76 |
  | GAffine | 3.03 | 0.23 | 0.69 | 4.69 |
  | ZNCC | 3.79 | 0.15 | 0.24 | 6.39 |
  | MI | 4.22 | 0.18 | 0.26 | 6.79 |
  | GradM | 3.05 | 0.12 | 0.17 | 4.33 |
  | Grad | 3.14 | 0.14 | 0.18 | 4.61 |
  | LMean (local-mean subtraction, 11x11) | 3.05 | 0.15 | 0.19 | 4.90 |
  | DF | 3.52 | 0.16 | 0.21 | 4.93 |
  | Census | 3.50 | 0.16 | 0.20 | 4.23 |

  — [Park 2017, Table IV](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/09/park2017icra.pdf)
- Authors' conclusions: "on our real datasets, Census performs better than all other methods. GradM still performs well, yielding the second best results"; GMedian/GAffine/LMean do not work as well on real illumination-change data as on TUM. "GradM consistently performs very well over all datasets." — [Park 2017](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/09/park2017icra.pdf)
- Recomputing descriptors at warped coordinates (rather than pre-computing on the template) always lowered drift (Table III, TUM+real): GradM 3.96 → 3.53, Grad 4.15 → 3.69, LMean 5.77 → 3.74, DF 7.17 → 3.78, Census 4.82 → 4.05 cm/s. — [Park 2017, Table III](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/09/park2017icra.pdf)
- Convergence-basin test (synthetic pairs with translation + degradation incl. Gaussian blur and intensity noise): BCA fails for small abrupt global/local light changes; gradient-based methods are robust to light changes "but have a smaller convergence basin"; GAffine-like methods have the largest basin; SIFT matching "performs poorly for intensity noise and blur, since those degradation types corrupt the image gradients" and gave less accurate alignments than direct methods. Results shown only as colour plots (Fig. 5), no table. — [Park 2017, Sec. V-D](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/09/park2017icra.pdf)
- Runtime (Table V, GTX 950 GPU, unoptimised, TUM+real): BCA 28±13 ms, GAffine 31±11, ZNCC 58±39, GradM 35±14 (pre-compute)/37±15 (re-compute), Grad 67/66, LMean 100/72, DF 114/124, Census 489±195 (pre)/355±114 (re) ms per frame. — [Park 2017, Table V](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/09/park2017icra.pdf)
- A later paper summarises Park et al. as favouring "the gradient magnitude due to its accuracy, robustness and speed while the census transform provided more accurate results at a much larger computational cost" (second-hand). — [Quenzel et al., arXiv 2004.04090](https://arxiv.org/pdf/2004.04090)

**Basalt's locally-scaled SSD (Usenko et al. 2019/2020, RA-L, "Visual-Inertial Mapping with Non-Linear Factor Recovery")**
- Residual (Eq. 5): each pixel of the patch in the target frame divided by the target patch mean, minus the source pixel divided by its patch mean; described as "invariant to intensity scaling". LSSD chosen over ZNCC because it is "computationally less expensive". Combined with inverse-compositional tracking (Baker & Matthews), coarse-to-fine pyramid, and a forward-backward check (points not returning to start are discarded). FAST corners on a 50 px grid, giving 80–120 features on EuRoC. — [arXiv 1904.06504v3 HTML, Sec. IV-A, VI](https://arxiv.org/html/1904.06504v3)
- Timing: VIO 7.83 ms/frame average on EuRoC (5.5–9.4 ms), Intel E5-1620 (4C/8T); KLT time not broken out. EuRoC stereo VIO ATE 0.04–0.13 m. No TUM VI or low-light results in the paper. — [arXiv 1904.06504v3](https://arxiv.org/html/1904.06504v3)
- In code: `data /= mean` and `residual[i] = num_valid_points * val / sum - data[i]` (division only, no mean subtraction). — [patch.h L75–100](https://gitlab.com/VladyslavUsenko/basalt/-/blob/0f3b2b52c807f70ff4e2973ce253c73329eea7bc/include/basalt/optical_flow/patch.h#L75-100), [patch.h L172–207](https://gitlab.com/VladyslavUsenko/basalt/-/blob/0f3b2b52c807f70ff4e2973ce253c73329eea7bc/include/basalt/optical_flow/patch.h#L172-207)

**Census / binary descriptors in least squares: Alismail, Kaess, Browning, Lucey "Bit-Planes" (arXiv 1604.00990; RA-L 2(2):444–451, 2017 "Direct Visual Odometry in Low Light using Binary Descriptors")**
- Method: 3x3 census-style comparisons of centre vs 8 neighbours; each bit stored as its own channel ("Bit-Planes") so squared Euclidean distance = Hamming distance and Gauss-Newton can be used. Pre-smoothing 3x3 Gaussian σ = 0.5; Tukey loss. — [ar5iv 1604.00990](https://ar5iv.labs.arxiv.org/html/1604.00990), [CMU RI page](https://www.ri.cmu.edu/publications/direct-visual-odometry-in-low-light-using-binary-descriptors)
- Smoothing ablation (synthetic small translations, Fig. 7): larger smoothing kernels, before or on the Bit-Planes, raised error; the smallest kernel gave the lowest error ("large kernels wash out the detail needed for small motions"). — [ar5iv 1604.00990](https://ar5iv.labs.arxiv.org/html/1604.00990)
- Results are mostly qualitative: on New Tsukuba "lamps" (synthetic illumination), robustness "far exceeds" FOVIS and DVO (plots only); in underground mines (1024x544 stereo, 77 cm baseline, ~400 m, no ground truth) FOVIS and two other packages "all fail too often" (no failure rate). Main failure mode: complete washout from the robot's LED in tight turns. — [ar5iv 1604.00990](https://ar5iv.labs.arxiv.org/html/1604.00990)
- Cost (New Tsukuba, first 500 frames, CPU with SSE2, hardware not named): raw intensity mean 7.32 ms/frame (median 5.0), Bit-Planes mean 64.19 ms (median 53.0). KITTI 1024x376 per stage: descriptor computation 0.28 ms (raw) vs 5.55 ms (Bit-Planes); Jacobian pre-computation 5.00 vs 13.92 ms; pyramid construction 0.44 ms. — [ar5iv 1604.00990, Tables I–II](https://ar5iv.labs.arxiv.org/html/1604.00990)

**Global histogram equalisation / CLAHE**
- OpenVINS applies a per-image preprocessing switch before building the KLT pyramid: `HISTOGRAM` → `cv::equalizeHist`, `CLAHE` → `cv::createCLAHE(10.0, 8x8)`, `NONE` → raw. Default is `HISTOGRAM`. — [TrackKLT.cpp L56–71](https://github.com/rpng/open_vins/blob/69488123ed9362dd44b6f28e7f4680abbff1442b/ov_core/src/track/TrackKLT.cpp#L56-L71), [VioManagerOptions.h L436, L468–475](https://github.com/rpng/open_vins/blob/69488123ed9362dd44b6f28e7f4680abbff1442b/ov_msckf/src/core/VioManagerOptions.h#L436)
- OKVIS2 has CLAHE written but commented out (`//mClahe = cv::createCLAHE(3.0, cv::Size(8, 8));`). — [Frontend.cpp L107, L2583](https://github.com/smartroboticslab/okvis2/blob/a2ea00688cd10988aae7bd52ab7935ce9a657ec0/okvis_frontend/src/Frontend.cpp#L2583)
- A user running OpenVINS on a ModalAI VOXL2 single camera saw ~1 feature regardless of `histogram_method` HISTOGRAM vs NONE (anecdotal, cause elsewhere). — [ModalAI forum](https://forum.modalai.com/topic/5265/near-zero-feature-extraction-single-camera.md)
- ORB-SLAM3 + a Histogram-Equalization-Prior enhancement module: APE RMSE 0.99 cm and 0.76 cm on two ETH3D low-light sequences, ">50%" better than plain ORB-SLAM3 (abstract-level claim). — [IEEE 10786104](https://ieeexplore.ieee.org/document/10786104/)
- DSO vs ORB-SLAM3 in dark indoor scenes lit only by a robot-mounted light: EnlightenGAN and CLAHE gave the best SSIM/MSE image-quality scores (image metrics, not tracking error, in the snippet available). — [Crocetti et al., J. Field Robotics 2025](https://onlinelibrary.wiley.com/doi/10.1002/rob.22595)
- Twilight SLAM (ETH3D semi-dark/dark): enhancer + ORB-SLAM3 / SuperPoint-SLAM combinations still had feature-detection failures in some runs. — [arXiv 2304.11310](https://arxiv.org/pdf/2304.11310)

**Photometric calibration / affine brightness (DSO line of work)**
- Engel et al.'s TUM monoVO benchmark supplies vignetting + response calibration; direct methods otherwise compensate auto-exposure by an affine brightness mapping between frames. — [arXiv 1607.02555](https://arxiv.org/pdf/1607.02555)
- Bergmann et al. online photometric calibration: on a manually disturbed ICL-NUIM sequence "DSO entirely fails without photometric calibration"; on 27 TUM monoVO sequences online calibration "significantly reduces the alignment errors compared to using no calibration", similar to ground-truth calibration. — [arXiv 1710.02081](https://arxiv.org/pdf/1710.02081)

**Motion blur**
- Jin, Favaro, Cipolla (CVPR 2005): track blurred regions by blurring the template (cost depends only on deformation + two blur vectors) rather than deblurring the frames. — [Adobe Research page](https://research.adobe.com/publication/visual-tracking-in-the-presence-of-motion-blur/), [mlanthology](https://mlanthology.org/cvpr/2005/jin2005cvpr-visual)
- MBA-VO (Liu, Zuo, Larsson, Pollefeys, ICCV 2021): DSO-based direct VO that estimates the camera trajectory within the exposure and models blur in alignment; claims accuracy close to sharp images; real time on a laptop with RTX 2080; releases a blur benchmark. — [arXiv 2103.13684](https://arxiv.org/abs/2103.13684), [CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_MBA-VO_Motion_Blur_Aware_Visual_Odometry_ICCV_2021_paper.html)

### Inferences
- Park et al.'s LMean (subtract an 11x11 local mean) is the closest literature analogue to a DoG band-pass's low-cut half; it improved real-illumination drift vs raw (5.56 → 4.90 cm/s, only with re-computation at warped coordinates) but was beaten by GradM and Census. A DoG additionally low-passes (σ≈1), which the Bit-Planes smoothing ablation suggests is harmful for very small motions with larger kernels but helpful against noise — consistent with the TARS measurement where KLT gained only slightly (0.16 → 0.15 px) while phase correlation (which uses the whole spectrum and is noise/blur-sensitive) gained a lot (lost 16.8% → 5.7%).
- Basalt's residual only cancels a multiplicative gain per patch. Analogue gain on the IMX219 is multiplicative, so Basalt already handles the "4x underexposed lifted by gain" brightness change; it does not cancel additive offsets (black-level error, flare) and does nothing against amplified noise — those are where a band-pass might add value.
- Census is the strongest literature option for real illumination change, but at ~10x the cost of raw SSD on desktop GPU (Park) and ~9x on CPU (Bit-Planes); on a Pi 5 this likely only fits if restricted to the sparse patches (Basalt's 24–52 pixel patterns), not dense images.

### Gaps
- No paper found that measures DoG/LoG band-pass, or spectral whitening, as a KLT/patch-tracker prefilter with tracking-error or lost-track numbers. The TARS measurement may be the only data point for this setup.
- No source with numbers for CLAHE/HE/gamma in front of KLT specifically (OpenVINS ships the option with no published ablation found). The HEP and Crocetti results are on ORB-SLAM3/DSO and partly image-quality metrics.
- Could not verify Park et al. with a ZNCC-on-patches (as opposed to whole-image) setup; their ZNCC is image-level and may not transfer to Basalt-size patches.
- ORB-SLAM low-light evaluations with per-sequence numbers (LaMARia report arXiv 2607.17699) were only available at abstract level.

---

## Q2. Datasets with low light / blur, and which methods won (with numbers)

### Takeaway
On OIVIO (onboard-light mines/tunnels) and UMA-VI (low texture, lights-off), descriptor/learned systems (AirSLAM, ORB-SLAM3) had lower ATE than Basalt; but on the synthetic "Dark EuRoC" gamma-darkening test, optical-flow methods (Basalt, DROID-SLAM) stayed stable while ORB-SLAM3 and OKVIS failed at the darkest levels. OKVIS (v1) beat Basalt on 7 of 10 UMA-VI sequences.

### Cited Findings
- **OIVIO** (Kasper, McGuire, Heckman, IROS 2019): 36 sequences, >145 min stereo + IMU, mines/tunnels with onboard light at ~100/50/15% of full intensity on the same trajectory. — [OIVIO page](https://arpg.colorado.edu/oivio/)
- **OIVIO results (AirSLAM paper, Table III, RMSE m)**: average Basalt 0.192, DROID-SLAM 0.175, ORB-SLAM3 0.065, AirSLAM 0.058, PL-SLAM 1.358; Kimera fails (F) on 3 of 9. Example: MN_015_GV_01 (15% light) Basalt 0.216 vs ORB-SLAM3 0.066; TN_050_GV_01 Basalt 0.521 vs ORB-SLAM3 0.082. Text: onboard illumination "has almost no impact on AirSLAM and ORB-SLAM3" but "reduces the accuracy of OKVIS, Basalt, and PL-SLAM". — [AirSLAM, arXiv 2408.03520](https://arxiv.org/html/2408.03520)
- **UMA-VI** (Zuñiga-Noël et al., IJRR 2020): 32 sequences, handheld rig, low texture and dynamic illumination incl. sudden lights-off. — [UMA-VI paper PDF](https://mapir.uma.es/papersrepo/2020/2020_The_UMA-VI_dataset_Visual%E2%80%93inertial_odometry_in_low-textured_and_dynamic_illumination_environments.pdf), [dataset page](https://mapir.isa.uma.es/mapirwebsite/?p=2108)
- **UMA-VI results (AirSLAM Table IV, RMSE m)**: Basalt vs OKVIS per sequence: conference-csc1 1.270/1.118; csc2 0.682/0.470; csc3 0.469/0.088; lab-module-csc-rev 0.486/0.861; lab-module-csc 0.403/0.579; long-walk-eng 5.046/3.005; third-floor-csc1 0.420/0.287; csc2 0.590/0.271; two-floors-csc1 0.760/0.154; csc2 1.211/0.679. ORB-SLAM3 failed on 6 of 10, PL-SLAM on 6. AirSLAM best on 7 of 10. — [AirSLAM, arXiv 2408.03520](https://arxiv.org/html/2408.03520)
- **Dark EuRoC** (AirSLAM Sec. VII-C3): EuRoC V2_01 darkened with V_out = A·V_in^(1/γ) at 12 levels; "ORB-SLAM3 and OKVIS RMSE rises as brightness drops, with tracking failures or large drift at L10 and L11"; AirSLAM "comparable to Basalt and DROID-SLAM"; "optical-flow methods were more stable than feature-based methods" (Fig. 11 only, no table). — [AirSLAM, arXiv 2408.03520](https://arxiv.org/html/2408.03520)
- **ETH3D dark/semi-dark**: HEP-enhanced ORB-SLAM3 0.99 / 0.76 cm APE RMSE on two low-light sequences (>50% better than ORB-SLAM3). — [IEEE 10786104](https://ieeexplore.ieee.org/document/10786104/); enhancer + ORB-SLAM3 combinations had detection failures — [Twilight SLAM, arXiv 2304.11310](https://arxiv.org/pdf/2304.11310)
- **TUM VI**: original benchmark evaluated OKVIS, ROVIO, VINS-Mono, BASALT (table not retrieved). — [arXiv 1804.06120](https://arxiv.org/pdf/1804.06120), [dataset page](https://cvg.cit.tum.de/data/datasets/vi-dataset). OKVIS2 paper Table II (TUM VI corridor, ATE m, averaged; extraction partly garbled — verify): ROVIO ~0.86, Basalt (causal VIO) ~0.34, OKVIS ~0.70, OKVIS2 VIO ~0.40, OKVIS2 SLAM / ORB-SLAM3 ~0.06–0.09. — [OKVIS2, arXiv 2202.09199](https://arxiv.org/pdf/2202.09199)
- **LaMARia low-light sequences** (2026 project report): Kimera-VIO the only one of six systems (ORB-SLAM3, DSO, Kimera-VIO, OpenVINS, DPVO, DPV-SLAM) to track all five sequences; ORB-SLAM3, DSO and OpenVINS failed or diverged on most hard/low-light sequences; DPVO never lost track but ~100 m absolute error in low light. — [arXiv 2607.17699](https://arxiv.org/abs/2607.17699)
- **Blur**: MBA-VO contributes a benchmark with varying blur levels (numbers not retrieved). — [arXiv 2103.13684](https://arxiv.org/abs/2103.13684)

### Inferences
- For a rig like TARS (dark + blur), the literature pattern is: KLT/optical-flow front ends (Basalt) degrade gracefully with darkness but lose accuracy (OIVIO), while descriptor front ends (OKVIS, ORB-SLAM3) are accurate when they work but fail abruptly at the darkest levels (Dark EuRoC). That matches testing OKVIS2 specifically on the darkest TARS recordings.
- None of these datasets is rolling-shutter fisheye with gain-lifted underexposure; TUM VI (global-shutter fisheye, photometric calibration) is the closest optics match.

### Gaps
- 4Seasons night and "Dark-Robot" were not researched within the budget; no numbers here.
- TUM VI dark-room results by method and the original Schubert et al. table were not retrieved.
- No dataset found that isolates prefilters (all comparisons are system-level).

---

## Q3. Basalt source: where patch optical flow lives, residual, pyramid, config, and where to insert a prefilter or alternative matcher

### Takeaway
Everything is in `include/basalt/optical_flow/*.h` (header-only templates); images are `uint16_t` (8-bit input shifted left by 8), the pyramid is a 5-tap binomial (1-4-6-4-1) subsample in `basalt-headers`, and tracking is inverse-compositional SE2 Gauss-Newton on a mean-scaled patch. The cleanest prefilter hook is right before `pyramid->at(i).setFromImage(...)` in `processFrame`; the cleanest alternative matcher hook is a new patch type (replacing `OpticalFlowPatch`) or a new `optical_flow_type` registered in `OpticalFlowFactory`.

### Cited Findings (upstream @ 0f3b2b52 unless noted)
**Files and variants**
- `include/basalt/optical_flow/`: `frame_to_frame_optical_flow.h` (408 lines, default), `multiscale_frame_to_frame_optical_flow.h` (471; "patches can be created at all pyramid levels, not just the lowest", L58), `patch_optical_flow.h` (413; keeps reference patches from the frame where the point was first seen, never erased, L53–58), `patch.h` (226), `patterns.h` (171), `optical_flow.h` (88). — [optical_flow dir](https://gitlab.com/VladyslavUsenko/basalt/-/tree/0f3b2b52c807f70ff4e2973ce253c73329eea7bc/include/basalt/optical_flow)
- Factory selects by `optical_flow_type` = `"patch"` (L48), `"frame_to_frame"` (L74), `"multiscale_frame_to_frame"` (L100), and by `optical_flow_pattern` ∈ {24, 52, 51, 50} via a switch (L49–68). — [optical_flow.cpp L44–110](https://gitlab.com/VladyslavUsenko/basalt/-/blob/0f3b2b52c807f70ff4e2973ce253c73329eea7bc/src/optical_flow/optical_flow.cpp#L44-110)

**Residual (patch.h)**
- `OpticalFlowPatch<Scalar, Pattern>` (L45–46). `setData` samples the pattern with bilinear `interp`, sums valid samples, then `data /= mean` (L75–100). `setDataJacSe2` uses `interpGrad` (L121) and builds the SE2 Jacobian with the mean-normalisation derivative folded in (`J_se2.row(i) -= grad_sum_se2.transpose() * data[i] / sum`, L138; `J_se2 *= mean_inv`, L144). `setFromImage` precomputes `H_se2_inv_J_se2_T` (inverse compositional) (L147–159). — [patch.h L45–170](https://gitlab.com/VladyslavUsenko/basalt/-/blob/0f3b2b52c807f70ff4e2973ce253c73329eea7bc/include/basalt/optical_flow/patch.h#L45-170)
- Validity: `valid = mean > epsilon && H_se2_inv_J_se2_T finite ...`, with the comment that an all-black patch "cannot be normalized" (L165–168). — [patch.h L165–168](https://gitlab.com/VladyslavUsenko/basalt/-/blob/0f3b2b52c807f70ff4e2973ce253c73329eea7bc/include/basalt/optical_flow/patch.h#L165-168)
- `residual()`: samples the warped pattern; if `sum < epsilon` returns zero residual; else `residual[i] = num_valid_points * val / sum - data[i]`; pixels out of bounds get `-1` and are skipped; returns true only if more than half the pattern is valid (L172–207). — [patch.h L172–207](https://gitlab.com/VladyslavUsenko/basalt/-/blob/0f3b2b52c807f70ff4e2973ce253c73329eea7bc/include/basalt/optical_flow/patch.h#L172-207)
- Same residual in the Monado fork (`residual[i] = num_valid_points * val / sum - data[i]`, L193). — [mateosss patch.h L193](https://gitlab.freedesktop.org/mateosss/basalt/-/blob/df6e970c8da7636eb401a09e3317fbeaaf829b9a/include/basalt/optical_flow/patch.h#L193)
- Patterns: Pattern24 (diamond, 24 px), Pattern52, Pattern51, Pattern50 (L42–101). — [patterns.h](https://gitlab.com/VladyslavUsenko/basalt/-/blob/0f3b2b52c807f70ff4e2973ce253c73329eea7bc/include/basalt/optical_flow/patterns.h#L42-101)

**Tracking loop (frame_to_frame_optical_flow.h)**
- `processFrame` (L111) builds one `ManagedImagePyr<uint16_t>` per camera in parallel: `pyramid->at(i).setFromImage(*new_img_vec->img_data[i].img, config.optical_flow_levels)` (first frame L123–131, later frames L145–152). — [L111–171](https://gitlab.com/VladyslavUsenko/basalt/-/blob/0f3b2b52c807f70ff4e2973ce253c73329eea7bc/include/basalt/optical_flow/frame_to_frame_optical_flow.h#L111-171)
- `trackPoints` (L181): forward track, then backward track; accept if `dist2 < config.optical_flow_max_recovered_dist2` (L214–226). — [L181–240](https://gitlab.com/VladyslavUsenko/basalt/-/blob/0f3b2b52c807f70ff4e2973ce253c73329eea7bc/include/basalt/optical_flow/frame_to_frame_optical_flow.h#L181-240)
- `trackPoint` (L243–270): coarse-to-fine from `optical_flow_levels` to 0; builds a new `PatchT` from the OLD pyramid at each level and calls `trackPointAtLevel`. `trackPointAtLevel` (L273–305): up to `optical_flow_max_iterations` GN steps, `inc = -H_se2_inv_J_se2_T * res`, `transform *= SE2::exp(inc)`, bounds check with margin 2. — [L243–305](https://gitlab.com/VladyslavUsenko/basalt/-/blob/0f3b2b52c807f70ff4e2973ce253c73329eea7bc/include/basalt/optical_flow/frame_to_frame_optical_flow.h#L243-305)
- `addPoints` (L311) detects on `pyramid->at(0).lvl(0)` (the same image used for tracking) with grid `optical_flow_detection_grid_size` (L320–321), then tracks new points cam0→cam1 (L338). `filterPoints` (L346) rejects stereo pairs with epipolar error > `optical_flow_epipolar_error` (L372–375). — [L311–380](https://gitlab.com/VladyslavUsenko/basalt/-/blob/0f3b2b52c807f70ff4e2973ce253c73329eea7bc/include/basalt/optical_flow/frame_to_frame_optical_flow.h#L311-380)
- Detector: per empty grid cell, copies the sub-image into an 8-bit `cv::Mat` (L199) and runs `cv::FAST` starting at threshold 40, halving down to ≥5 until enough points (L209–232). — [keypoints.cpp L161–235](https://gitlab.com/VladyslavUsenko/basalt/-/blob/0f3b2b52c807f70ff4e2973ce253c73329eea7bc/src/utils/keypoints.cpp#L161-235)

**Pyramid and image type (basalt-headers @ b7f915b1)**
- `ManagedImagePyr::setFromImage` stores a mipmap and calls `subsample` per level (L71–80); subsampling convolves with the 5-tap kernel `{1, 4, 6, 4, 1}` separably then decimates by 2 (L91–141). — [image_pyr.h](https://gitlab.com/VladyslavUsenko/basalt-headers/-/blob/b7f915b195c87c8373bf93f97a0710bcdaffd718/include/basalt/image/image_pyr.h#L71-141)
- `Image::interp` (bilinear) and `interpGrad` (value + gradient, "smoother than interpGradBilinearExact") (L337–359). — [image.h](https://gitlab.com/VladyslavUsenko/basalt-headers/-/blob/b7f915b195c87c8373bf93f97a0710bcdaffd718/include/basalt/image/image.h#L337-359)
- Dataset loaders convert 8-bit images to 16-bit by `val << 8` (EuRoC L112, L124; KITTI L108; rosbag L140; UZH L112). — [dataset_io_euroc.h L105–130](https://gitlab.com/VladyslavUsenko/basalt/-/blob/0f3b2b52c807f70ff4e2973ce253c73329eea7bc/include/basalt/io/dataset_io_euroc.h#L105-130)

**Config flags**
- Upstream `VioConfig`: `optical_flow_type, optical_flow_detection_grid_size, optical_flow_max_recovered_dist2, optical_flow_pattern, optical_flow_max_iterations, optical_flow_levels, optical_flow_epipolar_error, optical_flow_skip_frames` (vio_config.h L48–55). Defaults: frame_to_frame, grid 50, recovered_dist2 0.09, pattern 51, iterations 5, levels 3, epipolar 0.005, skip 1 (vio_config.cpp L48–55); shipped `data/euroc_config.json` uses recovered_dist2 0.04. — [vio_config.h](https://gitlab.com/VladyslavUsenko/basalt/-/blob/0f3b2b52c807f70ff4e2973ce253c73329eea7bc/include/basalt/utils/vio_config.h#L48-55), [vio_config.cpp](https://gitlab.com/VladyslavUsenko/basalt/-/blob/0f3b2b52c807f70ff4e2973ce253c73329eea7bc/src/utils/vio_config.cpp#L48-55), [euroc_config.json](https://gitlab.com/VladyslavUsenko/basalt/-/blob/0f3b2b52c807f70ff4e2973ce253c73329eea7bc/data/euroc_config.json)
- Monado fork adds: `optical_flow_detection_num_points_cell, _min_threshold, _max_threshold, _nonoverlap, optical_flow_matching_guess_type, optical_flow_matching_default_depth, optical_flow_image_safe_radius` ("mask black corners in cameras" — relevant for fisheye), and a landmark-recall group (`optical_flow_recall_enable, _all_cams, _num_points_cell, _over_tracking, _update_patch_viewpoint, _max_patch_dist, _max_patch_norms`). Its `trackPointAtLevel` takes an extra `&error` output (L404, L457); safe radius used at L508. — [mateosss vio_config.h](https://gitlab.freedesktop.org/mateosss/basalt/-/blob/df6e970c8da7636eb401a09e3317fbeaaf829b9a/include/basalt/utils/vio_config.h), [mateosss frame_to_frame L388–520](https://gitlab.freedesktop.org/mateosss/basalt/-/blob/df6e970c8da7636eb401a09e3317fbeaaf829b9a/include/basalt/optical_flow/frame_to_frame_optical_flow.h#L388-520), recall feature MR: [MR !35](https://gitlab.freedesktop.org/mateosss/basalt/-/merge_requests/35)
- No histogram equalisation / CLAHE / gamma / preprocessing anywhere in upstream or the fork (grep for `clahe|equalizeHist|preprocess` found only calibration histogram comments). — grep of both trees at the commits above.

### Inferences (integration plan, all untested)
1. **Prefilter hook A (simplest, all variants):** filter `*new_img_vec->img_data[i].img` into a new `ManagedImage<uint16_t>` just before `setFromImage` in `processFrame` (frame_to_frame L123–131/L145–152; same pattern in `multiscale_*` L136/L156 and `patch_optical_flow.h` L131/L148). Don't modify `new_img_vec` in place: it is forwarded downstream as `transforms->input_images` (used by the GUI and, in the Monado fork, by recall). Gate it behind a new config field (e.g. `optical_flow_prefilter`), parsed in `vio_config.cpp`.
2. **Signed output problem:** a DoG is zero-mean and signed, but the pyramid type is `uint16_t` and the residual divides by the patch mean. Feeding a raw DoG would make `mean ≈ 0` → patches marked invalid (patch.h L165–168) or zero residuals (L189–190). Options: (a) add a constant offset (e.g. 32768) after the DoG — then the mean-scaling becomes a near-constant divisor and the residual degenerates to plain SSD on the band-passed image (gain invariance lost, but the DoG's low-cut already removes slow illumination); (b) write a new patch type with a zero-mean (or ZNCC) residual on a signed/float pyramid, which needs `ManagedImagePyr<float>` or `int16_t` and touching the templates.
3. **Pyramid interaction:** prefiltering level 0 then building the binomial pyramid means coarse levels see the band-pass of the band-pass (the σ=8 low-cut becomes σ=1 px at level 3). A DoG with σ_outer=8 px removes most content that level 3 (1/8 scale) relies on for large-motion capture. Alternative: build the pyramid from the raw image and prefilter each level with fixed per-level σ (hook inside or after `ManagedImagePyr::setFromImage`).
4. **Detection:** `addPoints` detects FAST on `pyramid->at(0).lvl(0)`, so a prefilter also changes detection, and keypoints.cpp truncates to 8 bit (L199). Either keep a raw level-0 copy for detection or rescale the filtered image to use the 8-bit range.
5. **Alternative matcher (e.g. phase correlation):** the narrowest swap is `trackPointAtLevel`/`trackPoint` (L243–305) — keep the coarse-to-fine loop, forward-backward check (L214–226) and epipolar filter unchanged, and replace the per-level GN update. Cleaner: a new `optical_flow_type` (e.g. `"phase_frame_to_frame"`) class copied from `FrameToFrameOpticalFlow` and registered in `OpticalFlowFactory` (optical_flow.cpp L44–110), so the backend contract (`OpticalFlowResult` observations as `AffineCompact2f`) is unchanged. Phase correlation yields translation only; the SE2 rotation part could be left identity or refined by one GN step.

### Gaps
- Whether `multiscale_frame_to_frame` (detects at all pyramid levels) is more blur-robust than `frame_to_frame` is not documented anywhere I found.
- The exact uint16→uint8 conversion in keypoints.cpp around L199 was not read line by line.

---

## Q4. OKVIS2 source: BRISK detection/matching and where images are preprocessed

### Takeaway
OKVIS2 does no image preprocessing: the camera image goes straight from `ThreadedSlam::getNextFrame` (variable literally named `filtered`, but unfiltered) into `MultiFrame::setImage`, then to a BRISK scale-space Harris detector and BRISK descriptor; matching is brute-force Hamming with a threshold of 60. A CLAHE call exists but is commented out. Prefilter hook = `getNextFrame`; an alternative matcher would be a new or replaced stage in `matchMotionStereo`/`matchStereo`/`matchToMap`.

### Cited Findings (@ a2ea0068)
- Image ingestion: `getNextFrame` copies each camera's `measurement.image` into `cv::Mat filtered` and calls `multiFrame->setImage(im, filtered)` (L251–268) — no filter is applied. — [ThreadedSlam.cpp L251–268](https://github.com/smartroboticslab/okvis2/blob/a2ea00688cd10988aae7bd52ab7935ce9a657ec0/okvis_multisensor_processing/src/ThreadedSlam.cpp#L251-L268)
- `Frame::setImage` stores the image (L69–71); `Frame::detect` calls `detector_->detect(image_, keypoints_)` (L139–151); `Frame::describe` calls `extractor_->compute(image_, keypoints_, descriptors_)` (L159–166). — [Frame.hpp](https://github.com/smartroboticslab/okvis2/blob/a2ea00688cd10988aae7bd52ab7935ce9a657ec0/okvis_cv/include/okvis/implementation/Frame.hpp#L139-L166)
- Detector/descriptor init: `brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator>(briskDetectionThreshold_, briskDetectionOctaves_, briskDetectionAbsoluteThreshold_, briskDetectionMaximumKeypoints_)` and `brisk::BriskDescriptorExtractor(rotationInvariance, scaleInvariance)`; CLAHE commented out (`//mClahe = cv::createCLAHE(3.0, cv::Size(8, 8));`, L2583; declaration commented at L107). — [Frontend.cpp L2579–2600](https://github.com/smartroboticslab/okvis2/blob/a2ea00688cd10988aae7bd52ab7935ce9a657ec0/okvis_frontend/src/Frontend.cpp#L2579-L2600)
- `detectAndDescribe` (L223) sets camera-aware extraction maps and a gravity-aligned extraction direction for the BRISK extractor before detection. — [Frontend.cpp L223–262](https://github.com/smartroboticslab/okvis2/blob/a2ea00688cd10988aae7bd52ab7935ce9a657ec0/okvis_frontend/src/Frontend.cpp#L223-L262)
- Matching: `briskMatchingThreshold_(60.0)` default (L146); Hamming distance on 48-byte descriptors `brisk::Hamming::PopcntofXORed(ddata + 48*k, old, 3)` (L339–351); `matchToMap` (L1260), `matchMotionStereo` (L1920), `matchStereo` (L2167); all templated over `PinholeCamera<RadialTangentialDistortion | EquidistantDistortion | RadialTangentialDistortion8>` (L714–724) — fisheye only via the equidistant model. — [Frontend.cpp](https://github.com/smartroboticslab/okvis2/blob/a2ea00688cd10988aae7bd52ab7935ce9a657ec0/okvis_frontend/src/Frontend.cpp#L1260)
- Config: `FrontendParameters { detection_threshold (uniformity radius px), absolute_threshold (Harris noise floor), matching_threshold, octaves, max_num_keypoints, keyframe_overlap, use_cnn, parallelise_detection, num_matching_threads }`. — [Parameters.hpp L125–135](https://github.com/smartroboticslab/okvis2/blob/a2ea00688cd10988aae7bd52ab7935ce9a657ec0/okvis_common/include/okvis/Parameters.hpp#L125-L135)
- BRISK itself is a git submodule (`external/brisk` @ 1ef8b42a) not read here. — `git submodule status` of the clone.
- In Dark EuRoC, OKVIS (v1) RMSE rose with darkness and failed/drifted at levels L10–L11, while Basalt stayed stable. — [AirSLAM, arXiv 2408.03520](https://arxiv.org/html/2408.03520)

### Inferences
- Prefilter hook: replace `cv::Mat filtered = frames.at(im).measurement.image;` with a call to a filter function (ThreadedSlam.cpp ~L261). Because detection, description and stereo/temporal matching all read `image_`, one hook covers everything. `absolute_threshold` (Harris noise floor) is the knob most tied to gain-lifted noise; a prefilter changes Harris responses, so it will need retuning.
- A signed DoG must be mapped back to 8-bit `CV_8U` for BRISK; BRISK's comparison-based descriptor is already invariant to monotonic intensity changes, so a band-pass mostly helps detection (noise suppression) rather than matching.
- Adding a patch matcher to OKVIS2 is a bigger change than in Basalt: OKVIS2 has no KLT stage; a patch/phase tracker would be either a new temporal-association step before `matchMotionStereo`, or a sub-pixel refinement of BRISK matches.

### Gaps
- OKVIS2-X (`ethz-mrl/OKVIS2-X`) was not cloned or read; whether it adds preprocessing is unknown.
- BRISK submodule smoothing parameters not checked.

---

## Q5. Existing forks/PRs with phase-based or frequency-domain tracking in VIO/SLAM front ends

### Takeaway
None found for Basalt, OKVIS2, VINS-Fusion, OpenVINS or ORB-SLAM3. Frequency-domain VO exists only as standalone pipelines (Fourier-Mellin VO, phase-correlation trajectory estimation).

### Cited Findings
- Searches for phase correlation in VINS-Fusion / OpenVINS / KLT replacements found no fork; front-end replacements found are learned (RAFT-based in VINS-Mono, DPVO, FMF-SLAM with FFT inside a learned flow). — [FMF-SLAM, arXiv 2506.18204](https://arxiv.org/html/2506.18204v2), [Deployment of VIO on UAVs, arXiv 2302.01867](https://arxiv.org/pdf/2302.01867)
- Phase-correlation-based trajectory from forward-facing video (standalone). — [arXiv 2312.12680](https://arxiv.org/pdf/2312.12680)
- Extended Fourier-Mellin VO for multi-depth scenes, aimed at feature-deprived scenes (standalone). — [ar5iv 2307.10015](https://ar5iv.labs.arxiv.org/html/2307.10015)
- A UC Riverside thesis notes Fourier-domain methods are sensitive to "frequency-dependent distortions" such as vibration/jitter. — [eScholarship qt26g7204g](https://escholarship.org/content/qt26g7204g/qt26g7204g.pdf)
- The Monado Basalt fork's notable front-end MR is landmark recall via optical flow, not frequency-domain. — [MR !35](https://gitlab.freedesktop.org/mateosss/basalt/-/merge_requests/35)
- KLT-family front ends remain the norm (granite/DLR, OV²SLAM). — [Wudenka et al. 2021](https://cvai.in.tum.de/_media/spezial/bib/wudenka2021monocular.pdf), [OV²SLAM arXiv 2102.04060](https://arxiv.org/pdf/2102.04060)

### Inferences
- A phase-correlation patch tracker in Basalt would be novel as far as public code goes; the Basalt factory pattern (Q3) makes it an isolated module.

### Gaps
- GitHub code search (not web search) for `phaseCorrelate` inside VIO repos was not run; a fork could exist without being indexed.

---

## Q6. Embedded cost of prefilters (separable Gaussian / DoG on ARM NEON)

### Takeaway
No published ms-per-frame figure was found for a separable Gaussian or DoG on a Cortex-A76 (Pi 5) at ~820x616. The nearest numbers are desktop: a 4-level image pyramid takes 0.31–0.44 ms (Bit-Planes paper), and census/binary descriptor computation 4–6 ms per frame on CPU at ~1024x376.

### Cited Findings
- Bit-Planes Table II (KITTI 1024x376, CPU with SSE2, CPU model not stated): pyramid construction 0.44 ms; raw-intensity "descriptor" 0.28 ms; Bit-Planes (census + 3x3 smoothing) 5.55 ms. — [ar5iv 1604.00990](https://ar5iv.labs.arxiv.org/html/1604.00990)
- Park et al.: GradM (Sobel) adds ~7 ms over raw on a GTX 950 (28 → 35 ms per frame, dense, unoptimised); Census 355–489 ms. — [Park 2017 Table V](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/09/park2017icra.pdf)
- Basalt full VIO 7.83 ms/frame on a 4-core Xeon (EuRoC 752x480 stereo). — [arXiv 1904.06504v3](https://arxiv.org/html/1904.06504v3)
- A fast Gaussian blur was reported 2–3x faster than OpenCV on a Raspberry Pi 3B+ (Cortex-A53); numbers on the repo page, not retrieved. — [zanazakaryaie/fastGaussianBlur](https://www.github.com/zanazakaryaie/fastGaussianBlur)
- Arm KleidiCV provides NEON/SVE2/SME Gaussian blur kernels with a benchmarking guide (Android targets). — [Arm Learning Path](https://learn.arm.com/learning-paths/mobile-graphics-and-gaming/explore-kleidicv-gaussian-blur-on-android/)

### Inferences
- A direct separable Gaussian at σ=8 needs ~49 taps per pass (±3σ), far more than Basalt's 5-tap pyramid kernel; on a Pi 5 the σ=8 term will dominate. Cheaper equivalents: (a) compute the low-pass from the pyramid (blur at level 2–3, upsample) — Basalt already builds it; (b) three running box filters (O(1) per pixel, approximates a Gaussian); (c) a recursive IIR Gaussian. These are standard techniques but their Pi 5 timings must be measured.
- Phase correlation on sparse patches (e.g. 32x32 FFT per feature, ~100–150 features x 2 cameras) is a different cost profile from dense filtering; also unmeasured.

### Gaps
- No Pi 5 / Cortex-A76 timing found for any of: separable Gaussian, DoG, CLAHE, census, at 820x616. This needs a local benchmark (median over warm runs, OpenCV built with NEON confirmed).
