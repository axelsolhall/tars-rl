# Phase congruency features and local-phase optical flow for VO feature detection and tracking

Scope note: "phase congruency" (PC, Kovesi) is a local-energy feature measure built from a bank of quadrature (log-Gabor or monogenic) filters. It is not the same thing as "phase correlation" (the FFT cross-power-spectrum shift estimator, OpenCV `phaseCorrelate`), which is what the rig test (0.35 px / 9% lost) used. Several papers below combine the two. All numbers are from the cited papers on their own data; none is a prediction for an IMX219 fisheye at 820x616 on a Pi 5.

## (a) Phase-congruency keypoint detectors and descriptors (RIFT, RIFT2, HOPC, LUIFT, MWF, PCSD, log-Gabor descriptors): accuracy vs SIFT/ORB/FAST under illumination, noise, blur

### Takeaway
PC-based detectors and descriptors are made for nonlinear radiometric differences between sensors (optical/SAR/IR/depth/map, day/night). On those multimodal pairs they beat SIFT by a wide margin: RIFT/RIFT2 match about 96–100% of pairs where SIFT matches 2–60%. They are slow (RIFT about 30 s per pair in MATLAB, RIFT2 about 11–13 s in C++ on a laptop i7). Their reported localisation accuracy is 1.8–2.8 px RMSE against affine ground truth. None of these papers measures sub-pixel frame-to-frame tracking precision, so they cannot be compared directly with a 0.11 px KLT result.

### Cited Findings
**RIFT (Li, Hu, Ai; arXiv 2018, IEEE TIP 2019)**
- Published under two names: "radiation-invariant feature transform" (arXiv 1804.09493) and "radiation-variation insensitive feature transform" (IEEE TIP 2019, doi 10.1109/TIP.2019.2959244). — [arXiv PDF](https://arxiv.org/pdf/1804.09493.pdf); [PubMed record](https://www.pubmed.gov/31869789/)
- Detection: RIFT computes a PC map for each log-Gabor orientation. It runs FAST on the minimum moment of the PC maps to get corner points and on the maximum moment to get edge points. The authors chose FAST "only" for time efficiency, and say other detectors would work. — [arXiv 1804.09493](https://arxiv.org/pdf/1804.09493.pdf)
- Description: the authors found the raw PC map unsuitable for description. Most PC values are close to zero, and the map "is sensitive to noise because it mainly contains edges". They instead build a Maximum Index Map (MIM): for each pixel, the index of the orientation with the largest summed log-Gabor amplitude over scales. A SIFT-like histogram of the MIM over a J×J patch forms the descriptor, which is normalised for illumination invariance. — [arXiv 1804.09493](https://arxiv.org/pdf/1804.09493.pdf)
- Default filter bank: 6 orientations (0°–150° in 30° steps). The parameter study tested 4 scales as best and patch J=96. Rotation invariance comes from building several MIMs (a "convolution sequence ring"). RIFT is not scale-invariant: images must be resampled to the same ground sample distance first. — [arXiv 1804.09493](https://arxiv.org/pdf/1804.09493.pdf)
- Evaluation: 6 multimodal remote-sensing datasets of 10 pairs each (optical-optical, IR-optical, SAR-optical, depth-optical, map-optical, day-night). Ground truth is an affine transform fitted to 5 hand-picked sub-pixel correspondences. A match counts as correct if its residual is under 3 px, and a pair fails if it has fewer than 4 correct matches. — [arXiv 1804.09493](https://arxiv.org/pdf/1804.09493.pdf)
- Success rate (%) per dataset: SIFT 50/30/30/0/20/60, SAR-SIFT 50/0/50/30/0/40, RIFT 100 on all 6. Averages are SIFT 31.7%, SAR-SIFT 28.3%, RIFT 100%. Across all 60 pairs RIFT averages NCM 122.4, ME 1.79 px and RMSE 1.94 px. Per-dataset RMSE is 1.82–1.99 px, and the authors attribute part of the error to the ground-truth model. SIFT was best on day-night (60%) "because of its resistance to illumination changes". — [arXiv 1804.09493](https://arxiv.org/pdf/1804.09493.pdf)
- Rotation test: one map-optical pair rotated 0–359° in 5° steps kept NCM above 40 at every angle. — [arXiv 1804.09493](https://arxiv.org/pdf/1804.09493.pdf)
- The arXiv preprint reports no run time. Code and datasets are linked from the authors' site. — [arXiv 1804.09493](https://arxiv.org/pdf/1804.09493.pdf); [author site](https://sites.google.com/site/jiayuanli2016whu/home)

**RIFT2 (Li et al., arXiv 2303.00319, 2023)**
- RIFT2 replaces the convolution-sequence ring with a dominant-index rotation technique and claims about 3× lower time and memory "in theory". — [ar5iv 2303.00319](https://ar5iv.labs.arxiv.org/html/2303.00319)
- Measured on a laptop with an Intel i7-8550U at 1.8 GHz and 8 GB RAM. The table gives time per pair without a unit, almost certainly seconds. SAR-optical / IR-optical: SIFT (VLFeat C++) 0.87 / 1.75; PSO-SIFT (MATLAB) 1.04 / 1.92; OS-SIFT (MATLAB) 1.78 / 2.27; RIFT (MATLAB) 29.24 / 35.82; RIFT2 (C++) 10.93 / 13.24. The paper does not state image sizes. — [ar5iv 2303.00319](https://ar5iv.labs.arxiv.org/html/2303.00319)
- Accuracy on 100 SAR-optical and 100 IR-optical pairs from the LNIFT datasets, with rotation. A correct match has a residual under 3 px, a pair needs at least 10 correct matches to succeed, and no RANSAC was used. Success rate SAR / IR: SIFT 2 / 3%, PSO-SIFT 26 / 32%, OS-SIFT 38 / 16%, RIFT 94 / 98%, RIFT2 96 / 97%. RMSE: SIFT 19.6 / 19.5 px, RIFT 3.12 / 2.45 px, RIFT2 2.79 / 2.62 px. NCM: SIFT 2 / 2, RIFT 106 / 123, RIFT2 102 / 118. — [ar5iv 2303.00319](https://ar5iv.labs.arxiv.org/html/2303.00319)
- Code: https://github.com/LJY-RS/RIFT2-multimodal-matching-rotation. The paper states no licence. — [ar5iv 2303.00319](https://ar5iv.labs.arxiv.org/html/2303.00319)

**HOPC (Ye, Shan, Bruzzone, Shen; IEEE TGRS 55(5) 2017)**
- HOPC is a dense descriptor (histogram of orientated phase congruency). Its similarity metric HOPCncc is the NCC of HOPC descriptors, used in template matching to find control points. On 6 multimodal remote-sensing pairs it beat NCC and mutual information. It needs prior georeferencing (it is a template search, not a detector). — [search summary of TGRS abstract](https://bvbvf1.bib-bvb.de/vufind/PrimoRecord/cdi_ieee_primary_7862734); [arXiv 2103.16871](https://arxiv.org/pdf/2103.16871)
- The RIFT authors criticise HOPC on three points. It needs accurate geographic information, it is sensitive to rotation and scale, and it uses Harris for detection, so the detection step stays intensity-gradient based. — [arXiv 1804.09493](https://arxiv.org/pdf/1804.09493.pdf)

**Older phase-based local features (Carneiro & Jepson, ECCV 2002 / CVPR 2003)**
- A descriptor built on the phase and amplitude of complex steerable filters outperformed local differential invariants under common illumination changes. In the CVPR 2003 version it also outperformed SIFT under illumination change, 2D rotation and sub-pixel translation. SIFT and differential invariants were better under scale and large shear. — [ECCV 2002 PDF](https://www.cs.toronto.edu/~jepson/papers/CarneiroJepsonECCV2002.pdf); [CVPR 2003 entry](https://mlanthology.org/cvpr/2003/carneiro2003cvpr-multi)

**Phase-based corner detection vs Harris/SUSAN**
- Zang et al. (DAGM 2006) report that a monogenic-curvature phase-based corner detector beats Harris and SUSAN under illumination change and noise, on the reasoning that local phase is independent of local amplitude. I could not open the PDF, so I have no numbers. — [Zang DAGM 2006](https://www.informatik.uni-kiel.de/inf/Sommer/doc/Publications/zd/dagm2006_Zang.pdf) (link from search; fetch refused)
- FAST-n (FAST-9) "is not very robust to the presence of noise", per its authors' own repeatability study. — [Rosten et al., arXiv 0810.2434](https://arxiv.org/pdf/0810.2434)

### Inferences
- The PC/MIM line of work optimises for invariance to how a surface maps to grey levels across sensors. A dark, gain-boosted visible camera has a different problem: the radiometric change between consecutive frames is roughly monotonic, while SNR and blur are bad. KLT with gain/bias compensation or ZNCC already handles monotonic changes. The extra invariance of PC is therefore mostly unneeded for frame-to-frame tracking, and its noise sensitivity (stated by the RIFT authors themselves) works against it.
- The 1.8–2.8 px RMSE figures come from matching heterogeneous images under 3 px inlier thresholds with an approximate affine ground truth. They say nothing about frame-to-frame sub-pixel precision, which is what KLT's 0.11 px measures.
- A cheap way to get the illumination-normalisation part of PC is the MIM or orientation-index idea, or simply contrast normalisation (ZNCC, census). Neither needs the full multi-scale log-Gabor bank.

### Gaps
- **LUIFT** ("luminance invariant feature transform"): a typeset/scispace PDF exists ([link](https://typeset.io/pdf/luift-luminance-invariant-feature-transform-3cf9xmq3m2.pdf)), but the fetch returned empty. I did not verify the authors, venue, method or numbers.
- **MWF and PCSD:** I found no source with numbers in this session. I could not confirm which papers these acronyms refer to.
- **Repeatability vs FAST/ORB:** I found no head-to-head numbers for PC detectors against FAST/ORB on standard repeatability benchmarks (Oxford/HPatches) under illumination, noise or blur. The phase-vs-Harris claims (Zang 2006) come from abstracts only.
- **HOPC numbers:** I found no HOPC correct-match rates or run times (full text not fetched).

## (b) VO/SLAM or feature trackers that use phase congruency for keypoints

### Takeaway
I found very few such systems, and all are thermal. A low-resolution (80×60) thermal stereo pipeline used PC keypoints plus phase correlation in PC space for sub-pixel refinement. A 2025 thermal SLAM paper (WTI-SLAM) reportedly uses multi-scale phase-congruency features. I found no visible-light low-light, endoscopy or underwater VO/SLAM that uses PC keypoints with published numbers against ORB/FAST/KLT.

### Cited Findings
- **Thermal sub-pixel stereo (arXiv 1912.00138; Infrared Physics & Technology 2020).**
  - Setup: two uncooled FLIR Lepton 2 cameras (80×60 px, 51° HFOV, at most 8 fps, 16 mm baseline). The authors use PC keypoints because the images are low-texture and noisy, with sudden brightness jumps from unsynchronised flat-field corrections. — [arXiv 1912.00138](https://arxiv.org/pdf/1912.00138)
  - Detector timing on an 80×60 image, Intel Core i7-3687U, C++/GCC 7, mean of 1000 runs: FAST 77 features in 90 ± 7 µs; ORB 117 in 245 µs; Shi-Tomasi 120 in 189 µs; GFTT 120 in 350 µs; SURF 14 in 1436 µs; KAZE 33 in 13.2 ms; PC 275–1734 features (threshold 0.3–0.01) in 5224 ± 480 µs. In the table all three PC rows are labelled "PhaseCong(0.01)", an apparent typo. — [arXiv 1912.00138](https://arxiv.org/pdf/1912.00138)
  - The authors call PC's cost "relatively high computation time compared to techniques such as FAST". A stereo pair took about 10 ms, which they judged acceptable at 8 fps. — [arXiv 1912.00138](https://arxiv.org/pdf/1912.00138)
  - They cite earlier work that cut PC time by downscaling from 320×240 to 160×120 and 80×60. — [arXiv 1912.00138](https://arxiv.org/pdf/1912.00138)
  - Sub-pixel refinement is a modified phase correlation applied in the phase-congruency feature space. Projecting a standing person's points gave a standard deviation of about 300 mm, against more than 1000 mm for ORB + OpenCV KNN. The method found 4× more matches than ORB + KNN on low-resolution images. — [arXiv 1912.00138](https://arxiv.org/pdf/1912.00138)
- **WTI-SLAM (Complex & Intelligent Systems, April 2025).**
  - A thermal-IR SLAM for weak-texture images. It reportedly uses multi-scale phase consistency (congruency) feature extraction and cites Kovesi's PC model. Its odometry combines feature-based methods with optical flow. — [Springer](https://link.springer.com/article/10.1007/s40747-025-01858-0) (via search summary; page behind redirect); [DOAJ](https://doaj.org/article/3fc3078a09e3425f9f21c29d545753d6)
  - Evaluated on the JPL, Airey and ViViD++ thermal datasets. It reports about 46% lower localisation error than mainstream thermal SLAM, and that ORB-SLAM2 finds few features and fails to initialise in low-texture areas. — [DOAJ](https://doaj.org/article/3fc3078a09e3425f9f21c29d545753d6)
- **Underwater acoustic (sonar) imagery.** A detector comparison found ORB and BRISK detectors best overall and FAST fastest. Running detectors on a PC layer or a Sobel layer was "most favorable". The imagery is acoustic, not optical. — [MDPI JMSE 10(11):1601](https://www.mdpi.com/2077-1312/10/11/1601)
- **Low-light visible SLAM.** The literature found instead fixes ORB/FAST with adaptive thresholds or image enhancement (AFE-ORB-SLAM, IRAF-SLAM), or switches to learned features. None of these uses PC. — [AFE-ORB-SLAM](https://link.springer.com/article/10.1007/s10846-022-01645-w); [IRAF-SLAM](https://arxiv.org/pdf/2507.07752); [Light-SLAM](https://arxiv.org/html/2407.02382v1)

### Inferences
- The thermal-stereo timing ratio is PC about 58× slower than FAST, at 80×60 on a 2012 laptop i7. Scaling that to 820×616 (105× more pixels, FFT-based so slightly worse than linear) puts PC detection on one frame on the order of 0.5 s on that CPU. That is my extrapolation, not a measurement. A Pi 5 Cortex-A76 is likely in the same order of magnitude per core. Full-frame PC per frame at 30 fps × 2 cameras therefore looks infeasible on CPU without heavy downscaling, fewer scales and orientations, or ROI-only computation.
- The thermal paper's design, PC keypoints followed by phase-correlation refinement in PC space, is the closest published analogue to "phase-based patch tracking". The authors did not report sub-pixel error against ground truth, only depth spread.

### Gaps
- I could not open the WTI-SLAM full text, so I have no ATE tables, per-frame run time or hardware, and no confirmation of exactly where PC sits in its pipeline.
- I found no endoscopy or optical-underwater VO using PC keypoints. This is a search-coverage gap, not proof that none exist.
- I found no study comparing PC keypoints with KLT on frame-to-frame track length or drift.

## (c) Local-phase optical flow (Fleet & Jepson 1990, Gautama & Van Hulle 2002, monogenic flow, steerable-pyramid phase): accuracy, robustness, cost

### Takeaway
On the classic 1994 benchmark, Fleet & Jepson phase flow was the most consistently accurate method on synthetic sequences. Translating Tree error was 0.23–0.32°, against several degrees for most others. On Yosemite it was no better than Lucas–Kanade (4.29° at 34% density vs LK 3.05° at 8.7% or 4.10° at 35%). It also uses 21 frames of temporal support, which is bad for causal VO latency. Monogenic, two-frame variants are much more robust to ±50% brightness change than intensity-based CLG (Yosemite error about 14–16° vs 47–52°). Under heavy noise they degrade (3.4° clean to 14.2° at SNR 10 dB).

### Cited Findings
**Barron, Fleet & Beauchemin, "Performance of optical flow techniques", IJCV 12(1) 1994**
- Implementation of Fleet & Jepson used: a single scale tuned to a spatiotemporal wavelength of 4.25 px-frames ("a more complete implementation would normally have 3–5 scales"), temporal support of 21 frames, and a phase-stability threshold τ. Only 2-D velocities with LS condition number below 10 and residual below 0.5 are kept. — [IJCV 1994 PDF](https://www.csd.uoc.gr/~hy576/bibliography/PerformanceOpticalFlow.pdf)
- Translating Tree (average angular error ± std, density):
  - Fleet & Jepson τ=2.5: 0.32° ± 0.38, 74.5%; τ=1.25: 0.23° ± 0.19, 49.7%; τ=1.0: 0.25° ± 0.21, 26.8%.
  - Heeger level 1: 4.53°, 57.8%. Anandan: 4.54°, 100%. Singh step 2: 1.11°, 99.6%.
  - — [IJCV 1994 PDF](https://www.csd.uoc.gr/~hy576/bibliography/PerformanceOpticalFlow.pdf)
- Diverging Tree: Fleet & Jepson τ=2.5 gave 0.99° ± 0.78 at 61.0%, τ=1.25 0.80° at 46.5%, τ=1.0 0.73° at 28.2%. Heeger gave 4.95° at 73.8%. — [IJCV 1994 PDF](https://www.csd.uoc.gr/~hy576/bibliography/PerformanceOpticalFlow.pdf)
- Yosemite, 2-D velocity:
  - Fleet & Jepson τ=1.25: 4.95° ± 12.39, 30.6%; τ=2.5: 4.29° ± 11.24, 34.1%.
  - Lucas–Kanade (λ₂ ≥ 1.0): 4.10° ± 9.58, 35.1%; (λ₂ ≥ 5.0): 3.05° ± 7.31, 8.7%.
  - Horn–Schunck original: 32.43°, 100%. Uras thresholded: 6.73°, 14.7%. Anandan: 15.84°, 100%.
  - — [IJCV 1994 PDF](https://www.csd.uoc.gr/~hy576/bibliography/PerformanceOpticalFlow.pdf)
- The authors' own reading: phase-based "produced the most consistently accurate results" on synthetic data but "is not significantly better than differential methods on the Yosemite sequence". Their reasons: only 15 frames were available, so the filters had to be retuned closer to Nyquist, which made them more sensitive to aliasing; and Yosemite has aliasing and the horizon occlusion. Excluding the sky, LK improves to 2.80°. — [IJCV 1994 PDF](https://www.csd.uoc.gr/~hy576/bibliography/PerformanceOpticalFlow.pdf)

**Gautama & Van Hulle, IEEE TNN 2002**
- The method filters spatially only, with quadrature pairs (temporal filtering removed), and estimates the temporal gradient of phase. It rejects component velocities whose phase is non-linear over time. It was chosen for robustness to lighting change and to deviations from pure translation. — [abstract mirror](https://data.zhenzhi.com/details/1/902125390); [Wadhwa et al. SIGGRAPH 2013 description](https://people.csail.mit.edu/mrub/papers/phasevid-siggraph13.pdf)
- A MATLAB implementation is on MathWorks File Exchange (#2422, "Phase-based Optical Flow"). — [File Exchange](https://ch.mathworks.com/matlabcentral/fileexchange/2422-phase-based-optical-flow)

**Monogenic signal flow (Felsberg; Zang et al.; Alessandrini et al.)**
- Felsberg, "Optical flow estimation from monogenic phase" (IWCM 2004, LNCS 3417): two-frame flow from monogenic phase, which avoids the orientation-dependent phase error of fixed-orientation quadrature filters. Sub-pixel accuracy comes from a multi-resolution analysis with 7 filter responses per scale. Its experiments are described as "basic", and I did not see any numbers. — [Springer/unpaywall entry](https://unpaywall.org/10.1007%2F978-3-540-69866-1_1)
- Zang, Wietzke, Schmaltz, Sommer, "Dense optical flow estimation from the monogenic curvature tensor" (SSVM 2007), on Yosemite with clouds at 100% density:
  - Average angular error ± std: 2D CLG 4.86° ± 8.48; 3D CLG 4.17°; ours 2D (γ=0) 3.37° ± 8.27; ours 2D (γ=0.1) 3.25°; ours 3D 2.67° ± 7.12.
  - — [SSVM 2007 PDF](https://www.mia.uni-saarland.de/Publications/zang-ssvm07.pdf)
- Same paper, illumination: frame 8 brightened by 50% gave 2D CLG 46.94° against monogenic 13.50°. Darkened by 50% it gave 2D CLG 52.14° against monogenic 15.83°. A sweep over relative grey value showed CLG "very sensitive" and the phase method "very robust". — [SSVM 2007 PDF](https://www.mia.uni-saarland.de/Publications/zang-ssvm07.pdf)
- Same paper, noise: additive Gaussian noise on frame 8 at SNR 10 dB gave 14.16° ± 12.76, against 3.37° clean. The error rose steadily as SNR fell from 40 to 10 dB. — [SSVM 2007 PDF](https://www.mia.uni-saarland.de/Publications/zang-ssvm07.pdf)
- Alessandrini et al., IEEE TIP 2013 (cardiac ultrasound/MR): monogenic-phase flow with a robust least-squares orientation estimate (for noise) and coarse-to-fine B-spline/pyramid refinement for large motions. The authors report it more accurate and faster than the Zang monogenic-curvature-tensor method. — [TIP 2013 PDF](https://www.creatis.insa-lyon.fr/~basarab/img/Alessandrini_TIP13.pdf)

**Steerable-pyramid phase**
- Wadhwa et al. (SIGGRAPH 2013) use complex steerable pyramid phase for motion processing and describe Gautama–Van Hulle as the phase-gradient flow method. — [SIGGRAPH 2013 PDF](https://people.csail.mit.edu/mrub/papers/phasevid-siggraph13.pdf)

### Inferences
- **Flow accuracy.** Phase-based flow is very accurate when motion is small relative to filter wavelength and the signal is band-limited (Translating Tree, 0.23°). It loses its edge on aliased or occluding real-ish content (Yosemite). Our rig's blurred, noisy frames are band-limited by blur, which helps, but noise at high gain hurts. This matches the rig finding that DoG band-pass helped phase correlation under blur.
- **Illumination.** The robustness advantage is clear against brightness-constancy methods with no photometric model, such as CLG. It is not shown against KLT with an affine gain/bias model or ZNCC, which already absorb global ±50% changes. I found no benchmark of that comparison.
- **Noise.** A 4× degradation from clean to SNR 10 dB (Zang) suggests phase flow is not inherently noise-immune. Phase is undefined where local amplitude is near zero, so methods need an amplitude or stability threshold. This drops density, as Fleet & Jepson's 26–75% density shows.

### Gaps
- I could not fetch Gautama & Van Hulle's own result tables (host refused). I have no Yosemite number for it here.
- I found no Middlebury (Baker et al.) or KITTI entries for classical phase-based flow in this session.
- Neither Felsberg's IWCM 2004 chapter nor the HAL "Multiscale optical flow computation from the monogenic signal" paper could be read (Anubis block), so I have no numbers for them.

## (d) Cost on CPUs / embedded hardware, and real-time implementations

### Takeaway
Phase methods reached real time only with massive parallelism: GPU (CUDA, 640×512 "and beyond", 2008) or FPGA. On CPU the only concrete per-image figure is 5.2 ms for PC on an 80×60 image (FAST: 90 µs). I found no Raspberry Pi or ARM real-time PC or phase-flow implementation.

### Cited Findings
- Pauwels & Van Hulle (CVPRW 2008), "Realtime phase-based optical flow on the GPU": a CUDA implementation that checks phase consistency over time and produces dense, reliable flow "in realtime at high resolutions (640×512 pixels and beyond)". The abstract gives no fps or GPU model. — [mlanthology](https://mlanthology.org/cvprw/2008/pauwels2008cvprw-realtime); [hgpu](https://hgpu.org/?p=3159)
- Pauwels, Tomasi, Díaz, Ros, Van Hulle (IEEE Trans. Computers 61(7) 2012) compare FPGA and GPU architectures for phase-based flow, stereo and local features (energy, orientation, phase). The abstract says this needs "a massive degree of parallelism to achieve real-time performance". I did not retrieve fps numbers. — [hgpu](https://hgpu.org/?p=4724); [UGR record](https://produccioncientifica.ugr.es/documentos/61b4ff8fff6ada1aa9a0c9d0?lang=en)
- A 2015 AMOS poster reports a real-time FPGA + CMOS-sensor PC implementation on sky data. I saw no fps figure. — [AMOS 2015 poster](https://amostech.com/TechnicalPapers/2015/Poster/Lambert1.pdf)
- Struyf et al. (2014) compare GPU (OpenCL) and FPGA (VHDL) versions of a line detector built on oriented PC energy. I did not get the numbers. — [SciTePress 2014](https://www.scitepress.org/Papers/2014/47303)
- CPU: PC on 80×60 in 5224 ± 480 µs vs FAST in 90 ± 7 µs (i7-3687U, C++). — [arXiv 1912.00138](https://arxiv.org/pdf/1912.00138)
- CPU: RIFT2 10.9–13.2 s per pair in C++ and RIFT 29–36 s in MATLAB on an i7-8550U, against SIFT at 0.9–1.8 s. — [ar5iv 2303.00319](https://ar5iv.labs.arxiv.org/html/2303.00319)
- Kovesi says `phasecongmono` (monogenic filters) "has excellent speed and much reduced memory requirements compared to the other phase congruency functions". `phasecong3` supersedes `phasecong`/`phasecong2` as "faster and requiring less memory". — [Kovesi matlabfns (Codeberg)](https://codeberg.org/peterkovesi/matlabfns)

### Inferences
- The cost driver is one forward FFT plus N_scales × N_orientations inverse FFTs per image: 4 × 6 = 24 for RIFT's settings. The monogenic version needs only about N_scales × 3 (one even plus two Riesz-odd responses per scale), which is why `phasecongmono` is faster. This reasoning follows from the method structure, not from a measured Pi number.
- For a patch tracker (about 100–300 tracks), computing local phase only in small ROIs around tracks (e.g. 32×32 patches, a few scales) would cost far less than full-frame PC. I found no published ROI-based phase tracker with timings.

### Gaps
- I found no published fps for any PC detector or phase-flow method on a Raspberry Pi, Cortex-A, or any embedded CPU.
- I did not obtain the numerical fps and resolution results from Pauwels 2008 or 2012 (abstracts only).

## (e) Open-source implementations and licences; could they replace FAST/KLT in a patch-tracking front end?

### Takeaway
Kovesi's MATLAB functions (phasecong3, phasecongmono, monofilt, gaborconvolve) and their MIT-licensed Python port `phasepack` are the reference implementations. RIFT and RIFT2 code is public (licence not stated). A Gautama–Van Hulle MATLAB flow is on File Exchange. Nothing I found is a drop-in, real-time replacement for FAST+KLT: the existing tools are detectors or descriptors for offline multimodal registration, or dense flow requiring GPU or long temporal support.

### Cited Findings
- Kovesi's MATLAB functions are now hosted on Codeberg. In the PhaseCongruency folder: phasecongmono.m, phasecong3.m (the edge and corner detector; phasecong/phasecong2 deprecated), monofilt.m, highpassmonogenic.m, phasesym.m / phasesymmono.m, gaborconvolve.m, spatialgabor.m and noisecomp.m (phase-preserving denoising). The repo has a LICENSE file, but its text was not shown to me. — [Codeberg peterkovesi/matlabfns](https://codeberg.org/peterkovesi/matlabfns)
- Python `phasepack` 1.5 on PyPI: phasecong, phasecongmono, phasesym and phasesymmono. Ported from Kovesi's MATLAB by Alistair Muldal, it uses pyFFTW when available and is MIT-licensed (per the search-result summary; the PyPI page did not render for direct verification). — [PyPI phasepack](https://pypi.org/project/phasepack)
- RIFT2 C++/MATLAB code: https://github.com/LJY-RS/RIFT2-multimodal-matching-rotation, with no licence stated in the paper. — [ar5iv 2303.00319](https://ar5iv.labs.arxiv.org/html/2303.00319)
- Gautama–Van Hulle phase-based optical flow, MATLAB (File Exchange #2422). — [File Exchange](https://ch.mathworks.com/matlabcentral/fileexchange/2422-phase-based-optical-flow)

### Inferences
- **Detector swap (FAST → PC corners).** Feasible only in a reduced form (downscaled image, 2–3 scales, monogenic filters), and the evidence that it improves things on dark visible-light frames is thin. The thermal results show more features, not better tracks. Keypoint location on a PC map is tied to edge and corner structure at the filter scale, so it may be more stable under blur than FAST's intensity-ring test. That is plausible but unmeasured in the sources.
- **Tracker swap (KLT → local-phase tracking).** The closest published options are (i) two-frame monogenic-phase flow (Felsberg; Alessandrini) evaluated in ROIs around tracks, and (ii) phase correlation on band-passed or PC-transformed patches (the thermal paper). The literature does not show either beating KLT's sub-pixel precision on visible-light data. The rig's own measurement (KLT 0.11 px median, 0% lost vs phase correlation 0.35 px, 9% lost) is the most relevant evidence available.
- **Cheapest likely-useful idea from this literature.** Band-pass (log-Gabor or DoG) or phase-normalise the patch before KLT or ZNCC, to get part of the illumination and blur benefit without a full PC pipeline. That is an inference to test, not a published result.

### Gaps
- I did not verify the licence text of Kovesi's Codeberg repo or of the RIFT/RIFT2 GitHub repos.
- I found no C++/OpenCV-contrib PC implementation with benchmarks in this session.
- I found no published comparison of phase-based patch tracking with KLT on standard VO datasets (EuRoC, TUM-VI) or on low-light visible sequences.
