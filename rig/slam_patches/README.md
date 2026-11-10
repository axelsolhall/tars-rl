# Patches to third-party SLAM code

Applied on top of the upstream checkouts in `~/dev/slam` (built by `build-basalt.sh` / `build-vio2.sh`).
Each patch is off unless its environment variable is set; `tars_scan.slam_run ... --set env.NAME=value` sets it.

| patch | upstream | what it does | switch |
|---|---|---|---|
| `basalt-stereo-hypotheses.patch` | https://gitlab.freedesktop.org/mateosss/basalt.git @ `df6e970c8d` | Stereo (cam0 → cam1) match of new points starts KLT from several depths and keeps the lowest residual; drops the point when a different position matches almost as well (repeating texture). Fixes the wrong map scale that a single depth guess caused on the board film. | `TARS_STEREO_DEPTHS=0.25,0.4,0.65,1.0,2,5`, `TARS_STEREO_AMBIGUITY=1.6` |
| `okvis2-prefilter.patch` | https://github.com/ethz-mrl/OKVIS2-X.git @ `38043e4afe` | Filters each image before detection: CLAHE or difference of Gaussians. No robust gain measured. | `TARS_PREFILTER=clahe:3:8` or `dog:1:8` |

Apply: `git -C ~/dev/slam/basalt apply $PWD/rig/slam_patches/basalt-stereo-hypotheses.patch`.
