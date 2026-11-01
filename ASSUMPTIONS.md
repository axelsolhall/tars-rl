# Assumptions

IDs match loopdesk. "Confirmed" = decided by Axel; "proposed" = a default picked to keep moving.

## TARS body (`tars_rl.body`, T3)

| ID | Assumption | Status | Depends | Where | How to test | If wrong |
|---|---|---|---|---|---|---|
| A1 | Simulation and training run in MuJoCo with MuJoCo Playground (MJX/JAX). | confirmed | | `tars_rl.body` | eval section 4: MJX vs CPU agreement and MJX steps/s | swap simulator; BodySpec stays, MJCF generator is replaced |
| A2 | Each hinge is driven by a position actuator, one float target angle per step. | confirmed | | `mjcf._actuator`, `spec.Servo` | eval section 2 tracking | change actuator type in `_actuator` |
| A5 | The hinge axis sits at mid-height of the slabs. | confirmed | | `mjcf.build_mjcf` (`hinge_z`) | `test_hinges_share_axis_through_centre` | one line in `build_mjcf` |
| A7 | Default body size is printable scale (about 30–40 cm tall); default 36 cm. | confirmed | | `spec.default_spec` | `test_defaults_follow_assumptions` | edit `default_spec` / pass another JSON |
| A8 | The body contract is an MJCF generated from a versioned BodySpec JSON (joint names, ranges, actuator order). | confirmed | | `spec.py`, `mjcf.py`, `body/assets/tars_v1.{json,xml}` | `test_committed_assets_match_generator`, `test_model_matches_spec` | |
| A9 | uv + pyproject.toml, src/ layout, ruff, pytest. | confirmed | | repo | `uv run ruff check .`, `uv run pytest` | |
| A14 | One central rigid piece plus two side arms, each on one hinge. | confirmed | A1 | `mjcf.build_mjcf` | viewer | new BodySpec version |
| A39 | Both arm hinges share one horizontal axis (y) through the centre of the central body. | confirmed | A14 | `mjcf.build_mjcf` | `test_hinges_share_axis_through_centre` | |
| A40 | Each arm has a slide joint along its own length (hinge point moves along the arm). | confirmed | A14 | `mjcf.build_mjcf` (slide after hinge) | `test_slide_moves_arm_along_its_length_when_swung` | |
| A41 | Each slide is a position actuator (one float target per step), modelling the non-backdrivable screw. | confirmed | A2 | `mjcf._actuator`, `slide_servo` | eval section 2 tracking | |
| A42 | Default slide range is ±25 % of arm length from centred (±90 mm at 36 cm). | confirmed | A8 | `spec.default_spec` | `test_defaults_follow_assumptions` | |
| A53 | Default slab sizes and masses: central 9×9×36 cm, 0.8 kg; arms 9×4.5×36 cm, 0.35 kg each; 4 mm gap. | proposed | A7 | `spec.default_spec` | weigh/measure the printed parts later | edit `default_spec`; regenerate assets |
| A54 | Default hinge range is ±150°. | proposed | A8 | `spec.default_spec` | do trained policies hit the limit? | edit `default_spec` |
| A52 | At the neutral pose (all targets 0) every slab stands on the floor: slabs are equal height and slides are centred. | proposed | A5, A40 | `mjcf.build_mjcf`, `spec.neutral` | eval section 1 | set `spec.neutral` or slab heights |
| A55 | Servo defaults: hinge kp 8 Nm/rad, kv 0.3, ±3 Nm; slide kp 3000 N/m, kv 300, ±20 N (a slow screw: full stroke ≈ 0.25 s). | proposed | A2, A41 | `spec.default_spec` | eval sections 2–3: rise time / error, tilt while lifting | edit `default_spec` |

Measured 2026-10-31: the first slide default (kv 40, ±80 N) threw an arm at up to 3.5 m/s, and the
recoil tipped the body over when an arm was lifted ≥ 60 mm. With kv 300 / ±20 N it stays within 0.6°.
Driving a slide fully *down* (−90 mm) still tips the body; that is expected, since it pushes one arm
9 cm into the floor. MJX vs CPU agree to ≈ 1 mm on an upright rollout, but while the slide force
is saturated MJX lags CPU by up to ~4 mm for ~0.1 s (check this again if T14's sim-to-sim numbers look off).

## Flag arena (`tars_rl.env`, T13)

| ID | Assumption | Status | Depends | Where | How to test | If wrong |
|---|---|---|---|---|---|---|
| A18 | Reward = progress toward flag + upright bonus − fall penalty − action effort, weights in one config. | confirmed | | `env.config.RewardConfig`, `FlagEnv.step` | `test_reward_terms_sum`; learning curves vs flag distance | edit `RewardConfig` / the `metrics` dict in `step` |
| A19 | The flag observation comes from one replaceable sensor function. | confirmed | | `env.sensors`, `FlagEnv(flag_sensor=...)` | `test_sensor_is_swappable` | |
| A22 | Flag at a uniform random direction, 3–10 body heights from the start. | confirmed | | `FlagEnv.sample_flag`, `EnvConfig.flag_*_heights` | `test_flag_placement_follows_a22` | |
| A23 | Success when the central body's centre is within one body height of the flag (xy). | confirmed | | `FlagEnv.step`, `EnvConfig.success_heights` | `test_success_when_flag_within_one_height` | |
| A24 | Fall when the central body's up axis tilts > 60° from vertical. | confirmed | | `FlagEnv.step`, `EnvConfig.fall_tilt_rad` | `test_fall_when_tilted_past_60_degrees`; eval fall rate | |
| A43 | One action per BodySpec actuator in actuator order (both hinges, both slides). | confirmed | A8, T3 | `FlagEnv.action_to_ctrl` (checks actuator order) | `test_sizes` | |
| A44 | Observation = flag vector + joint pos/vel (actuator order) + gravity in body frame + body angular velocity. | confirmed | A8, A14 | `FlagEnv.observe` (layout in module docstring) | `test_reset_observation` | |
| A50 | Flat, infinite floor, no obstacles. | confirmed | | T3's floor plane, `env.arena` | | |
| A51 | The rollout file stores the flag position next to TARS's poses. | confirmed | A10 | `env.rollout.FlagRollout` | `test_rollout_roundtrip_keeps_trajectory_keys` | |
| A63 | Actions are in [−1, 1], mapped linearly onto each actuator's ctrlrange (0 = centred target). | proposed | A43 | `FlagEnv.action_to_ctrl` | `test_action_maps_onto_ctrlrange` | change `action_to_ctrl` only |
| A64 | Policy step 0.02 s (10 physics steps); episodes time out after 20 s (1000 steps). | proposed | | `EnvConfig.ctrl_dt`, `episode_seconds` | eval timeout rate / length for trained policies | edit `EnvConfig` |
| A65 | Flag observation = flag position relative to the central body, in its own frame (3 floats, m). | proposed | A19, A44 | `sensors.ground_truth_flag` | `test_sensor_is_swappable` | change the sensor (and `FLAG_OBS_DIM`) |
| A66 | A flag rollout is the Trajectory .npz (T9 keys) plus `flag_pos`, one frame per policy step. | proposed | A10, A51, T9 | `env.rollout` | T9 renders it with the flag | move `flag_pos` into `tars_rl.viz.Trajectory` |
| A67 | The env is a small hand-written MJX reset/step class, not a `mujoco_playground` dependency. | proposed | A1 | `env.flag_env` | T14 trains on it through a thin wrapper | subclass Playground's `MjxEnv`; bodies of reset/step stay |
| A68 | Each episode starts TARS in the same neutral pose at rest at the origin. | proposed | | `FlagEnv.reset` | held-out-seed success of trained policies | add qpos noise in `reset` |

Reward weights (progress 1.0 per m, upright 0.02, fall 1.0, effort 0.005) are untuned values that
run; T14 tunes them.

## TARS in Unreal (`tars_rl.unreal` + `unreal/TarsRl`, T10)

| ID | Assumption | Status | Depends | Where | How to test | If wrong |
|---|---|---|---|---|---|---|
| A56 | TARS runs in Unreal under Unreal's own physics (Chaos); later the trained policy runs inside Unreal. | confirmed | A1 | `unreal/TarsRl` | eval with `--unreal`: stand time, pose gap | stream MuJoCo poses into Unreal instead |
| A57 | Unreal builds the TARS actor from the same BodySpec JSON as the MuJoCo model. | confirmed | A8 | `unreal.export`, `TarsActor::LoadSpec` | `tests/unreal` (exported copy not stale) | — |
| A69 | Each arm hangs on two chained Chaos constraints (central→hinge→invisible 10 g carriage→slide→arm), standing in for MuJoCo's hinge-then-slide joints. | proposed | A56, A57 | `TarsActor::BuildBody` | per-joint tracking error in Unreal | change `BuildBody` only |
| A70 | Comparison sequence: 10 s neutral stand, then ±0.5 rad hinge swings, ±4 cm slides, opposite hinges, one combo; 0.5 s ramps, 1.5 s holds (32 s). | proposed | | `unreal.sequence.default_sequence` | MuJoCo reference stands all 32 s | edit `default_sequence`, re-export |
| A71 | The Unreal project is C++ for UE 5.5 on the Mac, engine template map, 60 Hz fixed frame rate, physics substeps ≤ 2 ms. | proposed | | `TarsRl.uproject`, `Config/DefaultEngine.ini` | it builds and plays | edit uproject / ini |
| A72 | Unreal drive gains = BodySpec SI gains × factors in actor properties (angular ×1e4 incl. torque limit, linear ×1, force limit ×100). | proposed | A56 | `ATarsActor` `AngularGainScale` / `LinearGainScale` | Unreal tracking error vs MuJoCo's | change the properties in the editor |
| A73 | Unreal records slab poses in its own frame (cm, y right, FQuat xyzw); only the Python eval converts to MuJoCo's frame. | proposed | | `TarsActor::RecordSample`, `unreal.poses.from_unreal_csv` | `tests/unreal` round trip | change `from_unreal_csv` only |
| A74 | Hinge / slide drive directions sit behind `HingeSign` / `SlideSign` actor properties, default +1. | proposed | | `ATarsActor` | eval per-joint corr ≈ +1 (≈ −1 means flip) | set the property to −1 |

## Self-guiding search (`tars_rl.tune`, T17)

| ID | Assumption | Status | Depends | Where | How to test | If wrong |
|---|---|---|---|---|---|---|
| A75 | The tuner is Bayesian optimisation in a box that re-centres on the best point and resizes. | confirmed | | `tune.search.BoxSearch` | `eval_search` table: box vs random, same tries | swap `BoxSearch._propose` / `tell` |
| A76 | Pick-one settings are numbers in [0, 1] split into equal bins, in the same box as numeric settings. | confirmed | | `tune.space.Choice` | `test_choice_bins_split_unit_range_equally`; eval `mixed5` | change `Choice.decode/encode` |
| A88 | The GP is fit to score ranks mapped to normal quantiles; batches are Thompson-sampling picks among candidates in the box. | proposed | A75 | `search._normal_scores`, `BoxSearch._propose` | eval table (rosenbrock4 has huge values) | edit those two functions |
| A89 | The random baseline samples the whole space; the box starts 0.4 wide in the middle. | proposed | | `search.random_search`, `eval_search` | eval section 2 starting `dist` | change the baseline or start box in the eval |
| A90 | Box rules: start 0.4 of each range, ×2 after 2 improving rounds, ÷2 after 3 non-improving ones, restart at 0.4 around the best below 0.4/32. | proposed | A75 | `search.SearchConfig`, `BoxSearch.tell` | eval section 2: shrinks only once the gap stalls | edit `SearchConfig` defaults |

Measured 2026-10-31 (`--budget 200 --repeats 5 --seed 0`, ≈ 70 s on the Mac): the box is closer to
the known best than random search on 8/8 functions at 200 tries (e.g. hartmann6 gap 0.0019 vs 1.07,
ackley5 2.35 vs 12.3 with the optimum outside the starting box).

## First training run (`tars_rl.train`, T15)

| ID | Assumption | Status | Depends | Where | How to test | If wrong |
|---|---|---|---|---|---|---|
| A58 | TARS learns with MuJoCo Playground's ready-made PPO trainer (brax). | confirmed | A1 | `train.run`, `train.adapter` | full run beats random and stand-still in `train.eval` | swap the trainer in `run.train`; adapter, eval and policy folder stay |
| A59 | Training settings start as copies of Playground's settings for its walking robots (Go1 joystick), changed only if TARS fails. | confirmed | A1 | `train.config.PROFILES["full"]` | learning curve in `progress.jsonl` | edit the `full` profile |
| A60 | A trained policy is one folder that also records what it was trained with (`policy.pkl`, `meta.json`, `progress.jsonl`, `rollout.npz`). | confirmed | A8 | `train.policy` | `test_policy_folder_roundtrip` | change `save_policy` / `load_policy` |
| A61 | Each run logs progress to one file at regular intervals (`progress.jsonl`: PPO metrics plus our flag eval on fixed held-out flags). | confirmed | | `train.run` | `test_read_progress`; eval prints the curve | change `progress_fn` / `policy_params_fn` |
| A62 | Full runs on the 5080 are started from the Mac by one SSH script. | confirmed | | `scripts/train_remote.sh` | `start` / `status` / `fetch` against the PC | edit the script |

Training uses Playground's auto-reset with `full_reset=True` (`run.WRAP_FOR_TRAINING`), so every
episode draws a new flag as A22 says; the default would replay each env's first flag all run.

## Rollout videos (`tars_rl.viz`, T9 / T16)

| ID | Assumption | Status | Depends | Where | How to test | If wrong |
|---|---|---|---|---|---|---|
| A10 | A rollout is a versioned Trajectory .npz: per-frame qpos, timestep, BodySpec version. | confirmed | A8 | `viz.trajectory` | `test_trajectory_roundtrip` | change `Trajectory` and its loaders |
| A46 | Quick-check MP4s use MuJoCo's offscreen renderer and imageio-ffmpeg. | confirmed | A1 | `viz.render` | `viz.eval` frame count OK | swap `render_mp4` |
| A93 | Flag-arena videos use one fixed camera per clip that frames TARS's whole path and the flag, instead of following TARS. | proposed | A46, A51 | `viz.flag.framing_camera` | `test_render_flag_mp4_shows_flag_and_body`; watch `--flag-sample` | change `framing_camera` or pass another camera to `render_mp4` |

## Network builder (`tars_rl.train.networks`, T18)

| ID | Assumption | Status | Depends | Where | How to test | If wrong |
|---|---|---|---|---|---|---|
| A77 | Building blocks: plain layers (optionally normalised) plus SotA residual normalised blocks (SimBa, BroNet), configurable so the tuner can evolve the shape. | confirmed | | `networks.Tower`, `NetSpec` | `train.eval_networks`; T19 trials (signpost: residual blocks never win or destabilise training) | add a block kind in `Tower` + `BLOCKS` |
| A91 | Every tower uses brax's lecun_uniform init and ends in a plain Dense head, whatever its block. | proposed | A77 | `networks.KERNEL_INIT`, `Tower` | T19: residual blocks with lecun_uniform vs orthogonal init | add an `init` field to `NetSpec` |
| A92 | Policy and value each get their own description, so the tuner shapes them separately. | proposed | A77 | `NetworkSpec` | T19: do the best policy and value shapes differ? | tie them in T19's search space |

The default description (`mlp`, swish, no norm, the profile's sizes) is brax's own MLP: same
parameter names and init, so a smoke run with seed 0 gives a byte-identical `policy.pkl` to T15's.
