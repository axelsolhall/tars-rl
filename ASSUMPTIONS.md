# Assumptions

IDs match loopdesk. "Confirmed" = decided by Axel; "proposed" = a default picked to keep moving.

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
