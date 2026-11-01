# VERIFY — what needs Axel's eyes

## T3 TARS body (`tars_rl.body`)

Setup once: `uv sync`

1. **Look and motion (Mac viewer).**
   - `uv run python -m tars_rl.body.view` holds the neutral pose; in the right panel, under Control,
     drag `left_hinge_act` / `left_slide_act` etc. to move each joint.
   - `uv run python -m tars_rl.body.view --demo` plays a scripted swing + slide.
   - Check: central body, two arms on either side, both swinging about one horizontal axis through
     the body's centre and sliding along their own length. Does it read as TARS?
2. **Numbers.** `uv run python -m tars_rl.body.eval --seconds 10` (add `--json out.json` to save).
   - Section 1: powered max tilt at neutral (should be ~0) vs unpowered; push rows show whether
     the actuators matter when disturbed.
   - Section 2: step tracking error, rise time, overshoot for all four joints.
   - Section 3: COM shift with the left arm slid fully up.
   - Section 4: MJX vs CPU max difference and MJX steps/s.
   - Exit code is non-zero if powered stand, tracking or MJX agreement fails.
3. **Regenerate the contract** after editing `default_spec`: `uv run python -m tars_rl.body.mjcf`
   (tests fail if `body/assets/tars_v1.{json,xml}` are stale).

## T13 Flag arena (`tars_rl.env`)

Setup once: `uv sync`. Needs T3's `tars_rl.body` on the branch.

1. **Baseline numbers.** `uv run python -m tars_rl.env.eval --episodes 64 --seed 0`
   (add `--json runs/env/baseline.json` to save). Both policies get the same 64 flags.
   - Expect a floor to beat: success ≈ 0 %, final distance ≈ start distance (closed ≈ 0),
     zero-action: 0 % falls and every episode a 20 s timeout; random: some falls, short or
     full-length episodes.
   - If zero-action falls, the body's neutral stand (T3) is the broken module, not the env.
2. **Flag in the scene.** The eval saves the random policy's episode 0 to
   `runs/env/random_ep0.npz` (Trajectory keys + `flag_pos`). Once T9 renders it, check TARS and
   the checkered flag appear together on the floor (the flag is the `flag` mocap body in
   `tars_rl.env.arena.build_arena_mjcf`; set `mocap_pos[0] = flag_pos`).
3. **Checks.** `uv run ruff check src/tars_rl/env tests/env` and `uv run pytest tests/env`.

## T24 Mark 2 senses (`tars_rl.env.sensors`)

1. **Senses table and tipped checks.** `uv run python -m tars_rl.env.eval --obs mark2 --episodes 16 --seed 0`
   - After the usual baseline table it lists the 11 mark2 channels (43 floats): name, units,
     nominal range and the range seen on a scripted run (stand, wiggle, lie on front, lie on back).
   - Lying on its front: the five `*_front` edges should light up (central bottom, both arms'
     bottom and top) and all three accelerometers should read about (−9.81, 0, 0); lying on its
     back, the `*_back` edges and (+9.81, 0, 0). Exit code 1 if any of these fails.
2. **Default unchanged.** `test_default_obs_is_benchmark_bit_for_bit` compares the default 17
   floats with T13's formula with `array_equal` over a random rollout, and checks that the default
   model has no added sites or sensors. Re-check against `runs/full-seed0` on the PC
   (`train.eval` should give the same numbers as before).

## T10 TARS in Unreal (`tars_rl.unreal`, `unreal/TarsRl`)

Setup once: `uv sync`; Unreal Engine 5.5 and Xcode on the Mac. Not built or run by the agent
(no Unreal on its side): the C++ compiles for the first time on your machine.

1. **Export the data Unreal reads** (only after a body or sequence change; tests fail if stale):
   `uv run python -m tars_rl.unreal.export` → `unreal/TarsRl/Content/TarsData/{body_spec,sequence}.json`.
2. **Build and play.** Right-click `unreal/TarsRl/TarsRl.uproject` → *Generate Xcode Project*,
   then double-click the `.uproject` and accept "rebuild modules". Press **Play**.
   - Check: TARS (3 grey slabs) stands still for 10 s, then swings and slides its arms
     through the sequence. The top-left line shows `t`, tilt and `standing` / `fell at`.
   - Nothing is replayed: `TarsActor` only sets joint drive targets each tick.
   - At 32 s the screen shows `Recorded …/Saved/TarsRuns/unreal_poses.csv` (also on Stop).
3. **Compare to MuJoCo.**
   `uv run python -m tars_rl.unreal.eval --unreal unreal/TarsRl/Saved/TarsRuns/unreal_poses.csv --out runs/unreal`
   - Prints stand time and per-joint tracking (mean/max error, corr) for both engines, and the
     max/mean position (cm) and rotation (deg) gap per body.
   - Baseline: `--reference-only` gives MuJoCo alone (stands 32 s, hinge mean err ≈ 0.014 rad).
   - A joint with corr ≈ −1: flip `HingeSign` / `SlideSign` on the TarsActor (A74).
     Sluggish or overshooting joints: tune `AngularGainScale` / `LinearGainScale` (A72).
   - Your call: are the gaps small enough that a MuJoCo-trained policy is worth trying in Unreal?
4. **Checks.** `uv run ruff check .` and `uv run pytest tests/unreal -q`.

## T17 Self-guiding search (`tars_rl.tune`)

Setup once: `uv sync`. Pure numpy; trials are test functions, no RL.

1. **Box vs random.** `uv run python -m tars_rl.tune.eval_search --budget 200 --repeats 5 --seed 0`
   (≈ 70 s; add `--json runs/tune/eval_search.json` to save, `--only branin levy4` for a subset).
   - Section 1: median gap to the known best after 25/50/100/200 tries, box next to random.
     Check: the box is closer on most functions (2026-10-31: 8/8). Exit code 1 if it wins on fewer than half.
2. **Box path.** Section 2 prints, per function, each round's tries, best gap, distance from the box
   centre to the optimum (unit coordinates) and side length.
   - Check: `dist` falls toward 0 (for ackley5/levy4/styblinski4/mixed5 the optimum starts outside
     the box), and `shrink` only appears once the gap stalls; `restart` early would mean it collapsed.
3. **Checks.** `uv run ruff check src/tars_rl/tune tests/tune` and `uv run pytest tests/tune`.

## T15 First training run (`tars_rl.train`)

1. **Mac smoke test.** `uv run python -m tars_rl.train.run --profile smoke --seed 0 --out runs/smoke`
   (~1 min), then `uv run python -m tars_rl.train.eval --policy runs/smoke --episodes 64 --seed 0`.
   Only proves the pipeline runs: 2,560 steps cannot learn, so "beats random and still: NO" is expected.
   Measured 2026-10-31: trained 0 % success, 81.2 % falls, closed −1.3 %; random 100 % falls;
   stand-still 0 % falls, return +20.0 (above the trained policy's +4.0).
2. **Full run on the 5080.** Free the GPU first, then from the Mac:
   `TARS_HOST=user@host scripts/train_remote.sh start 0`, `... status`, `... fetch`, then the eval
   with `--policy runs/full-seed0`. Sign-off: success rate and distance closed clearly above the
   random and stand-still columns.
3. **Learning curve.** The eval prints it (from `progress.jsonl`): success should rise then flatten.
   Watch for return rising while distance left does not shrink: standing still earns the
   0.02/step upright bonus, +20 over a full episode (A18's signpost).
4. **Recorded run.** `runs/<name>/rollout.npz` (Trajectory + `flag_pos`): render with T9 and check
   TARS moves toward the flag.
5. **Checks.** `uv run ruff check src/tars_rl/train tests/train` and `uv run pytest tests/train`.

## T9 — Rollout to MP4 (`tars_rl.viz`)

On the Mac:

    uv run python -m tars_rl.viz.eval --out /tmp/tars_sample.mp4
    open /tmp/tars_sample.mp4

On the Ubuntu box, headless over SSH (`MUJOCO_GL=egl` is set automatically when there is no `DISPLAY`):

    uv run python -m tars_rl.viz.eval --out /tmp/tars_sample.mp4

- [ ] TARS is visible for the whole clip (the camera tracks the `central` body).
- [ ] Both arm hinges swing in phase; both arm slides move a quarter period later.
  (No policy: the 0.5 rad swing tips TARS over at ~1.5 s and it lies flat after; that is expected.)
- [ ] Eval prints `frames N (expected N) OK` (4 s at 30 fps = 120 frames).
- [ ] `render speed` (s render per sim s) noted on the Mac: ____  and on the Ubuntu box: ____

## T16 — Flag-arena video (`tars_rl.viz`)

On the Mac (headless on the Ubuntu box works the same way, as in T9):

    uv run python -m tars_rl.viz.eval --flag-sample --out /tmp/tars_flag.mp4
    open /tmp/tars_flag.mp4

The eval runs one fixed random-policy episode in the flag arena (seed 0, `--seed` to change),
saves it as a flag rollout next to the video (`/tmp/tars_flag.npz`), and renders from that file.
Any saved flag rollout renders the same way, e.g. a trained run:
`uv run python -m tars_rl.viz.eval --rollout runs/<name>/rollout.npz --out /tmp/run.mp4`.

Measured 2026-11-01 on the Mac (seed 0): 83 frames × 0.02 s = 1.66 s (the random policy falls),
flag at x +1.159, y +0.561 m, distance 1.288 → 1.173 m (closer by 0.115 m), 50 frames OK,
render ≈ 0.3–0.9 s per sim s.

- [ ] TARS and the checkered flag are both in view for the whole clip (one fixed camera
  frames TARS's path and the flag, A93).
- [ ] The flag stands where the eval's `flag` line says (for seed 0: right of TARS and a
  little nearer the camera), and its distance from TARS looks like the printed one
  (≈ 3.5 body heights at the start; the body is 0.36 m tall).
- [ ] The `flag distance` line matches the video: start → end, TARS ends `closer` or `farther`.
- [ ] Eval prints `frames N (expected N) OK`.

## T18 Network builder (`tars_rl.train.networks`)

1. **Eval.** `uv run python -m tars_rl.train.eval_networks --profile smoke --seed 0` (~5 min on the
   Mac): trains the default and four other shapes (mlp+layernorm, SimBa, BroNet, SimBa-RMS-GELU)
   with the same smoke settings and prints parameter counts, training time, steps/s and held-out
   score next to random and stand-still. Smoke scores only prove each shape trains; ranking shapes
   is T19's job. Exits 1 if a policy folder recorded a different description than it trained.
2. **Default = T15.** Verified 2026-11-01: a smoke run (seed 0) before and after T18 gives a
   byte-identical `policy.pkl` and identical learning-curve numbers; `test_default_matches_brax_exactly`
   checks the param tree and init values against brax for both profiles. The eval's default row
   (0 % success, 81.2 % falls, closed −1.3 %) equals T15's smoke measurement above.
   Measured 2026-11-01 (~6 min): every shape trains; all smoke scores sit at the random/still floor,
   as expected from 2,560 steps.
3. **Description recorded.** Each policy folder's `meta.json` has `network` (fields) and
   `network_str` (short form, e.g. `policy=simba-2x32-relu-layer-e4 value=...`). Folders from T15
   (version 1) still load as the plain MLP they were.
4. **Your call:** are SimBa and BroNet (with the layer/RMS norm and activation options) the right
   SotA block set to start T19's search from, or should another block (e.g. SimbaV2's hyperspherical
   normalisation) be added first?
5. **Checks.** `uv run ruff check src/tars_rl/train tests/train` and `uv run pytest tests/train`.

## T19 Tuning campaign (`tars_rl.tune.campaign`)

1. **Mac smoke campaign.** `uv run python -m tars_rl.tune.campaign --profile smoke --trials 6 --seed 0
   --out runs/tune_smoke` (~8 min), then `uv run python -m tars_rl.tune.eval_campaign --campaign
   runs/tune_smoke`. Only proves the loop: 5,120 steps per trial cannot learn, so every score sits
   near 0 and "best above published: YES/NO" is noise. Measured 2026-11-01: 6 trials, ~80 s each,
   totals −0.004 … +0.004, 0 % success everywhere; `runs/tune_smoke/campaign.png` written.
2. **5080 campaign.** Free the GPU first, then from the Mac:
   `TARS_HOST=user@host scripts/train_remote.sh tune 48` (folder `runs/tune-full`),
   `... status tune-full`, `... fetch tune-full`, then the eval with `--campaign runs/tune-full`.
   Each trial is 50M steps (A107); check trial 0's `wall_s` in its record to size the campaign.
   - [ ] The best-score-per-trial plot (left panel of `campaign.png`) rises above the dashed
     published-settings line (trial 0 = Go1 settings, same 50M steps; T15's 200M run is longer).
3. **Resume.** Stop the campaign mid-trial (kill the process on the PC, or Ctrl-C the smoke run),
   then run the same command again.
   - [ ] The log says `trial N: already done ... not retrained` for every finished trial and only
     the unfinished trial and later ones train (`test_stopped_campaign_resumes_without_retraining_
     finished_trials` checks the same with a fake trainer).
4. **Score parts.** Eval section 1 prints raw success / closed / speed, then each weighted share
   (`=succ`, `+clos`, `+spd`) adding up to `total` (A106).
   - [ ] You can see from those columns why a trial ranked where it did.
5. **Checks.** `uv run ruff check src/tars_rl/tune tests/tune` and `uv run pytest tests/tune`.
