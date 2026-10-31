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
