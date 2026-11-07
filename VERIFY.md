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

Setup once: `uv sync`; Unreal Engine 5.5 for Linux on the GPU box in `~/UnrealEngine`, then
`scripts/unreal.sh build` (2026-11-03: moved from the Mac to the box). On the box, step 2 is
`scripts/unreal.sh sequence` (window) or `scripts/unreal.sh sequence-fast` (headless, ~40 s)
instead of the Xcode route below.

Status 2026-11-04 (UE 5.5.4, box): builds clean (18 s). Fixes from the first day:
- **Floor:** the floor search hit the spectator pawn, so TARS dropped 1.2 m.
- **Joint drives:** Chaos's drives replaced by MuJoCo's servo, computed in C++ before every 2 ms
  physics step (one engine frame per step, no substeps; see `TarsActor.h`). Chaos's drives hit
  0.6 rad in 20 ms where MuJoCo's servo takes 0.5 s. Signs (A74) are both +1.
- **Contacts:** no bounce, the floor adds no friction of its own, sharp box edges (collision
  margin 0), and no inertia conditioning on the slabs (it made a lone slab tip at the wrong
  force, by up to 2.6x).
- **One joint per arm:** each arm hung on a hinge constraint, a carriage body, then a slide
  constraint. Chaos's iterative joint solver made that chain soft: with a 10 g carriage TARS slid
  at ~8 N where MuJoCo and Bullet hold ~20 N and every policy fell; a 100 g carriage got close but
  kept free fall, drag and airborne spin a few percent off. Now one constraint per arm holds both
  joints (twist = hinge, linear z = slide), as MuJoCo puts both in the arm's one body. Its order
  matters: central first, or Chaos measures the slide along the central slab instead of the arm.
- **Friction iterations:** Chaos applies friction in only the first 4 of its 8 position
  iterations. With that, the arms' feet slid back 2 cm while both hinges swung, where MuJoCo and
  Bullet hold them, and TARS fell over backwards within 2 s. `DefaultEngine.ini` now sets
  `p.Chaos.Solver.Collision.PositionFrictionIterations=8`. Shock propagation, static-friction
  stiffness and velocity friction iterations changed nothing.
- `scripts/unreal.sh fast` no longer reports ready off an old log.

Where it stands: Unreal now agrees with MuJoCo and Bullet on every probe
(`tars_rl.unreal.probe`, below), and one 20 ms step at a time it moves TARS like Bullet does
(`tars_rl.unreal.onestep`, below). The scripted sequence stands all 32 s. Its hinges track the
targets worse than MuJoCo's (corr ~0.5 vs 0.9), and Bullet's do too: MuJoCo's soft floor lets an
arm's foot press in and turn, the hard ones stall it. Tried without effect (before the carriage
fix): friction (also `p.CollisionFriction`), solver iterations up to 64/32, projection,
constraint priorities, stabilizers, a 10 m floor, corner-sphere contacts.

1. **Export the data Unreal reads** (only after a body or sequence change; tests fail if stale):
   `uv run python -m tars_rl.unreal.export` → `unreal/TarsRl/Content/TarsData/{body_spec,sequence}.json`.
2. **Build and play.** Right-click `unreal/TarsRl/TarsRl.uproject` → *Generate Xcode Project*,
   then double-click the `.uproject` and accept "rebuild modules". Press **Play**.
   - Check: TARS (3 grey slabs) stands still for 10 s, then swings and slides its arms
     through the sequence. The top-left line shows `t`, tilt and `standing` / `fell at`.
   - Nothing is replayed: `TarsActor` applies MuJoCo's servo forces before every 2 ms step.
   - At 32 s the screen shows `Recorded …/Saved/TarsRuns/unreal_poses.csv` (also on Stop).
3. **Compare to MuJoCo.**
   `uv run python -m tars_rl.unreal.eval --unreal unreal/TarsRl/Saved/TarsRuns/unreal_poses.csv --out runs/unreal`
   - Prints stand time and per-joint tracking (mean/max error, corr) for both engines, and the
     max/mean position (cm) and rotation (deg) gap per body.
   - Baseline: `--reference-only` gives MuJoCo alone (stands 32 s, hinge mean err ≈ 0.014 rad).
   - A joint with corr ≈ −1: flip `HingeSign` / `SlideSign` on the TarsActor (A74). Both are +1.
     The gain scales of A72 are gone: the servo is computed in SI and converted exactly.
   - Your call: are the gaps small enough that a MuJoCo-trained policy is worth trying in Unreal?
4. **Checks.** `uv run ruff check .` and `uv run pytest tests/unreal -q`.

## Engine probes and Bullet (`tars_rl.unreal.probe`, `tars_rl.unreal.bullet`)

Small experiments run identically in each engine, next to rigid-body mechanics' answer. They say
which part of an engine's TARS differs before any policy runs in it. Bullet (PyBullet) is a
third engine for sim-to-sim: the `sim2sim` extra, Linux only (it does not build on macOS 26).

0. **Tests:** `uv run pytest tests/unreal/test_probe.py -q` (Mac: the Bullet ones skip). Box:
   `uv run --extra sim2sim pytest tests/unreal/test_probe.py -q`.
1. **Probes:** `scripts/unreal.sh fast 1`, then
   `uv run --extra sim2sim python -m tars_rl.unreal.probe --engines mujoco bullet unreal`.
   2026-11-04, Unreal with one joint per arm and friction in all iterations (with the 10 g
   carriage and Chaos's default friction iterations, in brackets):

   | probe | MuJoCo | Bullet | Unreal | rigid |
   |---|---|---|---|---|
   | free fall, hinge at 0.04 / 0.10 s (rad) | 0.233 / 0.495 | 0.233 / 0.495 | 0.233 / 0.495 (- / 0.496) | |
   | slide at 0.5 s (mm) | 38.1 | 38.1 | 38.1 (38.1) | |
   | both hinges to 0.5: tilt at 0.18 s / max (deg) | 8.2 / 8.4 | 8.7 / 9.0 | 8.9 / 8.9 (- / 89.7) | |
   | the same: tilt at 2 s (deg) / moved (mm) | 0.2 / +41 | 0.0 / +33 | 0.0 / +36 (fell) | |
   | push 1 cm up: moves from (N) | 17.5 | 20.0 | 17.5 (7.5) | 23.5 |
   | push at centre: tips from (N) | 7.5 | 7.5 | 7.5 (5.5) | 7.4 |
   | 30 N at the feet for 0.3 s: slid (mm) / max tilt (deg) | 105 / 5.4 | 104 / 4.4 | 104 / 6.6 | 97 / |
   | sequence: fell at | never | never | never (10.8 s) | |

   - With a 100 g carriage and Chaos's default 4 friction iterations, the hinge swing still fell,
     slowly: past 30 deg at 1 s, after the old 0.5 s window. The probe now runs it 2 s.
   - Velocity-only friction (`PositionFrictionIterations 0`) scores policies higher but is wrong:
     TARS slides from 5 N and the drag tips it 59 deg.
   - Bullet yaws 14.5 deg in the hinge swing and Unreal 5.8 deg; MuJoCo stays at 0. Both are
     iterative solvers working through the left and right arm in turn.

1b. **One step at a time (`tars_rl.unreal.onestep`):** a policy walks in CPU MuJoCo; each engine
   is put in each of MuJoCo's states and steps once with the policy's action. Errors are grouped
   by the slabs on the floor. Whole episodes can't show this: they drift apart within a second.
   `uv run --extra sim2sim python -m tars_rl.unreal.onestep --policy runs/chain-dr/transformer
   --engines bullet unreal` (Unreal: `scripts/unreal.sh fast 1`). Tests:
   `tests/unreal/test_onestep.py`. 2026-11-04, 2000 steps, mean error after 20 ms:

   | slabs down (steps) | Bullet: v cm/s / spin deg/s / hinge deg/s | Unreal: the same |
   |---|---|---|
   | all (2000) | 4.1 / 18.9 / 15.9 | 4.2 / 22.5 / 16.9 |
   | both arms (634) | 1.8 / 12.1 / 8.0 | 1.7 / 15.6 / 9.2 |
   | central only (430) | 5.3 / 13.9 / 4.7 | 5.3 / 15.7 / 5.6 |
   | in the air (106) | 0.06 / 0.4 / 0.9 | 0.2 / 4.9 / 1.2 |
   | landing or lifting (785) | 5.9 / 29.1 / 30.0 | 6.1 / 32.8 / 30.9 |

   - Step by step, Unreal is now as close to MuJoCo as Bullet is. Both differ from MuJoCo most
     where contacts change (soft vs hard contact).
   - MuJoCo rests TARS ~1 mm into the floor; the tool lifts that out before handing a state to a
     hard-contact engine. Without the lift Unreal popped the slab out each step and looked twice
     as wrong as Bullet with the central slab down: an artifact of the tool, not of Unreal.
   - Yet whole episodes still score lower in Unreal than in Bullet (below). One teleport per step
     wipes what Chaos carries between steps: contact manifolds and static-friction anchors, which
     pull a slipped foot back to where it first stuck. Turning off friction restore
     (`p.Chaos.Collision.Manifold.EnableFrictionRestore 0`) lifted the chain-dr transformer from
     2.47 to 2.78 flags on the flat floor (32 episodes, before the one-joint change). That is
     the lead for the rest of the gap. Decided 2026-11-04: friction restore stays on (Unreal
     strict, not wrong; off, TARS drifts 0.5 mm in 2 s standing still).

2. **Policies in Bullet:** `uv run --extra sim2sim python -m tars_rl.unreal.verify --policy <run>
   --level <level> --engines mujoco bullet`. 2026-11-04, chain-dr transformer, 32 episodes
   (flags per episode / falls):

   | engine | flat | bumpy | steps | slopes | obstacles | apartment |
   |---|---|---|---|---|---|---|
   | MuJoCo | 4.47 / 0 % | 0.09 / 69 % | 2.09 / 34 % | 1.41 / 16 % | 4.44 / 0 % | 2.12 / 41 % |
   | Bullet | 3.88 / 0 % | 0.03 / 44 % | 1.59 / 22 % | 1.09 / 12 % | 3.97 / 0 % | 1.53 / 44 % |

   - Your call: is Bullet the stage-4 gate until Unreal's contact is fixed?
3. **Friction hardness (`train.run --contact-dr`):** elliptic friction cone, impratio
   log-uniform 1-10 per env.
   - Why: MuJoCo's default friction creeps 0.17 cm/s under a 10 N push that Bullet and Unreal
     hold still; impratio 10 creeps 0.02 cm/s.
   - Cost: about 2x the physics on CPU; 1.35x the training time on the 5080 (2.3 h for 200M).
   - Tests: `uv run pytest tests/env/test_randomize.py -q`.
   - A/B 2026-11-04, flat floor, transformer: `runs/contact/transformer` against
     `runs/chain-dr/transformer` (flags / falls %). MJX flat 4.25/1.2 (4.30/0), randomized bodies
     3.69/3.1 (3.80/3.9). Levels, 32 episodes, MuJoCo · Bullet · Unreal:
     flat 3.97/0 · 3.81/0 · 2.38/22 (4.47/0 · 3.88/0 · 2.78/6);
     steps 1.88/44 · 2.00/28 · 1.31/31 (2.09/34 · 1.59/22 · 1.66/28);
     obstacles 4.66/0 · 4.16/0 · 2.31/16 (4.44/0 · 3.97/0 · 3.03/0);
     apartment 1.47/66 · 1.19/62 · 1.75/28 (2.12/41 · 1.53/44 · 1.41/50).
   - So: MuJoCo to Bullet loses 4 % of the flags on the flat floor instead of 13 % (a smaller
     sim-to-sim gap), but this flat-floor transformer is no better in Bullet or Unreal.
   - Stage-3 A/B the same day, gru on terrain + randomized bodies, seed 0:
     `runs/stage3b/gru-contact` against `runs/stage3/gru` (2.0 h vs 1.2 h to train). MJX flat
     3.56/1.2 (3.76/0), randomized bodies 2.92/2.0 (3.05/2.0), rough ground 1.64/23 (1.59/29;
     rough third 0.48/37 vs 0.36/44). Levels, MuJoCo · Bullet · Unreal:
     flat 3.84/0 · 3.72/0 · 3.16/0 (3.66/0 · 3.19/3 · 2.22/25);
     steps 2.16/22 · 2.03/6 · 1.62/12 (2.22/16 · 1.72/6 · 1.28/28);
     slopes 1.75/19 · 1.88/9 · 1.56/19 (1.81/28 · 1.38/12 · 1.06/34);
     obstacles 3.44/3 · 3.88/0 · 3.34/3 (4.06/3 · 3.41/0 · 2.50/25);
     apartment 3.62/12 · 3.03/16 · 1.94/38 (3.50/19 · 2.59/19 · 1.25/62).
   - So: better in Bullet and Unreal on every level, and Unreal's falls on the flat floor go from
     25 % to 0.
   - Seed control, `runs/stage3b/gru-seed1` (no contact-dr, seed 1), Unreal falls %: flat 19,
     steps 19, slopes 28, obstacles 16, apartment 25 (seed 0: 25, 28, 34, 25, 62; contact-dr: 0,
     12, 19, 3, 38). Contact-dr falls less than either plain seed on four levels; the apartment
     varies too much by seed (25-62 %) to say. Bullet flags flat 3.28, obstacles 3.19, apartment
     2.53, below contact-dr's 3.72, 3.88, 3.03.
   - Decided 2026-11-04: contact-dr stays in the recipe (transfer first, training time no
     concern).

## Rough-ground training (`tars_rl.env.terrain`, roadmap stage 3)

Built 2026-11-03; first run 2026-11-04 (`runs/stage3/transformer`, transformer, terrain + body
randomization, 200M steps, 2.0 h on the 5080; MJX terrain costs 1.9x the flat floor). One 80 m heightfield, flat at one end and rough at the other
(tiles, thresholds, rugs, mounds); every training env starts on its own flat pad of it, turned its
own way. Train with `uv run python -m tars_rl.train.run --profile full --terrain --out <run>`.

0. **Tests:** `uv run pytest tests/env/test_terrain.py -q`. These cover: flat pads; rougher
   along x; MJX resting TARS on rough ground without sinking, checked against the grid itself; the
   nominal terrain resting TARS exactly like the plane; every env on its own patch.
1. **Look at it:** `uv run python -m tars_rl.env.terrain --png runs/terrain.png`.
   - Prints heights, steps (5 cm apart) and slopes (over 1 m) by difficulty; TARS stands on 64
     random pads (fell 0); and MJX's cost with terrain against the flat floor.
   - 2026-11-03 Mac: steps up to 7 cm and slopes about 10 deg at the rough end; cost 1.6x on CPU.
   - Check: the picture gets rougher left to right. Is the rough end what the apartment should
     ask of TARS?
2. **Baseline (flat-trained):** `uv run python -m tars_rl.train.eval --policy runs/chain-dr/default
   --terrain --domain-rand --episodes 256`.
   - Adds a table by difficulty. 2026-11-03, 48 episodes on the Mac: easy 3.14 flags/0 % falls,
     medium 0.56/31 %, rough 0.00/11 % (stuck, not falling).
3. **After terrain training:** the same eval on the new run. 2026-11-04, transformer (flags per
   episode / falls; flat-trained `runs/chain-dr/transformer` after the arrow):
   - rough ground: easy 2.73 / 4 % (2.84 / 5 %), medium 1.44 / 28 % (0.41 / 18 %), rough
     0.42 / 49 % (0.02 / 10 %, stuck rather than falling);
   - flat floor 3.37 / 0.8 % (4.30 / 0 %), randomized bodies 2.86 / 5.9 % (3.80 / 3.9 %);
   - levels, 32 episodes, MuJoCo | Bullet: apartment 2.97 / 22 % | 2.62 / 3 % (2.12 / 41 % |
     1.53 / 44 %), bumpy 0.31 / 59 % | 0.16 / 38 % (0.09 / 69 % | 0.03 / 44 %), steps 1.81 / 31 %
     | 1.53 / 16 % (2.09 / 34 % | 1.59 / 22 %), slopes 1.44 / 25 % | 1.53 / 16 % (1.41 / 16 % |
     1.09 / 12 %);
   - so: much more robust where the apartment is concerned, ~22 % slower on the flat floor.
   - gru, the same recipe (`runs/stage3/gru`, 200M steps in 1.2 h): rough ground easy 3.09 / 5 %,
     medium 1.58 / 33 %, rough 0.36 / 44 % (all bands 1.59 / 29 % vs the transformer's
     1.46 / 28 %); flat floor 3.76 / 0 %, randomized bodies 3.05 / 2.0 %; levels in CPU MuJoCo
     apartment 3.50 / 19 %, steps 2.22 / 16 %, slopes 1.81 / 28 %, bumpy 0.22 / 56 %. It matches
     or beats the transformer almost everywhere, in 60 % of the time. Your call: which network is
     the standard for stage 3?
   - Scoring fix the same day: `train.eval` without `--terrain` kept a terrain run's own terrain,
     so its "flat" score was a rough-ground one (`tests/train/test_eval_env.py`).
   - Your call: does the rough third improve without losing the flat floor?
   - Then the held-out test levels: `uv run python -m tars_rl.unreal.verify --policy <run>
     --level bumpy|steps|slopes --engines mjx mujoco`.
   - CPU MuJoCo is no reference on the heightfield itself: its small-sphere contact sank 15-20 mm
     into rough ground. Engines are compared on the box-built levels.

## Unreal verification of trained policies (`tars_rl.unreal.verify`, roadmap stage 4)

Setup once (box): Unreal 5.5.4 for Linux unzipped to `~/UnrealEngine` (~60 GB), then
`scripts/unreal.sh build` (18 s; the engine ships prebuilt).

2026-11-04 end of day, 32 episodes per level, flags per episode / falls %, MuJoCo · Bullet ·
Unreal (all Unreal fixes above; `runs/sim2sim/levels/`, older Unreal sweeps in its
`before-*` folders):

| policy | flat | steps | slopes | obstacles | apartment |
|---|---|---|---|---|---|
| chain-dr transformer | 4.47/0 · 3.88/0 · 2.78/6 | 2.09/34 · 1.59/22 · 1.66/28 | 1.41/16 · 1.09/12 · 1.00/19 | 4.44/0 · 3.97/0 · 3.03/0 | 2.12/41 · 1.53/44 · 1.41/50 |
| chain-dr gru | 4.12/0 · 3.38/3 · 2.41/41 | 2.03/38 · 1.66/25 · 1.50/34 | 1.50/38 · 1.19/38 · 0.94/28 | 4.75/0 · 3.69/3 · 2.38/31 | 1.66/50 · 1.44/34 · 1.19/72 |
| chain-dr default | 3.97/0 · 3.41/3 · 1.34/47 | 1.75/16 · 1.94/22 · 1.16/44 | 1.19/25 · 1.12/19 · 0.31/47 | 4.28/0 · 3.62/3 · 1.66/59 | 1.62/59 · 1.75/47 · 1.22/75 |
| stage3 transformer | 3.56/0 · 2.94/0 · 2.28/9 | 1.81/31 · 1.53/16 · 1.25/19 | 1.44/25 · 1.53/16 · 1.34/16 | 3.69/3 · 2.97/0 · 2.34/6 | 2.97/22 · 2.62/3 · 1.47/22 |
| stage3 gru | 3.66/0 · 3.19/3 · 2.22/25 | 2.22/16 · 1.72/6 · 1.28/28 | 1.81/28 · 1.38/12 · 1.06/34 | 4.06/3 · 3.41/0 · 2.50/25 | 3.50/19 · 2.59/19 · 1.25/62 |
| contact-dr transformer (flat) | 3.97/0 · 3.81/0 · 2.38/22 | 1.88/44 · 2.00/28 · 1.31/31 | 1.00/28 · 1.06/16 · 0.72/28 | 4.66/0 · 4.16/0 · 2.31/16 | 1.47/66 · 1.19/62 · 1.75/28 |
| stage3b gru-contact | 3.84/0 · 3.72/0 · 3.16/0 | 2.16/22 · 2.03/6 · 1.62/12 | 1.75/19 · 1.88/9 · 1.56/19 | 3.44/3 · 3.88/0 · 3.34/3 | 3.62/12 · 3.03/16 · 1.94/38 |
| stage3c transformer-s0 | 3.78/0 · 3.16/3 · 2.56/9 | 2.12/25 · 1.53/19 · 1.34/12 | 1.62/25 · 1.47/22 · 1.06/38 | 3.94/0 · 3.22/0 · 2.28/12 | 3.31/25 · 2.34/22 · 1.12/44 |
| stage3c transformer-s1 | 2.94/0 · 3.22/0 · 3.06/0 | 1.69/19 · 1.69/0 · 1.50/9 | 1.03/41 · 1.34/34 · 1.38/22 | 2.94/3 · 3.22/0 · 2.78/0 | 2.81/19 · 2.66/6 · 1.47/47 |
| stage3c gru-s1 | 3.81/0 · 3.09/0 · 2.34/19 | 2.38/22 · 1.59/6 · 0.84/28 | 1.41/41 · 1.44/16 · 1.12/38 | 4.22/0 · 3.50/0 · 2.50/19 | 3.34/9 · 2.56/12 · 1.66/44 |

(bumpy: every policy under 0.40 flags in every engine.)

2026-11-05, stage3c (`runs/stage3c/`): the full recipe (terrain + body DR + `--contact-dr`),
transformer seeds 0 and 1 (2.8 h each) and gru seed 1 (1.9 h) as the seed control for
stage3b gru-contact (seed 0). Same 32 episodes per level.
- Seeds differ as much as recipes: gru-contact seed 1 falls 19 % on Unreal's flat floor where
  seed 0 fell 0 %; transformer seed 1 walks Unreal's flat floor at 3.06/0, seed 0 at 2.56/9.
- With contact-dr, both transformer seeds fall less in Unreal than gru seed 1 on flat, steps
  and obstacles; slopes and the apartment fall 22-47 % for every policy.
- MuJoCo scores (256 episodes, `eval_*.json`), transformer-s0: flat 3.47/1 %, randomized bodies
  2.95/7 %, rough terrain 1.60/48 %.

- Bullet walks every policy at ~80-90 % of MuJoCo's flags, with few falls. Unreal walks the
  transformers at ~60-80 %, but the gru and default policies fall 25-75 % of the time there.
- Unreal now agrees with MuJoCo and Bullet on every probe, and one step at a time it is as close
  to MuJoCo as Bullet (Engine probes, 1b). What remains is state Chaos carries between steps
  (friction anchors; see 1b). Friction restore off: transformer flat 3.09/6, gru 2.66/28.
- The gru trained on terrain with friction hardness (`--contact-dr`, Engine probes 3) is the
  first policy that walks in Unreal at ~85 % of Bullet without falling on the flat floor.
- Decided 2026-11-04: trained in MuJoCo, evaluated in both Unreal and Bullet. The gate is that it
  works there: falls are the failure signal; flags per episode show the training transfers.
  No pass threshold is set in advance.
- Earlier today I put the gap on MuJoCo's soft contacts: a MuJoCo run with
  `cone="elliptic" impratio="10" noslip_iterations="10"` failed like Unreal. That was MuJoCo's
  20-iteration solver going erratic. With 100 iterations, hard MuJoCo contacts walk fine (3.7-4.2
  flags).

0. **Without Unreal** (any machine): `uv run pytest tests/env/test_levels.py tests/unreal -q`.
   A stand-in speaking the bridge protocol (MuJoCo behind it) must reproduce MuJoCo exactly;
   `uv run python -m tars_rl.env.levels --png runs/levels` prints and draws the six levels.
1. **The loop matches training.** `uv run python -m tars_rl.unreal.verify --policy <run>
   --engines mjx mujoco --episodes 64`: the mjx column (training physics, same flags) should
   match `tars_rl.train.eval` on the same run within noise. The mujoco (CPU) column is already a
   small sim-to-sim step: box-floor contacts differ from MJX (2026-11-03, the old GRU fell 72 %
   in MJX vs 50 % in CPU MuJoCo on the Euler body).
2. **Watch it.** On the box: `scripts/unreal.sh view`, then
   `uv run python -m tars_rl.unreal.verify --policy <run> --engines unreal --realtime --level flat`.
   - Check: TARS (3 slabs) walks to a red flag; the flag jumps when tagged; the top line shows
     episode, time, flags and distance; the camera follows.
   - Try `--level bumpy | steps | slopes | obstacles | apartment` (boxes as in the PNGs).
3. **The gate (fast).** `scripts/unreal.sh fast 4`, then
   `uv run python -m tars_rl.unreal.verify --policy <run> --level flat --episodes 64
   --ports 5577 5578 5579 5580 --json runs/unreal/<run>-flat.json`, `scripts/unreal.sh stop`.
   - Prints mjx, mujoco and unreal side by side with each gap to mjx; `steps_per_s` says how
     fast each ran.
   - Your call: is the flat-floor gap small enough to move on to the harder levels?
   - Big gap with a falling TARS: first run T10 (`scripts/unreal.sh sequence` + its eval);
     the body itself (servo, signs A74) may not match before the policy is to blame.

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

## T20 G3 comparison (`tars_rl.tune.compare`)

1. **Mac smoke comparison.** After T19's smoke campaign (`runs/tune_smoke`):
   `uv run python -m tars_rl.tune.compare --campaign runs/tune_smoke --profile smoke --seeds 2
   --points 3 --out runs/compare_smoke`, then `uv run python -m tars_rl.tune.eval_compare --compare
   runs/compare_smoke`. Only proves the pipeline: smoke runs cannot learn, so success is 0 % and both
   verdicts say NO.
2. **5080 comparison.** After the 5080 campaign (`runs/tune-full`), free the GPU, then from the Mac:
   `TARS_HOST=user@host scripts/train_remote.sh compare tune-full` (folder `runs/compare-full`, 5
   points x 3 fresh seeds, A112), `... status compare-full`, `... fetch compare-full`, then the eval
   with `--compare runs/compare-full`.
   - [ ] Section 1 / top-left of `compare.png`: mean held-out success climbs from the published start
     (red) through the in-between points to the final best (blue), and the start's and final best's
     seed dots don't overlap (both verdict lines YES).
   - [ ] Section 2: every run stopped `plateau`, not `budget` (A110); if not, it was still learning.
   - [ ] `compare.png` reads without the code: axes with units, legend naming the published start
     and the final best, learning curves in millions of steps, the box path with each retrained
     trial's round marked.
   - [ ] `final_best.mp4` shows the tuned TARS reaching the flag (the eval prints "reached flag YES"
     for the recorded episode).
3. **Resume.** Run the same `compare` command again: every finished (trial, seed) logs
   `already done, not retrained` (`test_compare_retrains_each_trial_once_per_fresh_seed_and_resumes`).
4. **Checks.** `uv run ruff check src/tars_rl/tune tests/tune` and `uv run pytest tests/tune`.
