# VERIFY — what needs Axel's eyes

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
