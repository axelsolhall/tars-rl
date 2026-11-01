"""Unreal vs MuJoCo eval (T10). Run: python -m tars_rl.unreal.eval [--unreal poses.csv]

Plays the scripted sequence in MuJoCo and prints its stand time and per-joint tracking.
With --unreal, also reads the poses Unreal recorded (Saved/TarsRuns/unreal_poses.csv) and
prints the same numbers for Unreal plus the per-body position / rotation gap between the two.
--reference-only skips Unreal. Exits non-zero if MuJoCo itself falls (then the body or the
sequence is broken, not Unreal).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tars_rl.body.mjcf import load_spec
from tars_rl.unreal.compare import format_gap, format_run, gap, run_metrics
from tars_rl.unreal.poses import from_unreal_csv, mujoco_reference, to_unreal_csv
from tars_rl.unreal.sequence import default_sequence


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--unreal", type=Path, help="CSV the Unreal TarsActor recorded")
    ap.add_argument("--reference-only", action="store_true", help="MuJoCo numbers only")
    ap.add_argument("--out", type=Path, help="directory for reference poses and metrics JSON")
    args = ap.parse_args(argv)

    spec = load_spec()
    seq = default_sequence(spec)
    ref, _ = mujoco_reference(spec, seq)
    report = {"sequence_s": seq.duration, "mujoco": run_metrics(spec, seq, ref)}
    print(f"sequence: {seq.duration:.1f} s, {len(seq.times)} keyframes, "
          f"{len(ref.t)} samples per engine")
    print(format_run("MuJoCo", report["mujoco"]))

    if args.unreal and not args.reference_only:
        ue = from_unreal_csv(args.unreal)
        report["unreal"] = run_metrics(spec, seq, ue)
        report["gap"] = gap(ref, ue)
        print(format_run("Unreal", report["unreal"]))
        print(format_gap(report["gap"]))
        if ue.t[-1] < seq.duration - 0.1:
            print(f"warning: Unreal recording ends at {ue.t[-1]:.1f} s of {seq.duration:.1f} s")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        ref.save(args.out / "mujoco_poses.npz")
        to_unreal_csv(ref, args.out / "mujoco_poses_ue_format.csv")
        (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}/mujoco_poses.npz, mujoco_poses_ue_format.csv, metrics.json")

    return 1 if report["mujoco"]["fell"] else 0


if __name__ == "__main__":
    sys.exit(main())
