"""G3 result eval: did the tuned settings beat the published start once retrained on fresh seeds?

Run: python -m tars_rl.tune.eval_compare --compare runs/compare_smoke

Reads a comparison folder (`tune.compare`) and its campaign and prints:
  1. per point (published start, in-between best-so-far points, final best): the trial, how far
     into the campaign it was the best, and held-out success and flag distance left as mean,
     standard deviation and min..max over seeds
  2. per retrained run: seed, stop reason, steps trained, the step kept, held-out results
  3. the verdict (A87's signpost): does mean held-out success climb from the start through the
     points to the final best, and do the start's and final best's seed spreads (min..max) stay
     apart? Printed as YES/NO; the eval exits 0 either way (the smoke comparison cannot learn)
and saves in the comparison folder:
  compare.png       held-out success and distance left against search progress with each seed
                    as a dot, every point's validation learning curve, and the search box's path
  final_best.mp4    the recorded held-out episode of the final best (video/rollout.npz)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from tars_rl.tune.campaign import SEARCH_FILE, read_records
from tars_rl.tune.compare import COMPARE_FILE, VIDEO_DIR, VIDEO_META, VIDEO_ROLLOUT, read_result
from tars_rl.tune.tars_space import PPO_KEYS, SPACE


def _stats(xs: list[float]) -> dict:
    return {"mean": statistics.fmean(xs), "sd": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs), "max": max(xs), "values": xs}


def summarize(compare: str | Path) -> dict:
    compare = Path(compare)
    m = json.loads((compare / COMPARE_FILE).read_text())
    campaign = Path(m["campaign"])
    rows = []
    for p in m["points"]:
        runs = [r for s in m["seeds"] if (r := read_result(compare, p["trial"], s)) is not None]
        if not runs:
            raise SystemExit(f"{compare}: point {p['index']} (trial {p['trial']}) has no "
                             f"finished run yet")
        rows.append(p | {
            "runs": runs,
            "success": _stats([r["heldout"]["success_rate"] for r in runs]),
            "dist": _stats([r["heldout"]["final_dist_m"] for r in runs]),
        })
    means = [r["success"]["mean"] for r in rows]
    start, final = rows[0], rows[-1]
    search = json.loads((campaign / SEARCH_FILE).read_text())
    vpath = compare / VIDEO_DIR / VIDEO_META
    return {
        "compare": str(compare), "manifest": m, "rows": rows, "boxes": search["boxes"],
        "trial_rounds": {r["id"]: r["round"] for r in read_records(campaign)},
        "climbs": all(b >= a for a, b in zip(means, means[1:], strict=False))
                  and means[-1] > means[0],
        "apart": final["success"]["min"] > start["success"]["max"],
        "same_trial": start["trial"] == final["trial"],
        "video": json.loads(vpath.read_text()) if vpath.exists() else None,
    }


def report(res: dict) -> None:
    m = res["manifest"]
    cp = m["compare_profile"]
    print(f"Comparison {res['compare']} of campaign {m['campaign']}: profile {m['profile']}, "
          f"seeds {m['seeds']}, {cp['heldout_episodes']} held-out flags (score seed "
          f"{m['score_seed']}), up to {cp['max_timesteps']:,} steps, stop after "
          f"{cp['patience']} evals without +{cp['min_delta']} validation quality")

    print("\n1. Points (held-out, over seeds: mean +- sd [min..max])")
    print(f"{'pt':>3}{'after':>7}{'trial':>6}  {'success':<24}{'dist left (m)':<26}label")
    for r in res["rows"]:
        s, d = r["success"], r["dist"]
        print(f"{r['index']:>3}{r['after_trials']:>7}{r['trial']:>6}  "
              f"{s['mean']:>6.1%} +-{s['sd']:>5.1%} [{s['min']:.0%}..{s['max']:.0%}]".ljust(42)
              + f"{d['mean']:>5.2f} +-{d['sd']:>4.2f} [{d['min']:.2f}..{d['max']:.2f}]".ljust(26)
              + r["label"])

    print("\n2. Retrained runs")
    print(f"{'trial':>5}{'seed':>5}  {'stop':<8}{'trained':>13}{'kept':>13}{'val q':>7}"
          f"{'success':>9}{'dist':>7}{'wall s':>8}")
    seen = set()
    for r in res["rows"]:
        if r["trial"] in seen:
            continue
        seen.add(r["trial"])
        for x in r["runs"]:
            st, h = x["stop"], x["heldout"]
            print(f"{x['trial']:>5}{x['seed']:>5}  {st['reason']:<8}{st['steps_trained']:>13,}"
                  f"{st['best_step']:>13,}{st['best_quality']:>+7.3f}{h['success_rate']:>9.1%}"
                  f"{h['final_dist_m']:>7.2f}{x['wall_s']:>8.0f}")
    if any(x["stop"]["reason"] == "budget" for r in res["rows"] for x in r["runs"]):
        print("  stop 'budget': still improving at max_timesteps, so that score may be low")

    v = res["video"]
    if v:
        print(f"\nVideo episode: {v['episode']} of {v['of_episodes']} held-out flags "
              f"({v['heldout_successes']} reached), reached flag "
              f"{'YES' if v['reached_flag'] else 'NO'}, {v['start_dist_m']:.2f} m -> "
              f"{v['final_dist_m']:.2f} m in {v['steps']} steps")

    if res["same_trial"]:
        print("\nThe campaign never beat trial 0: the final best is the published start.")
    print(f"\nHeld-out success climbs from the published start to the final best: "
          f"{'YES' if res['climbs'] else 'NO'}")
    print(f"Start's and final best's seed spreads stay apart: {'YES' if res['apart'] else 'NO'}")


MIDDLE = ("C2", "C4", "C1", "C5", "C6", "C8", "C9")


def _colour(i: int, n: int) -> str:
    """Published start red, final best blue, the points in between their own colours."""
    return "C3" if i == 0 else "C0" if i == n - 1 else MIDDLE[(i - 1) % len(MIDDLE)]


def plot(res: dict, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    rows, n = res["rows"], len(res["rows"])
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    xs = [r["after_trials"] for r in rows]
    ns = len(res["manifest"]["seeds"])
    for ax, key, ylab, scale in ((axs[0, 0], "success", "held-out success (% of flags)", 100),
                                 (axs[0, 1], "dist", "held-out distance left to flag (m)", 1)):
        ax.plot(xs, [r[key]["mean"] * scale for r in rows], c="k", lw=1, zorder=1,
                label="mean over seeds")
        for i, r in enumerate(rows):
            vals = [v * scale for v in r[key]["values"]]
            ax.errorbar(r["after_trials"], r[key]["mean"] * scale, yerr=r[key]["sd"] * scale,
                        fmt="o", c=_colour(i, n), capsize=4, zorder=2,
                        label=f"{r['label']} (trial {r['trial']})")
            ax.scatter([r["after_trials"]] * len(vals), vals, s=12, c=_colour(i, n), alpha=0.5,
                       marker="x", zorder=3)
        ax.scatter([], [], c="grey", marker="x", label="one seed")
        ax.set(xlabel="search progress (campaign trials so far)", ylabel=ylab,
               title=f"{ylab.split(' (')[0].capitalize()}, retrained on {ns} fresh "
                     f"seed{'s' if ns > 1 else ''} (mean +- sd)")
        ax.set_xticks(xs)
        ax.legend(fontsize=7)
    axs[0, 0].set_ylim(-5, 105)

    a = axs[1, 0]
    for i, r in enumerate(rows):
        for j, x in enumerate(r["runs"]):
            c = [p for p in x["curve"] if "step" in p]
            a.plot([p["step"] / 1e6 for p in c], [p["success_rate"] * 100 for p in c],
                   c=_colour(i, n), alpha=0.8, label=r["label"] if j == 0 else None)
            a.axvline(x["stop"]["best_step"] / 1e6, c=_colour(i, n), ls=":", lw=0.8)
    a.set(xlabel="environment steps (millions)", ylabel="validation success (% of flags)",
          title="Learning curves of the retrained points (dotted: step kept)", ylim=(-5, 105))
    a.legend(fontsize=7)

    a = axs[1, 1]
    boxes = res["boxes"]
    rounds = [b["round"] for b in boxes]
    for j, k in enumerate(PPO_KEYS):  # own colours, so they don't read as the points' colours
        i = SPACE.names.index(k)
        a.plot(rounds, [b["centre"][i] for b in boxes], marker=".", label=k,
               c=plt.cm.viridis(j / (len(PPO_KEYS) - 1)))
    for i, r in enumerate(rows):
        rd = res["trial_rounds"].get(r["trial"])
        if rd is not None:
            a.axvline(rd, c=_colour(i, n), ls="--", lw=1)
            a.annotate(f"trial {r['trial']}", (rd, 1.0), fontsize=7, rotation=90, va="top")
    a.set(xlabel="search round", ylim=(0, 1),
          ylabel="box centre (share of each setting's range, 0..1)",
          title="Search box path (dashed: round each retrained trial came from)")
    a.xaxis.set_major_locator(MaxNLocator(integer=True))
    a.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def render_video(compare: Path) -> tuple[Path, object] | None:
    rp = compare / VIDEO_DIR / VIDEO_ROLLOUT
    if not rp.exists():
        return None
    from tars_rl.env.rollout import FlagRollout
    from tars_rl.viz.flag import arena_model, render_flag_mp4

    model, _ = arena_model()
    out = compare / "final_best.mp4"
    return out, render_flag_mp4(model, FlagRollout.load(rp), out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--compare", required=True, help="folder written by tars_rl.tune.compare")
    p.add_argument("--no-video", action="store_true", help="skip rendering final_best.mp4")
    p.add_argument("--json", help="also write the summary to this JSON file")
    a = p.parse_args(argv)
    res = summarize(a.compare)
    report(res)
    print(f"\nplot: {plot(res, Path(a.compare) / 'compare.png')}")
    if not a.no_video:
        v = render_video(Path(a.compare))
        if v is None:
            print("video: none recorded")
        else:
            out, st = v
            print(f"video: {out} ({st.frames} frames at {st.fps} fps, {st.video_seconds:.1f} s; "
                  f"rendered in {st.render_seconds:.1f} s)")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
