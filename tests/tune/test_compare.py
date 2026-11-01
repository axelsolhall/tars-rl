import json

import pytest

from tars_rl.tune import campaign as camp
from tars_rl.tune import compare as cmp
from tars_rl.tune.eval_compare import main as eval_main
from tars_rl.tune.eval_compare import summarize


def _rec(i, score):
    return {"id": i, "score": score, "settings": {"i": i}, "network_str": "-"}


def test_points_are_start_evenly_spaced_best_so_far_and_final_best():
    recs = [_rec(i, s) for i, s in enumerate([0.1, 0.0, 0.3, 0.2, 0.5, 0.4, 0.6, 0.6, 0.1])]
    pts = cmp.pick_points(recs, 5)
    assert [p["after_trials"] for p in pts] == [1, 3, 5, 7, 9]
    assert [p["trial"] for p in pts] == [0, 2, 4, 6, 6]  # ties keep the earlier trial
    assert pts[0]["label"] == "published start" and pts[-1]["label"] == "final best"
    assert pts[-1]["campaign_score"] == max(r["score"] for r in recs)


def test_points_need_trial_0_and_two_points():
    with pytest.raises(ValueError, match="trial 0"):
        cmp.pick_points([_rec(1, 0.1)], 3)
    with pytest.raises(ValueError, match="at least 2"):
        cmp.pick_points([_rec(0, 0.1)], 1)


class FakeRetrain:
    """Held-out success grows with the campaign score; seeds add a little noise."""

    def __init__(self):
        self.calls = []

    def __call__(self, cfg, out_dir, seed, profile, score_seed, log):
        self.calls.append((out_dir.parent.name, seed, score_seed))
        out_dir.mkdir(parents=True, exist_ok=True)
        q = min(1.0, 1e3 * cfg.learning_rate) * 0.5 + 0.01 * seed
        curve = [{"kind": "curve", "step": 0, "success_rate": 0.0, "closed_frac": 0.0},
                 {"kind": "curve", "step": 100, "success_rate": q, "closed_frac": q}]
        return {"heldout": {"success_rate": q, "final_dist_m": 1 - q, "closed_frac": q},
                "curve": curve,
                "stop": {"reason": "plateau", "best_step": 100, "steps_trained": 200,
                         "best_quality": q, "patience": 2, "min_delta": 0.01}}


def fake_record(policy_dir, out, episodes, score_seed, log):
    out.mkdir(parents=True, exist_ok=True)
    meta = {"policy": str(policy_dir), "episode": 0, "of_episodes": episodes,
            "reached_flag": True, "fell": False, "steps": 10, "start_dist_m": 1.0,
            "final_dist_m": 0.1, "heldout_successes": 3}
    (out / cmp.VIDEO_META).write_text(json.dumps(meta))
    return meta


@pytest.fixture
def campaign_dir(tmp_path):
    def runner(cfg, out_dir, seed, episodes, score_seed, log):  # higher lr scores higher
        q = min(1.0, 1e3 * cfg.learning_rate)
        return {"success_rate": q, "closed_frac": q}, []

    out = tmp_path / "camp"
    camp.campaign("smoke", 7, 0, out, runner=runner, log=lambda *_: None)
    return out


def test_compare_retrains_each_trial_once_per_fresh_seed_and_resumes(campaign_dir, tmp_path):
    out = tmp_path / "cmp"
    r = FakeRetrain()
    res = cmp.compare(campaign_dir, "smoke", 2, 3, out, runner=r, recorder=fake_record,
                      log=lambda *_: None)
    trials = {p["trial"] for p in res["points"]}
    assert res["seeds"] == [1, 2]  # campaign trained with seed 0 (A87: fresh seeds)
    assert len(r.calls) == 2 * len(trials)
    assert {c[2] for c in r.calls} == {1}  # held-out flags: campaign score seed + 1
    for t in trials:
        for s in (1, 2):
            rec = cmp.read_result(out, t, s)
            assert rec["trial"] == t and rec["seed"] == s and "heldout" in rec
    again = FakeRetrain()
    cmp.compare(campaign_dir, "smoke", 2, 3, out, runner=again, recorder=fake_record,
                log=lambda *_: None)
    assert again.calls == []
    with pytest.raises(ValueError, match="different"):
        cmp.compare(campaign_dir, "smoke", 3, 3, out, runner=again, recorder=fake_record,
                    log=lambda *_: None)


def test_eval_compare_prints_verdict_and_writes_plot(campaign_dir, tmp_path, capsys):
    out = tmp_path / "cmp"
    cmp.compare(campaign_dir, "smoke", 2, 3, out, runner=FakeRetrain(), recorder=fake_record,
                log=lambda *_: None)
    assert eval_main(["--compare", str(out), "--no-video"]) == 0
    text = capsys.readouterr().out
    assert "published start" in text and "final best" in text
    assert "climbs from the published start" in text and "spreads stay apart" in text
    assert (out / "compare.png").stat().st_size > 0
    res = summarize(out)
    assert res["rows"][0]["trial"] == 0
    assert res["rows"][0]["success"]["min"] <= res["rows"][0]["success"]["max"]
    assert res["video"]["reached_flag"] is True
