import json

import numpy as np
import pytest

from tars_rl.train.config import PROFILES as TRAIN_PROFILES
from tars_rl.tune import campaign as camp
from tars_rl.tune.eval_campaign import main as eval_main
from tars_rl.tune.score import FAILED, WEIGHTS, blend, curve_speed
from tars_rl.tune.tars_space import SPACE, START, network_of, to_config


def test_start_is_the_published_go1_settings_and_t15_networks():
    full = TRAIN_PROFILES["full"]
    cfg = to_config(SPACE.decode(SPACE.encode(START)), full)
    for k in ("learning_rate", "entropy_cost", "discounting", "max_grad_norm", "reward_scaling"):
        assert getattr(cfg, k) == pytest.approx(getattr(full, k))
    assert cfg.unroll_length == full.unroll_length
    assert cfg.num_updates_per_batch == full.num_updates_per_batch
    # The tuner searches the towers only; memory is not part of its space.
    assert (cfg.network.policy, cfg.network.value) == (full.network.policy, full.network.value)


@pytest.mark.parametrize("profile", sorted(camp.PROFILES))
def test_every_point_of_the_space_gives_a_valid_config(profile):
    tp = camp.PROFILES[profile]
    base = TRAIN_PROFILES[tp.base]
    from dataclasses import replace

    base = replace(base, num_timesteps=tp.num_timesteps, num_evals=tp.num_evals)
    rng = np.random.default_rng(0)
    for u in [np.zeros(SPACE.dim), np.ones(SPACE.dim), *rng.uniform(size=(200, SPACE.dim))]:
        to_config(SPACE.decode(u), base)  # .check() raises on a bad one


def test_residual_block_without_norm_gets_layer_norm():
    s = dict(START, p_block="simba", p_norm="none")
    assert network_of(s).policy.norm == "layer"


def test_blend_weights_success_most_speed_least():
    assert WEIGHTS["success"] > WEIGHTS["closed"] > WEIGHTS["speed"]
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)
    flat = [{"step": 0, "success_rate": 0.5, "closed_frac": 0.5},
            {"step": 10, "success_rate": 0.5, "closed_frac": 0.5}]
    s = blend(1.0, 0.5, flat)
    assert s.speed == pytest.approx(0.5)
    assert s.total == pytest.approx(0.6 * 1.0 + 0.3 * 0.5 + 0.1 * 0.5)
    assert blend(0.0, float("nan"), flat).total == FAILED


def test_speed_rewards_learning_early():
    early = [{"step": 0, "success_rate": 0, "closed_frac": 0},
             {"step": 1, "success_rate": 1, "closed_frac": 1},
             {"step": 10, "success_rate": 1, "closed_frac": 1}]
    late = [dict(c, step=s) for c, s in zip(early, (0, 9, 10), strict=True)]
    assert curve_speed(early) > curve_speed(late)


class FakeRunner:
    """Scores settings without training: higher learning rate = better, up to a point."""

    def __init__(self, fail_at=None, stop_at=None):
        self.calls, self.fail_at, self.stop_at = [], fail_at, stop_at

    def __call__(self, cfg, out_dir, seed, episodes, score_seed, log):
        n = len(self.calls)
        self.calls.append(out_dir.name)
        if n == self.stop_at:
            raise KeyboardInterrupt  # the campaign is stopped mid-trial
        if n == self.fail_at:
            raise FloatingPointError("NaN loss")
        q = float(np.exp(-abs(np.log10(cfg.learning_rate) + 3.0)))
        curve = [{"step": 0, "success_rate": 0.0, "closed_frac": 0.0},
                 {"step": 100, "success_rate": q, "closed_frac": q}]
        return {"success_rate": q, "closed_frac": q}, curve


def test_campaign_writes_one_record_per_trial_and_marks_failures(tmp_path):
    r = FakeRunner(fail_at=2)
    s = camp.campaign("smoke", 7, 0, tmp_path, runner=r, log=lambda *_: None)
    recs = camp.read_records(tmp_path)
    assert len(r.calls) == 7 and len(s.scored()) == 7 and [x["id"] for x in recs] == list(range(7))
    for rec in recs:
        for key in ("settings", "network_str", "score", "parts", "curve", "round", "box"):
            assert key in rec or (rec["error"] and key == "curve")
    assert recs[2]["error"].startswith("FloatingPointError") and recs[2]["score"] == FAILED
    assert recs[0]["settings"] == pytest.approx(SPACE.decode(SPACE.encode(START)))
    assert [x["round"] for x in recs] == [0, 0, 0, 1, 1, 1, 2]


def test_stopped_campaign_resumes_without_retraining_finished_trials(tmp_path):
    first = FakeRunner(stop_at=4)  # stops during trial 4 (round 1, after trial 3 finished)
    with pytest.raises(KeyboardInterrupt):
        camp.campaign("smoke", 6, 0, tmp_path, runner=first, log=lambda *_: None)
    assert len(camp.read_records(tmp_path)) == 4
    second = FakeRunner()
    s = camp.campaign("smoke", 6, 0, tmp_path, runner=second, log=lambda *_: None)
    assert second.calls == ["0004", "0005"]
    assert len(s.scored()) == 6 and len(camp.read_records(tmp_path)) == 6
    with pytest.raises(ValueError):  # same folder, different campaign
        camp.campaign("smoke", 6, 1, tmp_path, runner=second, log=lambda *_: None)


def test_eval_prints_score_parts_and_writes_plot(tmp_path, capsys):
    camp.campaign("smoke", 6, 0, tmp_path, runner=FakeRunner(), log=lambda *_: None)
    assert eval_main(["--campaign", str(tmp_path), "--json", str(tmp_path / "e.json")]) == 0
    out = capsys.readouterr().out
    assert "=succ" in out and "+clos" in out and "+spd" in out and "Box path" in out
    assert (tmp_path / "campaign.png").stat().st_size > 0
    res = json.loads((tmp_path / "e.json").read_text())
    assert len(res["rows"]) == 6 and res["baseline"]["id"] == 0
