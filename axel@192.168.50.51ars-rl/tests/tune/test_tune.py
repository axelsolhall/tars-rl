import numpy as np
import pytest

from tars_rl.tune import BoxSearch, Choice, Float, Int, SearchConfig, Space, random_search, run
from tars_rl.tune.eval_search import main as eval_main
from tars_rl.tune.space import Box
from tars_rl.tune.testfuncs import suite


def test_settings_roundtrip_through_unit_cube():
    space = Space((Float("lr", 1e-5, 1e-2, log=True), Float("clip", 0.1, 0.3),
                   Int("layers", 1, 4), Choice("act", ("relu", "tanh", "elu"))))
    s = {"lr": 3e-4, "clip": 0.2, "layers": 3, "act": "tanh"}
    back = space.decode(space.encode(s))
    assert back["lr"] == pytest.approx(3e-4)
    assert back["clip"] == pytest.approx(0.2)
    assert back["layers"] == 3 and back["act"] == "tanh"


def test_choice_bins_split_unit_range_equally():
    c = Choice("act", ("a", "b", "c"))
    assert [c.decode(u) for u in (0.0, 0.32, 0.34, 0.66, 0.67, 1.0)] == list("aabbcc")


def test_box_is_clipped_at_the_hard_limits():
    b = Box(np.array([0.9, 0.5]), 0.4)
    assert np.allclose(b.lows, [0.7, 0.3]) and np.allclose(b.highs, [1.0, 0.7])
    pts = b.sample(np.random.default_rng(0), 500)
    assert (pts >= b.lows).all() and (pts <= b.highs).all()


def _sphere_space():
    return Space(tuple(Float(f"x{i}", -1, 1) for i in range(2)))


def test_first_round_starts_at_the_given_centre_inside_the_start_box():
    s = BoxSearch(_sphere_space(), start={"x0": 0.5, "x1": -0.5})
    batch = s.ask()
    assert batch[0].settings == pytest.approx({"x0": 0.5, "x1": -0.5})
    u = np.array([t.u for t in batch])
    assert (np.abs(u - [0.75, 0.25]) <= 0.2 + 1e-9).all()


def test_box_recentres_on_best_and_obeys_goal():
    for goal, pick in (("min", np.argmin), ("max", np.argmax)):
        s = BoxSearch(_sphere_space(), goal=goal)
        batch = s.ask()
        scores = [t.settings["x0"] for t in batch]
        rec = s.tell(batch, scores)
        assert rec.centre == batch[int(pick(scores))].u


def test_box_shrinks_after_failed_rounds_then_restarts():
    cfg = SearchConfig(batch=4, init_batch=4, fail_rounds=2, min_length=0.1, init_length=0.4)
    s = run(BoxSearch(_sphere_space(), cfg), lambda st: 1.0, 4 + 4 * 8)  # never improves
    events = [b.event for b in s.boxes]
    lengths = [b.length for b in s.boxes]
    assert events[:7] == ["start", "keep", "keep", "shrink", "keep", "shrink", "keep"]
    assert lengths[3] == 0.2 and lengths[5] == 0.1
    assert "restart" in events and lengths[events.index("restart")] == 0.4


def test_box_grows_after_improving_rounds():
    cfg = SearchConfig(batch=2, init_batch=2, success_rounds=2)
    counter = iter(range(1000, 0, -1))  # every new trial is better than all before
    s = run(BoxSearch(_sphere_space(), cfg), lambda st: next(counter), 2 + 2 * 2)
    assert [b.event for b in s.boxes] == ["start", "keep", "keep", "grow"]
    assert s.boxes[-1].length == 0.8


def test_run_uses_exact_budget_and_save_load_resumes(tmp_path):
    f = lambda st: st["x0"] ** 2 + st["x1"] ** 2  # noqa: E731
    s = run(BoxSearch(_sphere_space(), seed=3), f, 30)
    assert len(s.trials) == 30 and all(t.score is not None for t in s.trials)
    s.save(tmp_path / "search.json")
    r = BoxSearch.load(tmp_path / "search.json", _sphere_space())
    assert [t.score for t in r.trials] == [t.score for t in s.trials]
    assert r.boxes == s.boxes
    run(r, f, 46)
    assert len(r.trials) == 46
    with pytest.raises(ValueError):
        BoxSearch.load(tmp_path / "search.json", Space((Float("y", 0, 1),)))


def test_ask_twice_without_tell_is_an_error():
    s = BoxSearch(_sphere_space())
    s.ask()
    with pytest.raises(RuntimeError):
        s.ask()


@pytest.mark.parametrize("fn", suite(), ids=lambda f: f.name)
def test_known_optimum_has_the_known_value(fn):
    for x in fn.x_opt:
        assert fn(x) == pytest.approx(fn.f_opt, abs=1e-4)
        assert fn.dist_to_opt(fn.space.encode(x)) < 1e-9


def test_random_search_covers_the_whole_space():
    u = random_search(_sphere_space(), 2000, 0)
    assert u.min() < 0.01 and u.max() > 0.99


def test_eval_runs_and_box_beats_random_on_a_small_budget(tmp_path, capsys):
    code = eval_main(["--budget", "60", "--repeats", "1", "--only", "branin", "levy4",
                      "--json", str(tmp_path / "e.json")])
    assert code == 0
    assert "box closer on 2/2" in capsys.readouterr().out
    assert (tmp_path / "e.json").exists()
