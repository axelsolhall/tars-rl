"""Self-guiding search (A75): Bayesian optimisation inside a box that re-centres on the best
point found and grows or shrinks with how well the last rounds went.

Loop, one round at a time:
  ask()   proposes a batch of settings: round 0 is the box centre plus random points in the
          starting box; later rounds fit a GP to every trial and take Thompson-sampling picks
          from candidates inside the current box.
  tell()  takes their scores, then moves the box centre onto the best point so far and
          doubles the box after `success_rounds` improving rounds in a row, halves it after
          `fail_rounds` non-improving ones. Below `min_length` it restarts at the starting
          length around the best point (the trials are kept).
Every trial and every box is kept in `trials` / `boxes` and saved as JSON, so a campaign can be
stopped, inspected and resumed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from tars_rl.tune.gp import GP
from tars_rl.tune.space import Box, Space

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SearchConfig:
    """Box rules (A90) and batch sizes. Lengths are fractions of each setting's full range."""

    batch: int = 8
    init_batch: int = 16
    init_length: float = 0.4
    min_length: float = 0.4 / 2**5
    max_length: float = 1.0
    success_rounds: int = 2
    fail_rounds: int = 3
    improve_tol: float = 1e-3  # relative improvement that counts as a success
    candidates: int = 1024


@dataclass
class Trial:
    id: int
    round: int
    u: list[float]
    settings: dict[str, Any]
    score: float | None = None


@dataclass
class BoxRecord:
    """The box used for `round`, and the best score known when it was set."""

    round: int
    centre: list[float]
    length: float
    best: float | None
    event: str  # start | grow | shrink | keep | restart


@dataclass
class BoxSearch:
    space: Space
    config: SearchConfig = field(default_factory=SearchConfig)
    start: dict[str, Any] | None = None  # centre of the starting box; None = middle of the space
    goal: str = "min"  # "min" or "max" of the score
    seed: int = 0
    trials: list[Trial] = field(default_factory=list)
    boxes: list[BoxRecord] = field(default_factory=list)
    _succ: int = 0
    _fail: int = 0

    def __post_init__(self):
        if self.goal not in ("min", "max"):
            raise ValueError(f"goal must be 'min' or 'max', got {self.goal!r}")
        self._rng = np.random.default_rng(self.seed)
        if not self.boxes:
            centre = (
                self.space.encode(self.start) if self.start is not None
                else np.full(self.space.dim, 0.5)
            )
            self.boxes.append(BoxRecord(0, centre.tolist(), self.config.init_length, None, "start"))

    # ---- state -------------------------------------------------------------------------------

    @property
    def round(self) -> int:
        return self.boxes[-1].round

    @property
    def box(self) -> Box:
        b = self.boxes[-1]
        return Box(np.array(b.centre), b.length)

    def scored(self) -> list[Trial]:
        return [t for t in self.trials if t.score is not None]

    def best(self) -> Trial | None:
        done = self.scored()
        if not done:
            return None
        return min(done, key=lambda t: self._loss(t.score))

    def _loss(self, score: float) -> float:
        return score if self.goal == "min" else -score

    # ---- ask / tell --------------------------------------------------------------------------

    def ask(self) -> list[Trial]:
        """The next round's batch. Every trial asked must be told before the next ask."""
        if any(t.score is None for t in self.trials):
            raise RuntimeError("tell() the pending trials before asking again")
        box = self.box
        if self.round == 0:
            u = box.sample(self._rng, self.config.init_batch)
            u[0] = box.centre
        else:
            u = self._propose(box)
        new = [
            Trial(len(self.trials) + i, self.round, x.tolist(), self.space.decode(x))
            for i, x in enumerate(u)
        ]
        self.trials.extend(new)
        return new

    def tell(self, trials: list[Trial], scores: list[float]) -> BoxRecord:
        """Record scores for the asked batch, then move and resize the box. Returns the new box."""
        prev = self.best()
        for t, s in zip(trials, scores, strict=True):
            self.trials[t.id].score = float(s)
        if any(t.score is None for t in self.trials):
            raise RuntimeError("some asked trials still have no score")
        best = self.best()
        cfg, length = self.config, self.boxes[-1].length
        event = "keep"
        if prev is not None:  # round 0 only sets the first centre
            p, b = self._loss(prev.score), self._loss(best.score)
            if b < p - cfg.improve_tol * max(abs(p), 1e-12):
                self._succ, self._fail = self._succ + 1, 0
            else:
                self._succ, self._fail = 0, self._fail + 1
            if self._succ >= cfg.success_rounds:
                length, event, self._succ = min(2 * length, cfg.max_length), "grow", 0
            elif self._fail >= cfg.fail_rounds:
                length, event, self._fail = length / 2, "shrink", 0
                if length < cfg.min_length:
                    length, event = cfg.init_length, "restart"
        rec = BoxRecord(self.round + 1, list(best.u), length, best.score, event)
        self.boxes.append(rec)
        return rec

    def _propose(self, box: Box) -> np.ndarray:
        cfg, d = self.config, self.space.dim
        done = self.scored()
        x = np.array([t.u for t in done])
        y = _normal_scores(np.array([self._loss(t.score) for t in done]))
        gp = GP(x, y)
        # Candidates: half uniform in the box, half the centre with a few settings resampled
        # (the TuRBO trick; keeps candidates near the best point in higher dimensions).
        n_uni = cfg.candidates // 2
        cand = box.sample(self._rng, cfg.candidates)
        mask = self._rng.random((cfg.candidates - n_uni, d)) < min(1.0, 20.0 / d)
        mask[np.arange(len(mask)), self._rng.integers(0, d, len(mask))] = True
        cand[n_uni:] = np.where(mask, cand[n_uni:], box.centre)
        draws = gp.thompson(cand, cfg.batch, self._rng)
        picked: list[int] = []
        for draw in draws:
            draw[picked] = np.inf
            picked.append(int(np.argmin(draw)))
        return cand[picked]

    # ---- persistence -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "names": self.space.names,
            "config": asdict(self.config),
            "goal": self.goal,
            "seed": self.seed,
            "start": self.start,
            "counters": [self._succ, self._fail],
            "trials": [asdict(t) for t in self.trials],
            "boxes": [asdict(b) for b in self.boxes],
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: str | Path, space: Space) -> BoxSearch:
        d = json.loads(Path(path).read_text())
        if d["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"search file has schema {d['schema_version']}, want {SCHEMA_VERSION}")
        if d["names"] != space.names:
            raise ValueError(f"search file settings {d['names']} != space {space.names}")
        s = cls(
            space, SearchConfig(**d["config"]), d["start"], d["goal"], d["seed"],
            [Trial(**t) for t in d["trials"]], [BoxRecord(**b) for b in d["boxes"]],
        )
        s._succ, s._fail = d["counters"]
        # The random stream restarts from a seed tied to the progress, so a resumed run is
        # reproducible but not identical to an uninterrupted one.
        s._rng = np.random.default_rng([s.seed, len(s.trials)])
        return s


def _normal_scores(loss: np.ndarray) -> np.ndarray:
    """Ranks mapped to standard-normal quantiles (A88): the GP sees order, not scale, so a few huge
    losses (diverged runs, steep test functions) do not flatten the rest."""
    ranks = np.argsort(np.argsort(loss))
    nd = NormalDist()
    return np.array([nd.inv_cdf((r + 0.5) / len(loss)) for r in ranks])


def random_search(space: Space, n: int, seed: int) -> np.ndarray:
    """The baseline: n points uniform over the whole space (unit coordinates)."""
    return np.random.default_rng(seed).uniform(0.0, 1.0, size=(n, space.dim))


def run(search: BoxSearch, objective, budget: int) -> BoxSearch:
    """Ask/tell until `budget` trials are scored; the last batch is cut to fit."""
    while len(search.scored()) < budget:
        batch = search.ask()
        extra = len(search.trials) - budget
        if extra > 0:
            del search.trials[-extra:]
            batch = batch[:-extra]
        search.tell(batch, [objective(t.settings) for t in batch])
    return search


__all__ = ["BoxRecord", "BoxSearch", "SearchConfig", "Trial", "random_search", "run"]
