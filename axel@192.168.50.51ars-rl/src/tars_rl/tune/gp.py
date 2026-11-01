"""A small Gaussian process in numpy: Matern-5/2 kernel, hyperparameters picked by a grid on the
log marginal likelihood. Enough for a few hundred trials; no extra dependency.
"""

from __future__ import annotations

import numpy as np

LENGTHSCALES = np.geomspace(0.03, 3.0, 12)
NOISES = (1e-6, 1e-4, 1e-2, 1e-1)


def matern52(a: np.ndarray, b: np.ndarray, ls: float) -> np.ndarray:
    d = np.sqrt(np.maximum(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1), 0.0)) / ls
    s5 = np.sqrt(5.0) * d
    return (1.0 + s5 + 5.0 / 3.0 * d**2) * np.exp(-s5)


class GP:
    """Fit on (x in unit cube, y); y is standardised inside, predictions are in y's units."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.y_mean = float(y.mean())
        self.y_std = float(y.std()) or 1.0
        z = (y - self.y_mean) / self.y_std
        best = None
        for ls in LENGTHSCALES:
            k = matern52(self.x, self.x, ls)
            for noise in NOISES:
                try:
                    chol = np.linalg.cholesky(k + (noise + 1e-8) * np.eye(len(z)))
                except np.linalg.LinAlgError:
                    continue
                alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, z))
                lml = -0.5 * z @ alpha - np.log(np.diag(chol)).sum()
                if best is None or lml > best[0]:
                    best = (lml, ls, noise, chol, alpha)
        if best is None:
            raise np.linalg.LinAlgError("no GP hyperparameters gave a valid fit")
        _, self.lengthscale, self.noise, self._chol, self._alpha = best

    def posterior(self, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Joint posterior mean and covariance at xs (standardised units)."""
        ks = matern52(xs, self.x, self.lengthscale)
        mean = ks @ self._alpha
        v = np.linalg.solve(self._chol, ks.T)
        cov = matern52(xs, xs, self.lengthscale) - v.T @ v
        return mean, cov

    def thompson(self, xs: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        """n joint posterior draws at xs, shape (n, len(xs)), standardised units."""
        mean, cov = self.posterior(xs)
        chol = np.linalg.cholesky(cov + 1e-6 * np.eye(len(xs)))
        return mean[None, :] + rng.standard_normal((n, len(xs))) @ chol.T

    def predict(self, xs: np.ndarray) -> np.ndarray:
        """Posterior mean in y's units."""
        ks = matern52(xs, self.x, self.lengthscale)
        return ks @ self._alpha * self.y_std + self.y_mean
