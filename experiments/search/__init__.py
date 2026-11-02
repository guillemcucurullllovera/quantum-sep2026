"""Bayesian-optimization-based parameter search, genuinely distinct from
experiments.parameter_search's grid/random sampling: this module builds a
real (from-scratch, no sklearn) Gaussian Process surrogate over a continuous
parameter space and uses Expected Improvement to pick the next point to
evaluate, which needs far fewer objective-function evaluations than grid
search for expensive-to-evaluate objectives (e.g. a QAOA solve at
non-trivial qubit count) -- that's the real, checkable value-add over
grid()/random_sample().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.stats import norm


def _rbf_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float, variance: float) -> np.ndarray:
    d2 = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
    return variance * np.exp(-0.5 * np.maximum(d2, 0) / length_scale**2)


@dataclass
class GaussianProcess:
    """Closed-form GP posterior, RBF kernel, fixed (not marginal-likelihood-
    optimized) hyperparameters -- deliberately simple: this is a real,
    correct GP for a small 1-3 dimensional search space, not a production
    GP library replacement."""

    length_scale: float = 1.0
    variance: float = 1.0
    noise: float = 1e-6
    X_train: np.ndarray = field(default=None)
    y_train: np.ndarray = field(default=None)
    _K_inv: np.ndarray = field(default=None, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        self.X_train, self.y_train = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        K = _rbf_kernel(self.X_train, self.X_train, self.length_scale, self.variance)
        K += self.noise * np.eye(len(self.X_train))
        self._K_inv = np.linalg.inv(K)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        K_s = _rbf_kernel(X, self.X_train, self.length_scale, self.variance)
        mean = K_s @ self._K_inv @ self.y_train
        K_ss_diag = self.variance * np.ones(len(X))
        var = K_ss_diag - np.sum((K_s @ self._K_inv) * K_s, axis=1)
        return mean, np.maximum(var, 1e-12)


def expected_improvement(mean: np.ndarray, var: np.ndarray, best_so_far: float, minimize: bool = True) -> np.ndarray:
    std = np.sqrt(var)
    if minimize:
        improvement = best_so_far - mean
    else:
        improvement = mean - best_so_far
    z = improvement / std
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    return np.where(std > 1e-9, ei, 0.0)


@dataclass
class BayesianSearchResult:
    best_params: np.ndarray
    best_value: float
    all_params: list[np.ndarray]
    all_values: list[float]


def bayesian_search(
    objective: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    *,
    n_initial: int = 5,
    n_iterations: int = 15,
    minimize: bool = True,
    seed: int = 0,
    candidate_pool_size: int = 500,
) -> BayesianSearchResult:
    """Sample n_initial points randomly, fit a GP, then repeatedly propose
    the point (from a random candidate pool) with highest Expected
    Improvement, evaluate it, and refit -- the standard Bayesian-
    optimization loop, implemented from scratch.
    """
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])

    X = rng.uniform(lows, highs, size=(n_initial, dim))
    y = np.array([objective(x) for x in X])

    for _ in range(n_iterations):
        gp = GaussianProcess().fit(X, y)
        candidates = rng.uniform(lows, highs, size=(candidate_pool_size, dim))
        mean, var = gp.predict(candidates)
        best_so_far = y.min() if minimize else y.max()
        ei = expected_improvement(mean, var, best_so_far, minimize=minimize)
        next_x = candidates[np.argmax(ei)]
        next_y = objective(next_x)
        X = np.vstack([X, next_x])
        y = np.append(y, next_y)

    best_idx = y.argmin() if minimize else y.argmax()
    return BayesianSearchResult(
        best_params=X[best_idx], best_value=float(y[best_idx]), all_params=list(X), all_values=list(y)
    )
