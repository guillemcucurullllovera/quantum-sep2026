"""Mean-variance (Markowitz) portfolio optimization — the standard classical
baseline that a QAOA-based portfolio optimizer under applications/portfolio/
should be benchmarked against.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import LinearConstraint, minimize


def markowitz_min_variance(
    cov: np.ndarray, *, long_only: bool = True
) -> np.ndarray:
    """Minimize w^T Cov w subject to sum(w) == 1 (and w >= 0 if long_only).

    Returns the weight vector. This is the pure minimum-variance frontier
    point (no return target) — deliberately simple so it is easy to reason
    about as a baseline, not a full efficient-frontier solver.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    w0 = np.full(n, 1.0 / n)

    budget = LinearConstraint(np.ones((1, n)), 1.0, 1.0)
    bounds = [(0.0, 1.0) if long_only else (-1.0, 1.0) for _ in range(n)]

    result = minimize(
        lambda w: w @ cov @ w,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=[budget],
    )
    if not result.success:
        raise RuntimeError(f"Markowitz optimization failed: {result.message}")
    return result.x


def equal_weight_baseline(n_assets: int) -> np.ndarray:
    return np.full(n_assets, 1.0 / n_assets)


def portfolio_variance(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.asarray(w) @ np.asarray(cov) @ np.asarray(w))
