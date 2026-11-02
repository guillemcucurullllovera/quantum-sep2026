"""Geometric Brownian Motion path simulation with optional antithetic
variance reduction. Shared engine for option_pricing and risk baselines so
neither reimplements path generation.
"""
from __future__ import annotations

import numpy as np


def gbm_paths(
    *,
    s0: float,
    mu: float,
    sigma: float,
    t: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
    antithetic: bool = True,
) -> np.ndarray:
    """Return array of shape (n_paths, n_steps + 1) of simulated GBM paths.

    With antithetic=True, n_paths must be even; half the draws use -Z instead
    of Z, halving variance for the same sample count on symmetric payoffs.
    """
    rng = np.random.default_rng(seed)
    dt = t / n_steps

    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("antithetic=True requires an even n_paths")
        half = n_paths // 2
        z = rng.standard_normal((half, n_steps))
        z = np.concatenate([z, -z], axis=0)
    else:
        z = rng.standard_normal((n_paths, n_steps))

    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.concatenate([np.zeros((n_paths, 1)), log_paths], axis=1)
    return s0 * np.exp(log_paths)


def gbm_terminal_prices(
    *, s0: float, mu: float, sigma: float, t: float, n_paths: int, seed: int | None = None,
    antithetic: bool = True,
) -> np.ndarray:
    """Terminal-price-only shortcut (no need to store the full path)."""
    paths = gbm_paths(
        s0=s0, mu=mu, sigma=sigma, t=t, n_steps=1, n_paths=n_paths, seed=seed, antithetic=antithetic
    )
    return paths[:, -1]
