"""Synthetic correlated-returns market generator, used for self-contained
testing without any real market-data dependency (and therefore no data
licensing/secrets concerns in this repository)."""
from __future__ import annotations

import numpy as np


def generate_synthetic_market(
    n_assets: int,
    *,
    seed: int = 0,
    mean_return_range: tuple[float, float] = (-0.02, 0.05),
    volatility_range: tuple[float, float] = (0.05, 0.30),
    avg_correlation: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate a plausible (expected_returns, covariance_matrix, asset_names)
    triple: random per-asset volatilities, a random factor-model correlation
    structure with the given average off-diagonal correlation, and returns
    drawn independently of the covariance (so risk and return aren't
    trivially correlated in a way that makes every constraint moot)."""
    rng = np.random.default_rng(seed)

    mu = rng.uniform(*mean_return_range, size=n_assets)
    vol = rng.uniform(*volatility_range, size=n_assets)

    # single-factor correlation model: corr_ij = beta_i * beta_j, scaled so
    # the average off-diagonal correlation matches avg_correlation
    beta = rng.normal(1.0, 0.3, size=n_assets)
    raw_corr = np.outer(beta, beta)
    np.fill_diagonal(raw_corr, raw_corr.diagonal().max())
    raw_corr /= raw_corr.max()
    off_diag_mean = (raw_corr.sum() - np.trace(raw_corr)) / (n_assets * (n_assets - 1))
    scale = avg_correlation / max(off_diag_mean, 1e-6)
    corr = np.clip(raw_corr * scale, -0.95, 0.95)
    np.fill_diagonal(corr, 1.0)
    corr = (corr + corr.T) / 2

    cov = np.outer(vol, vol) * corr
    # ensure PSD (single-factor construction can drift slightly with clipping)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-8, None)
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

    names = [f"ASSET_{i:03d}" for i in range(n_assets)]
    return mu, cov, names
