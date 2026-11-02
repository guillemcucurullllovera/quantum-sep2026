"""Parametric VaR estimators, complementing core.metrics' historical
(empirical) VaR/CVaR with distribution-assumption-based alternatives —
useful as a sanity cross-check and for small-sample regimes where the
empirical estimate is noisy.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def parametric_var_gaussian(mean: float, std: float, alpha: float = 0.95) -> float:
    """Gaussian-assumption VaR: -mean + std * z_alpha, returned as a positive
    loss magnitude, consistent with core.metrics.value_at_risk's convention.
    """
    z = norm.ppf(alpha)
    return float(-mean + std * z)


def cornish_fisher_var(
    mean: float, std: float, skew: float, excess_kurtosis: float, alpha: float = 0.95
) -> float:
    """Cornish-Fisher expansion VaR: adjusts the Gaussian quantile for
    skewness/kurtosis, which matters for asymmetric tail-risk payoffs
    (e.g. short-volatility strategies) where Gaussian VaR understates risk.
    """
    z = norm.ppf(alpha)
    z_cf = (
        z
        + (z**2 - 1) * skew / 6
        + (z**3 - 3 * z) * excess_kurtosis / 24
        - (2 * z**3 - 5 * z) * skew**2 / 36
    )
    return float(-mean + std * z_cf)
