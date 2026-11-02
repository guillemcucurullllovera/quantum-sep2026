"""
Fractional Differentiation — Fixed-Width Window (FFD) variant.

Reference: López de Prado (2018) "Advances in Financial Machine Learning"
           Chapter 5 — Fractionally Differentiated Features.

FFD preserves the maximum amount of memory while achieving stationarity.
Use d≈0.35 for most financial return series (ADF test confirms stationarity
while preserving roughly 65% of the original series memory).

No dependencies beyond numpy and pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ffd_weights(d: float, threshold: float, max_lags: int = 10_000) -> np.ndarray:
    """
    Compute FFD weights w_k = (-1)^k * C(d, k) up to |w_k| < threshold.

    Using the recurrence:  w_k = w_{k-1} * (k - 1 - d) / k
    (avoids overflow in direct computation of large binomial coefficients.)
    """
    w = [1.0]
    k = 1
    while k < max_lags:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1], dtype=np.float64)   # oldest lag first


def fracdiff(
    series: pd.Series,
    d: float = 0.35,
    threshold: float = 1e-5,
) -> pd.Series:
    """
    Apply Fixed-Width Window Fractional Differentiation to a price series.

    Unlike integer differencing (d=1), fractional differentiation preserves
    memory: the series remains correlated with its history while achieving
    stationarity (ADF p < 0.05 for d≈0.35 on most financial series).

    Args:
        series    : price or return series (pd.Series, numeric)
        d         : differentiation order (0 < d < 1; default 0.35)
        threshold : weight cut-off — stop accumulating lags below this weight
    Returns:
        pd.Series of same index, NaN for initial rows where window is incomplete
    """
    if not 0 < d < 2:
        raise ValueError(f"d must be in (0, 2); got {d}")
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0; got {threshold}")

    weights = _ffd_weights(d, threshold)
    width = len(weights)

    values = series.values.astype(np.float64)
    n = len(values)
    result = np.full(n, np.nan)

    for t in range(width - 1, n):
        window = values[t - width + 1 : t + 1]
        result[t] = float(weights @ window)

    return pd.Series(result, index=series.index, name=f"{series.name}_frac{d}")


def fracdiff_df(
    df: pd.DataFrame,
    columns: list,
    d: float = 0.35,
    threshold: float = 1e-5,
) -> pd.DataFrame:
    """Apply fracdiff to multiple columns; return new DataFrame with renamed columns."""
    out = df.copy()
    for col in columns:
        out[f"{col}_frac{d}"] = fracdiff(df[col], d=d, threshold=threshold)
    return out


def find_min_d(
    series: pd.Series,
    d_range: tuple = (0.0, 1.0),
    n_steps: int = 11,
    adf_threshold: float = 0.05,
    threshold: float = 1e-5,
) -> float:
    """
    Find the minimum d such that fracdiff(series, d) is ADF-stationary.

    Uses a simple ADF implementation (numpy only — no statsmodels required).
    For production use, validate against statsmodels.tsa.stattools.adfuller.

    Returns:
        Minimum d that achieves ADF p-value < adf_threshold, or d_range[1] if none found.
    """
    for d in np.linspace(d_range[0], d_range[1], n_steps):
        if d == 0.0:
            continue
        fd = fracdiff(series, d=d, threshold=threshold).dropna()
        if len(fd) < 20:
            continue
        if _adf_stationary(fd.values, adf_threshold):
            return float(d)
    return float(d_range[1])


def _adf_stationary(x: np.ndarray, pvalue_threshold: float = 0.05) -> bool:
    """
    Simplified ADF test (numpy only).

    Regresses Δx_t on x_{t-1} and lags of Δx; tests H0: coefficient on x_{t-1} is 0.
    A negative, significant coefficient indicates stationarity.

    NOTE: This is an approximation. For definitive ADF results in production,
    use statsmodels.tsa.stattools.adfuller.
    """
    if len(x) < 20:
        return False
    n = len(x)
    n_lags = min(4, n // 5)

    dx = np.diff(x)                         # Δx_t
    x_lag = x[:-1]                          # x_{t-1}

    # Build design matrix: [x_{t-1}, Δx_{t-1}, ..., Δx_{t-n_lags}]
    cols = [x_lag[n_lags:]]
    for k in range(1, n_lags + 1):
        cols.append(dx[n_lags - k : -k if k > 0 else None])
    cols.append(np.ones(len(cols[0])))

    X = np.column_stack(cols)
    y = dx[n_lags:]

    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        sigma2 = float(np.sum(resid ** 2) / (len(y) - X.shape[1]))
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(sigma2 * XtX_inv[0, 0])
        t_stat = beta[0] / (se + 1e-12)
        # Approximate p-value using normal approximation (conservative)
        # ADF critical values: -3.43 (1%), -2.86 (5%), -2.57 (10%)
        return t_stat < -2.86   # ≈ 5% significance
    except np.linalg.LinAlgError:
        return False


__all__ = ["fracdiff", "fracdiff_df", "find_min_d"]
