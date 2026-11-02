"""Shared metric functions used across algorithms, applications, and
benchmarks. Every function here is a plain, dependency-light, unit-tested
building block — no hidden state, no I/O.
"""
from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=float) - risk_free / periods_per_year
    if r.std(ddof=1) == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / r.std(ddof=1))


def max_drawdown(cumulative_returns: np.ndarray) -> float:
    c = np.asarray(cumulative_returns, dtype=float)
    running_max = np.maximum.accumulate(c)
    drawdowns = (c - running_max) / running_max
    return float(drawdowns.min())


def value_at_risk(pnl_samples: np.ndarray, alpha: float = 0.95) -> float:
    """Historical VaR: the loss threshold not exceeded with probability alpha.

    Returns a positive number representing the loss magnitude at the given
    confidence level (i.e. VaR_95 = 0.03 means a 3% loss is the 95th
    percentile worst outcome).
    """
    losses = -np.asarray(pnl_samples, dtype=float)
    return float(np.percentile(losses, alpha * 100))


def expected_shortfall(pnl_samples: np.ndarray, alpha: float = 0.95) -> float:
    """Historical CVaR / Expected Shortfall: mean loss beyond the VaR threshold."""
    losses = -np.asarray(pnl_samples, dtype=float)
    var = np.percentile(losses, alpha * 100)
    tail = losses[losses >= var]
    if tail.size == 0:
        return float(var)
    return float(tail.mean())


def probability_of_loss(pnl_samples: np.ndarray) -> float:
    pnl = np.asarray(pnl_samples, dtype=float)
    return float(np.mean(pnl < 0))


def calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error for a binary probabilistic classifier."""
    probs, labels = np.asarray(probs, dtype=float), np.asarray(labels, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def is_positive_semidefinite(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    m = np.asarray(matrix, dtype=float)
    m = (m + m.T) / 2  # symmetrize against float noise
    eigvals = np.linalg.eigvalsh(m)
    return bool(eigvals.min() >= -tol)


def nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Project a near-PSD matrix (e.g. a noisy estimated quantum kernel) onto
    the PSD cone by clipping negative eigenvalues, per Higham's method.
    """
    m = np.asarray(matrix, dtype=float)
    m = (m + m.T) / 2
    eigvals, eigvecs = np.linalg.eigh(m)
    eigvals_clipped = np.clip(eigvals, 0, None)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
