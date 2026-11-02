"""VaR-breach backtesting: the standard Kupiec (1995) proportion-of-failures
(POF) test for whether the realized breach rate of a VaR estimate matches
its target exceedance rate -- the Kupiec-test-adjacent diagnostic every real
VaR deployment runs before trusting the number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from applications.tail_risk.errors import EmptySampleError

__all__ = ["BreachTestResult", "count_var_breaches", "kupiec_pof_test", "backtest_var_breaches"]


@dataclass(frozen=True)
class BreachTestResult:
    n_obs: int
    n_breaches: int
    breach_rate: float
    expected_rate: float
    alpha: float
    kupiec_lr_stat: float
    kupiec_p_value: float
    reject_at_95: bool


def count_var_breaches(actual_pnl: np.ndarray, var_estimate) -> np.ndarray:
    """Boolean breach mask: True where the realized loss exceeded the VaR
    estimate, i.e. `actual_pnl < -var_estimate`. `var_estimate` may be a
    scalar (one static VaR applied to the whole series) or an array the same
    length as `actual_pnl` (a rolling/walk-forward VaR series, one estimate
    per observation)."""
    pnl = np.asarray(actual_pnl, dtype=float).reshape(-1)
    if pnl.size == 0:
        raise EmptySampleError("actual_pnl is empty; cannot count VaR breaches.")
    var = np.broadcast_to(np.asarray(var_estimate, dtype=float), pnl.shape)
    return pnl < -var


def kupiec_pof_test(n_obs: int, n_breaches: int, alpha: float) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures likelihood-ratio test.

    H0: the true breach probability equals the target exceedance rate
    p = 1 - alpha. Under H0, LR ~ chi2(1).

        LR = -2 * ln[ (1-p)^(n-x) * p^x / (1 - x/n)^(n-x) * (x/n)^x ]

    where n = n_obs, x = n_breaches. Returns (LR statistic, p-value).
    """
    if n_obs <= 0:
        raise ValueError(f"n_obs must be > 0; got {n_obs}")
    if not (0 <= n_breaches <= n_obs):
        raise ValueError(f"n_breaches must be in [0, n_obs]; got {n_breaches} of {n_obs}")
    if not (0.5 <= alpha < 1.0):
        raise ValueError(f"alpha must be in [0.5, 1.0); got {alpha}")

    p = 1.0 - alpha
    n, x = n_obs, n_breaches
    pi_hat = x / n

    # Guard the degenerate x=0 / x=n cases where log(0) would otherwise fire.
    eps = 1e-12
    pi_hat_c = min(max(pi_hat, eps), 1.0 - eps)
    p_c = min(max(p, eps), 1.0 - eps)

    log_l_null = (n - x) * np.log(1.0 - p_c) + x * np.log(p_c)
    log_l_alt = (n - x) * np.log(1.0 - pi_hat_c) + x * np.log(pi_hat_c)
    lr = float(max(-2.0 * (log_l_null - log_l_alt), 0.0))
    p_value = float(chi2.sf(lr, df=1))
    return lr, p_value


def backtest_var_breaches(actual_pnl: np.ndarray, var_estimate, alpha: float = 0.95) -> BreachTestResult:
    """Count real VaR breaches in `actual_pnl` against `var_estimate` and run
    the Kupiec POF test against the target exceedance rate `1 - alpha`.
    """
    breaches = count_var_breaches(actual_pnl, var_estimate)
    n_obs = int(breaches.size)
    n_breaches = int(breaches.sum())
    breach_rate = n_breaches / n_obs
    expected_rate = 1.0 - alpha
    lr, p_value = kupiec_pof_test(n_obs, n_breaches, alpha)
    return BreachTestResult(
        n_obs=n_obs, n_breaches=n_breaches, breach_rate=breach_rate, expected_rate=expected_rate,
        alpha=alpha, kupiec_lr_stat=lr, kupiec_p_value=p_value, reject_at_95=p_value < 0.05,
    )
