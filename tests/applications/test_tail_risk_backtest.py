import numpy as np
import pytest

from applications.tail_risk.backtest import backtest_var_breaches, count_var_breaches, kupiec_pof_test
from applications.tail_risk.errors import EmptySampleError


def test_count_var_breaches_exact_known_count():
    # Construct a series with EXACTLY 7 breaches out of 100 at var=1.0:
    # values strictly below -1.0 are breaches.
    n, var, k = 100, 1.0, 7
    pnl = np.full(n, 0.5)
    pnl[:k] = -2.0  # -2.0 < -1.0 -> breach
    breaches = count_var_breaches(pnl, var)
    assert breaches.sum() == k
    assert breaches.dtype == bool


def test_backtest_matches_expected_rate_does_not_reject():
    # 5 breaches out of 100 at alpha=0.95 -> breach_rate exactly matches the
    # expected 5% exceedance rate; Kupiec should not reject H0.
    n, var, k = 100, 1.0, 5
    pnl = np.full(n, 0.3)
    pnl[:k] = -1.5
    result = backtest_var_breaches(pnl, var, alpha=0.95)
    assert result.n_obs == n
    assert result.n_breaches == k
    assert result.breach_rate == pytest.approx(0.05)
    assert result.expected_rate == pytest.approx(0.05)
    assert result.kupiec_lr_stat == pytest.approx(0.0, abs=1e-6)
    assert result.kupiec_p_value > 0.9
    assert result.reject_at_95 is False


def test_backtest_over_breaching_rejects_null():
    # 20 breaches out of 100 at alpha=0.95 (expected 5) -- a real,
    # substantial over-breach that the Kupiec test should flag.
    n, var, k = 100, 1.0, 20
    pnl = np.full(n, 0.3)
    pnl[:k] = -2.0
    result = backtest_var_breaches(pnl, var, alpha=0.95)
    assert result.n_breaches == k
    assert result.kupiec_lr_stat > 0.0
    assert result.kupiec_p_value < 0.05
    assert result.reject_at_95 is True


def test_zero_breaches_does_not_crash_log():
    n, var = 50, 1.0
    pnl = np.full(n, 0.2)  # never breaches
    result = backtest_var_breaches(pnl, var, alpha=0.95)
    assert result.n_breaches == 0
    assert np.isfinite(result.kupiec_lr_stat)
    assert 0.0 <= result.kupiec_p_value <= 1.0


def test_all_breach_does_not_crash_log():
    n, var = 20, 1.0
    pnl = np.full(n, -5.0)  # every observation breaches
    result = backtest_var_breaches(pnl, var, alpha=0.95)
    assert result.n_breaches == n
    assert np.isfinite(result.kupiec_lr_stat)


def test_rolling_var_array_input():
    # var_estimate can also be a per-observation array (a rolling VaR
    # series) -- breach counting must broadcast correctly.
    pnl = np.array([-2.0, -0.5, -3.0, 0.1, -1.5])
    var_series = np.array([1.0, 1.0, 2.0, 1.0, 1.0])
    breaches = count_var_breaches(pnl, var_series)
    # -2.0 < -1.0 True; -0.5 < -1.0 False; -3.0 < -2.0 True;
    # 0.1 < -1.0 False; -1.5 < -1.0 True
    assert list(breaches) == [True, False, True, False, True]


def test_empty_actual_pnl_raises():
    with pytest.raises(EmptySampleError):
        count_var_breaches(np.array([]), 1.0)


def test_kupiec_pof_test_direct_call():
    lr, p = kupiec_pof_test(n_obs=250, n_breaches=12, alpha=0.95)
    assert lr >= 0.0
    assert 0.0 <= p <= 1.0
