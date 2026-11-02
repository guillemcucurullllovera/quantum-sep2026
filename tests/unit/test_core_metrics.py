import numpy as np

from core.metrics import (
    expected_shortfall,
    is_positive_semidefinite,
    max_drawdown,
    mse,
    nearest_psd,
    probability_of_loss,
    sharpe_ratio,
    value_at_risk,
)


def test_mse_zero_for_identical():
    a = np.array([1.0, 2.0, 3.0])
    assert mse(a, a) == 0.0


def test_sharpe_ratio_known_case():
    returns = np.array([0.01] * 252)  # constant positive daily return, zero vol
    assert sharpe_ratio(returns) == 0.0  # std=0 guarded, not inf/nan


def test_max_drawdown_detects_known_drop():
    cum = np.array([1.0, 1.2, 0.9, 1.1])
    dd = max_drawdown(cum)
    # peak 1.2 -> trough 0.9 is a 25% drawdown
    assert abs(dd - (-0.25)) < 1e-9


def test_var_and_cvar_ordering():
    rng = np.random.default_rng(0)
    pnl = rng.normal(0, 1, 10000)
    var95 = value_at_risk(pnl, 0.95)
    cvar95 = expected_shortfall(pnl, 0.95)
    assert cvar95 >= var95  # CVaR is always >= VaR at the same confidence level


def test_probability_of_loss_bounds():
    pnl = np.array([1, -1, 1, -1, 1])
    assert probability_of_loss(pnl) == 0.4


def test_nearest_psd_fixes_negative_eigenvalue():
    # a slightly-broken "kernel matrix" with a tiny negative eigenvalue,
    # the kind a noisy quantum kernel estimate can produce
    broken = np.array([[1.0, 0.99], [0.99, 1.0]]) - 0.001 * np.eye(2)
    broken[0, 1] += 0.02  # break symmetry-adjacent PSD-ness slightly
    fixed = nearest_psd(broken)
    assert is_positive_semidefinite(fixed)
