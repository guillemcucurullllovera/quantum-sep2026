import numpy as np

from classical.risk import cornish_fisher_var, parametric_var_gaussian, value_at_risk


def test_parametric_var_matches_empirical_for_gaussian_data():
    rng = np.random.default_rng(0)
    pnl = rng.normal(0, 1, 200_000)
    empirical = value_at_risk(pnl, 0.95)
    parametric = parametric_var_gaussian(mean=0.0, std=1.0, alpha=0.95)
    assert abs(empirical - parametric) < 0.02


def test_cornish_fisher_reduces_to_gaussian_when_no_skew_kurtosis():
    g = parametric_var_gaussian(mean=0.0, std=1.0, alpha=0.95)
    cf = cornish_fisher_var(mean=0.0, std=1.0, skew=0.0, excess_kurtosis=0.0, alpha=0.95)
    assert abs(g - cf) < 1e-9
