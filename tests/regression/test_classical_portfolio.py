import numpy as np

from classical.portfolio import markowitz_min_variance, equal_weight_baseline, portfolio_variance


def test_markowitz_weights_sum_to_one_and_are_long_only():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 5))
    cov = A @ A.T + np.eye(5) * 0.1  # PSD
    w = markowitz_min_variance(cov)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-8).all()


def test_markowitz_beats_equal_weight_on_variance():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 5))
    cov = A @ A.T + np.eye(5) * 0.1
    w_opt = markowitz_min_variance(cov)
    w_eq = equal_weight_baseline(5)
    assert portfolio_variance(w_opt, cov) <= portfolio_variance(w_eq, cov) + 1e-9
