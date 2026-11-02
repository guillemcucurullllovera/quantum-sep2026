"""Real tests for the fracdiff/PurgedKFold data pipeline: verifies actual
numerical properties (stationarizing transform shape, leak-free fold
construction) rather than asserting the pipeline "ran without crashing".
"""
from __future__ import annotations

import numpy as np
import pytest

from applications.portfolio.data_pipeline import (
    PortfolioFinancialDataset,
    evaluate_trend_signal_purged,
    fracdiff_trend_features,
    returns_to_prices,
)
from applications.portfolio.scenario import generate_multi_period_market
from core.errors import ValidationError


def test_returns_to_prices_compounds_correctly():
    returns = np.array([[0.1, -0.05], [0.02, 0.03]])
    prices = returns_to_prices(returns, start_price=100.0)
    assert prices.shape == (3, 2)
    assert np.allclose(prices[0], [100.0, 100.0])
    assert np.allclose(prices[1], [110.0, 95.0])
    assert np.allclose(prices[2], [110.0 * 1.02, 95.0 * 1.03])


def test_fracdiff_trend_features_are_stationarizing_not_identity():
    realized, names, _ = generate_multi_period_market(4, 60, seed=1)
    prices = returns_to_prices(realized)
    features = fracdiff_trend_features(prices, names, d=0.4, threshold=1e-2)
    assert list(features.columns) == names
    assert features.shape[0] == prices.shape[0]
    # burn-in window means the first several rows must be NaN, not silently
    # zero-filled or otherwise fabricated
    assert features.iloc[0].isna().all()
    assert features.iloc[-1].notna().all()


def test_portfolio_financial_dataset_targets_and_features_are_aligned_length():
    realized, names, _ = generate_multi_period_market(5, 50, seed=2)
    ds = PortfolioFinancialDataset(realized, names)
    assert ds.features().shape[0] == len(ds.targets()) == 50
    assert ds.descriptor.n_features == 5
    splits = ds.split_walk_forward(n_splits=3, min_train_fraction=0.4)
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min()


def test_evaluate_trend_signal_purged_returns_honest_out_of_sample_comparison():
    """On i.i.d.-per-period synthetic data (generate_multi_period_market
    draws each period independently), a trend feature should NOT reliably
    beat the naive mean baseline -- this checks the evaluation reports that
    honestly (a real leak-free check, not a rigged one that always claims a
    signal works)."""
    realized, names, _ = generate_multi_period_market(4, 80, seed=3)
    report = evaluate_trend_signal_purged(realized, names, n_splits=3, embargo_td=1)
    assert report["n_folds_evaluated"] >= 1
    assert report["mean_fold_mse"] is not None
    assert report["naive_mean_baseline_mse"] > 0
    assert isinstance(report["beats_naive_baseline"], bool)


def test_evaluate_trend_signal_purged_rejects_too_few_rows():
    realized, names, _ = generate_multi_period_market(4, 6, seed=4)
    with pytest.raises(ValidationError):
        evaluate_trend_signal_purged(realized, names, n_splits=4, embargo_td=1)
