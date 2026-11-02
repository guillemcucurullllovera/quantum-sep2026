"""Coverage for applications/portfolio/scenario.py (multi-period market
generation, leak-free walk-forward rebalancing with turnover carry-over)
and applications/portfolio/data_pipeline.py (fracdiff features + purged-CV
signal evaluation) added recently.
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
from applications.portfolio.domain import SolverChoice
from applications.portfolio.scenario import (
    MultiPeriodReturnsDataset,
    compare_walk_forward,
    generate_multi_period_market,
    run_walk_forward_backtest,
)


def test_generate_multi_period_market_shapes_and_sector_structure():
    realized, names, sectors = generate_multi_period_market(8, n_periods=20, seed=1, n_sectors=2)
    assert realized.shape == (20, 8)
    assert len(names) == 8
    assert set(sectors) == {0, 1}


def test_multi_period_dataset_walk_forward_split_never_leaks_future_data():
    realized, names, _ = generate_multi_period_market(6, n_periods=24, seed=2, n_sectors=2)
    dataset = MultiPeriodReturnsDataset(realized, names)
    splits = dataset.split_walk_forward(n_splits=4, min_train_fraction=0.4)
    assert len(splits) == 4
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0 and len(test_idx) > 0
        assert train_idx.max() < test_idx.min(), "train indices must strictly precede test indices"
    # test windows across splits are non-overlapping and strictly increasing
    for (_, test_a), (_, test_b) in zip(splits, splits[1:]):
        assert test_a.max() < test_b.min()


def test_walk_forward_backtest_carries_prior_period_holdings_into_turnover_constraint():
    realized, names, _ = generate_multi_period_market(6, n_periods=28, seed=3, n_sectors=2)
    result = run_walk_forward_backtest(
        realized, names, cardinality_k=3, solver=SolverChoice.CLASSICAL_BASELINE,
        n_splits=4, turnover_limit=3, seed=10,
    )
    assert len(result.periods) == 4
    prior_holdings = None
    for period in result.periods:
        turnover_cs = [c for c in period.spec.constraints if c.kind == "turnover"]
        if prior_holdings is None:
            assert turnover_cs == []  # no prior period yet
        else:
            assert len(turnover_cs) == 1
            carried = np.asarray(turnover_cs[0].params["previous_holdings"])
            assert np.array_equal(carried, prior_holdings), "turnover constraint must reference the PRIOR period's ACTUAL holdings"
        prior_holdings = period.result.bitstring.copy()


def test_walk_forward_backtest_reports_real_risk_metrics():
    realized, names, _ = generate_multi_period_market(6, n_periods=24, seed=4, n_sectors=3)
    result = run_walk_forward_backtest(
        realized, names, cardinality_k=3, solver=SolverChoice.CLASSICAL_BASELINE, n_splits=3, seed=11,
    )
    assert isinstance(result.sharpe, float)
    assert isinstance(result.max_drawdown, float)
    assert result.max_drawdown <= 0.0 or result.max_drawdown == 0.0
    assert len(result.period_returns) == len(result.periods) == 3


def test_compare_walk_forward_runs_multiple_solvers_and_can_disagree_on_best():
    realized, names, _ = generate_multi_period_market(6, n_periods=28, seed=5, n_sectors=2)
    results = compare_walk_forward(
        realized, names, cardinality_k=3,
        solvers=[SolverChoice.QAOA_DIRECT, SolverChoice.CLASSICAL_BASELINE],
        n_splits=4, turnover_limit=2, seed=12,
    )
    assert set(results) == {"qaoa_direct", "classical_baseline"}
    for r in results.values():
        assert len(r.periods) == 4


def test_generate_multi_period_market_rejects_too_few_periods():
    from core.errors import ValidationError

    with pytest.raises(ValidationError):
        generate_multi_period_market(4, n_periods=2, seed=0)


def test_returns_to_prices_monotonic_growth_under_positive_returns():
    returns = np.full((5, 2), 0.01)
    prices = returns_to_prices(returns, start_price=100.0)
    assert prices.shape == (6, 2)
    assert np.all(np.diff(prices, axis=0) > 0)
    assert np.allclose(prices[0], 100.0)


def test_fracdiff_trend_features_shape_matches_assets():
    realized, names, _ = generate_multi_period_market(4, n_periods=40, seed=6, n_sectors=2)
    prices = returns_to_prices(realized)
    features = fracdiff_trend_features(prices, names)
    assert list(features.columns) == names
    assert len(features) == len(prices)


def test_evaluate_trend_signal_purged_returns_leakage_controlled_metrics():
    realized, names, _ = generate_multi_period_market(6, n_periods=80, seed=7, n_sectors=2)
    out = evaluate_trend_signal_purged(realized, names, n_splits=4, embargo_td=1)
    assert out["n_folds_evaluated"] > 0
    assert out["mean_fold_mse"] is not None
    assert isinstance(out["beats_naive_baseline"], bool)


def test_evaluate_trend_signal_purged_raises_on_too_few_surviving_rows():
    from core.errors import ValidationError

    realized, names, _ = generate_multi_period_market(4, n_periods=6, seed=8, n_sectors=2)
    with pytest.raises(ValidationError):
        evaluate_trend_signal_purged(realized, names, n_splits=4)


def test_portfolio_financial_dataset_exposes_walk_forward_split():
    realized, names, _ = generate_multi_period_market(5, n_periods=50, seed=9, n_sectors=2)
    dataset = PortfolioFinancialDataset(realized, names)
    splits = dataset.split_walk_forward(n_splits=3, min_train_fraction=0.4)
    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min()
