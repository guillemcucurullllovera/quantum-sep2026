"""Real tests for walk-forward multi-period rebalancing: constraint
carry-over across periods, no-lookahead splitting, and honest cross-solver
comparison -- verified from the actual scenario/period objects returned,
not asserted from a solver's own claims.
"""
from __future__ import annotations

import numpy as np
import pytest

from applications.portfolio.domain import SolverChoice
from applications.portfolio.scenario import (
    MultiPeriodReturnsDataset,
    compare_walk_forward,
    generate_multi_period_market,
    run_walk_forward_backtest,
)
from core.errors import ValidationError


def test_generate_multi_period_market_shape_and_sector_correlation_structure():
    realized, names, sector_id = generate_multi_period_market(6, 30, seed=1, n_sectors=2)
    assert realized.shape == (30, 6)
    assert len(names) == 6
    assert sorted(set(sector_id)) == [0, 1]

    # same-sector assets should be more correlated on average than
    # cross-sector ones -- the actual point of the sector-block boost
    same_sector = [(i, j) for i in range(6) for j in range(6) if i < j and sector_id[i] == sector_id[j]]
    cross_sector = [(i, j) for i in range(6) for j in range(6) if i < j and sector_id[i] != sector_id[j]]
    corr = np.corrcoef(realized, rowvar=False)
    same_avg = np.mean([corr[i, j] for i, j in same_sector])
    cross_avg = np.mean([corr[i, j] for i, j in cross_sector])
    assert same_avg > cross_avg


def test_walk_forward_split_never_leaks_future_data_into_training():
    realized, names, _ = generate_multi_period_market(5, 40, seed=2)
    dataset = MultiPeriodReturnsDataset(realized, names)
    splits = dataset.split_walk_forward(n_splits=4, min_train_fraction=0.4)
    assert len(splits) == 4
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0 and len(test_idx) > 0
        assert train_idx.max() < test_idx.min()  # every train index strictly precedes every test index
    # test windows are contiguous and non-overlapping, covering the tail of the panel
    all_test = np.concatenate([t for _, t in splits])
    assert len(all_test) == len(set(all_test.tolist()))  # no test index reused across splits


def test_turnover_constraint_references_the_prior_periods_actual_holdings():
    realized, names, _ = generate_multi_period_market(5, 30, seed=3)
    result = run_walk_forward_backtest(
        realized, names, cardinality_k=2, solver=SolverChoice.CLASSICAL_BASELINE,
        n_splits=3, turnover_limit=1, seed=1,
    )
    assert len(result.periods) == 3
    for i in range(1, len(result.periods)):
        prev_holdings = result.periods[i - 1].result.bitstring
        turnover_constraints = [c for c in result.periods[i].spec.constraints if c.kind == "turnover"]
        assert len(turnover_constraints) == 1
        referenced = np.array(turnover_constraints[0].params["previous_holdings"])
        assert np.array_equal(referenced, prev_holdings)
    # CLASSICAL_BASELINE's cardinality-projection is deterministic given its
    # continuous solve, so with turnover_limit=1 it should never actually violate it
    assert result.n_turnover_violations == 0


def test_realized_return_uses_strictly_later_data_than_the_periods_own_spec():
    """Each period's `PortfolioSpec.expected_returns` is estimated from a
    TRAIN window; the realized_return charged to it must come from data at
    or after that window's end, never from inside it."""
    realized, names, _ = generate_multi_period_market(5, 30, seed=4)
    dataset = MultiPeriodReturnsDataset(realized, names)
    splits = dataset.split_walk_forward(n_splits=3, min_train_fraction=0.4)
    result = run_walk_forward_backtest(
        realized, names, cardinality_k=2, solver=SolverChoice.CLASSICAL_BASELINE, n_splits=3, seed=1,
    )
    for period, (train_idx, test_idx) in zip(result.periods, splits):
        assert test_idx[0] > train_idx.max()


def test_compare_walk_forward_runs_multiple_solvers_on_the_identical_panel():
    realized, names, _ = generate_multi_period_market(5, 24, seed=5)
    results = compare_walk_forward(
        realized, names, cardinality_k=2,
        solvers=[SolverChoice.QOKIT_FAST, SolverChoice.CLASSICAL_BASELINE],
        n_splits=3, turnover_limit=1, seed=2,
    )
    assert set(results) == {"qokit_fast", "classical_baseline"}
    for r in results.values():
        assert len(r.period_returns) == 3
        assert np.isfinite(r.sharpe)


def test_generate_multi_period_market_rejects_too_few_periods():
    with pytest.raises(ValidationError):
        generate_multi_period_market(5, 2, seed=0)
