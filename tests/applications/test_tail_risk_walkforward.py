import numpy as np
import pytest

from applications.tail_risk.walk_forward import rolling_tail_risk
from applications.tail_risk.errors import EmptySampleError


def _synthetic_pnl(seed: int = 1, n: int = 2000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_t(df=5, size=n) * 0.01


def test_rolling_tail_risk_fold_count_and_no_leakage():
    pnl = _synthetic_pnl(n=2000)
    n_splits = 4
    report = rolling_tail_risk(pnl, n_splits=n_splits, alpha=0.95, min_train_fraction=0.5)

    assert report.n_splits == n_splits
    assert len(report.folds) == n_splits

    # Reconstruct the underlying split to directly verify the no-leakage
    # property this module claims: every fold's train indices strictly
    # precede its test indices, and test windows are non-overlapping and
    # strictly increasing across folds.
    from applications.tail_risk.walk_forward import _PnLSeriesDataset

    dataset = _PnLSeriesDataset(pnl)
    splits = dataset.split_walk_forward(n_splits, min_train_fraction=0.5)
    prev_test_end = -1
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min(), "train indices must strictly precede test indices"
        assert test_idx.min() > prev_test_end, "test windows must be strictly increasing, non-overlapping"
        prev_test_end = test_idx.max()

    # Each fold's reported sizes must match the split it was computed from.
    for fold, (train_idx, test_idx) in zip(report.folds, splits):
        assert fold.n_train == len(train_idx)
        assert fold.n_test == len(test_idx)
        assert fold.cvar_estimate >= fold.var_estimate - 1e-9


def test_var_series_and_cvar_series_accessors():
    pnl = _synthetic_pnl(seed=2, n=1500)
    report = rolling_tail_risk(pnl, n_splits=3, alpha=0.95)
    var_series = report.var_series()
    cvar_series = report.cvar_series()
    assert len(var_series) == 3
    assert len(cvar_series) == 3
    for v, c in zip(var_series, cvar_series):
        assert c >= v - 1e-9


def test_breach_test_present_per_fold():
    pnl = _synthetic_pnl(seed=3, n=1800)
    report = rolling_tail_risk(pnl, n_splits=3, alpha=0.9)
    for fold in report.folds:
        bt = fold.breach_test
        assert bt.n_obs == fold.n_test
        assert 0 <= bt.n_breaches <= bt.n_obs
        assert bt.expected_rate == pytest.approx(0.1)


def test_empty_series_raises():
    with pytest.raises(EmptySampleError):
        rolling_tail_risk(np.array([]), n_splits=3)
