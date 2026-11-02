"""Rolling walk-forward evaluation harness -- the first-class comparison
mechanism for this application layer (not a single train/test split).

Every method (quantum or classical) is evaluated identically: unsupervised
feature extraction happens once over the whole series (causal by
construction), then only the ridge readout is refit per fold, and every
fold's train indices are asserted to be strictly earlier than its test
indices -- the walk-forward-safety invariant is checked, not assumed.
Per-method failures (e.g. a too-short series for one method's window) are
captured into the result rather than aborting the whole comparison, so one
broken method never hides the others.
"""
from __future__ import annotations

import time

import numpy as np

from applications.time_series import methods as method_adapters
from applications.time_series.domain import (
    QUANTUM_METHODS,
    FoldMetric,
    MethodChoice,
    MethodResult,
    TimeSeriesForecastSpec,
    ValidationStrategy,
)
from applications.time_series.errors import InsufficientDataError
from applications.time_series.preprocessing import ArrayDataset
from core.metrics import mae, mse


def walk_forward_folds(n: int, n_splits: int, min_train_fraction: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Delegates to core.dataset_interfaces.FinancialDataset.split_walk_forward
    (via a throwaway ArrayDataset) so this application layer shares the
    platform's one tested walk-forward-safety implementation rather than
    re-deriving it.
    """
    helper = ArrayDataset(np.zeros((n, 1)), np.zeros(n), name="_walk_forward_helper")
    return helper.split_walk_forward(n_splits, min_train_fraction)


def _folds_for(spec: TimeSeriesForecastSpec, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if spec.validation == ValidationStrategy.SINGLE_SPLIT:
        n_test = max(2, int(n * spec.test_fraction))
        if n - n_test < 2:
            raise InsufficientDataError(f"Series of length {n} too small for test_fraction={spec.test_fraction}")
        return [(np.arange(0, n - n_test), np.arange(n - n_test, n))]
    return walk_forward_folds(n, spec.n_splits, spec.min_train_fraction)


def _evaluate_naive(method: MethodChoice, y: np.ndarray, folds) -> list[FoldMetric]:
    fold_metrics = []
    for i, (train_idx, test_idx) in enumerate(folds):
        assert train_idx.max() < test_idx.min(), "walk-forward invariant violated: train overlaps/postdates test"
        y_train, y_test = y[train_idx], y[test_idx]
        if method == MethodChoice.NAIVE_CONSTANT:
            y_pred = np.full(len(test_idx), y_train.mean())
        else:  # NAIVE_PERSISTENCE: predict the last observed actual value (a real random-walk baseline)
            y_pred = y[test_idx - 1]
        fold_metrics.append(
            FoldMetric(fold=i, n_train=len(train_idx), n_test=len(test_idx), mse=mse(y_test, y_pred), mae=mae(y_test, y_pred))
        )
    return fold_metrics


def _evaluate_learned(method: MethodChoice, X: np.ndarray, y: np.ndarray, spec: TimeSeriesForecastSpec, folds) -> tuple[list[FoldMetric], str | None]:
    extracted = method_adapters.extract_features(method, X, y, spec)
    feats, y_aligned = extracted.features, extracted.y_aligned
    fold_metrics = []
    for i, (train_idx, test_idx) in enumerate(folds):
        train_idx = train_idx[train_idx < len(feats)]
        test_idx = test_idx[test_idx < len(feats)]
        if len(train_idx) < 5 or len(test_idx) < 1:
            continue
        assert train_idx.max() < test_idx.min(), "walk-forward invariant violated: train overlaps/postdates test"
        M_train, M_test = feats[train_idx].T, feats[test_idx].T
        W = method_adapters.ridge_fit(M_train, y_aligned[train_idx], spec.ridge_delta)
        y_pred = method_adapters.ridge_predict(W, M_test)
        fold_metrics.append(
            FoldMetric(
                fold=i, n_train=len(train_idx), n_test=len(test_idx),
                mse=mse(y_aligned[test_idx], y_pred), mae=mae(y_aligned[test_idx], y_pred),
            )
        )
    return fold_metrics, extracted.notes


def rolling_evaluate_method(method: MethodChoice, X: np.ndarray, y: np.ndarray, spec: TimeSeriesForecastSpec) -> MethodResult:
    """Evaluate one method across every walk-forward fold (or the single
    train/test split, if spec.validation == SINGLE_SPLIT) and return one
    normalized MethodResult. Never raises for a method-specific failure --
    that is captured into `.failed`/`.error` so the caller can compare
    whatever succeeded without one bad method aborting the whole run.
    """
    n = len(y)
    is_quantum = method in QUANTUM_METHODS
    t0 = time.perf_counter()
    fold_metrics: list[FoldMetric] = []
    failed = False
    error: str | None = None
    notes: str | None = None
    try:
        folds = _folds_for(spec, n)
        if method in (MethodChoice.NAIVE_CONSTANT, MethodChoice.NAIVE_PERSISTENCE):
            fold_metrics = _evaluate_naive(method, y, folds)
        else:
            fold_metrics, notes = _evaluate_learned(method, X, y, spec, folds)
    except Exception as e:  # noqa: BLE001 - deliberately broad: isolate one method's failure from the rest
        failed = True
        error = f"{type(e).__name__}: {e}"

    wall_time_s = time.perf_counter() - t0

    if fold_metrics:
        mses = [f.mse for f in fold_metrics]
        maes = [f.mae for f in fold_metrics]
        mean_mse, std_mse, mean_mae = float(np.mean(mses)), float(np.std(mses)), float(np.mean(maes))
    else:
        mean_mse = std_mse = mean_mae = float("nan")
        if not failed:
            failed = True
            error = "no walk-forward fold produced enough aligned rows to evaluate"

    return MethodResult(
        method=method, is_quantum=is_quantum, fold_metrics=fold_metrics,
        mean_mse=mean_mse, std_mse=std_mse, mean_mae=mean_mae,
        wall_time_s=wall_time_s, notes=notes, failed=failed, error=error,
    )


__all__ = ["walk_forward_folds", "rolling_evaluate_method"]
