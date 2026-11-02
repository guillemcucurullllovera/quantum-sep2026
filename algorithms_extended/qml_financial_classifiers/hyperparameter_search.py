"""Walk-forward-safe k-fold hyperparameter search for QuantumSVC.

Genuinely new capability: nothing else in the platform does cross-validated
hyperparameter search over (feature map, kernel kind, C) jointly. Not
adapted from qiskit-machine-learning (it delegates this to
sklearn.model_selection.GridSearchCV, which assumes iid data and would leak
future information if used directly on a financial time series) -- this
uses core.dataset_interfaces' walk-forward splitting instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from algorithms_extended.qml_financial_classifiers.qsvc_pipeline import QuantumSVC
from core.dataset_interfaces import FinancialDataset


@dataclass
class SearchResult:
    best_params: dict
    best_score: float
    all_results: list[dict] = field(default_factory=list)


def grid_search_walk_forward(
    dataset: FinancialDataset,
    param_grid: dict[str, list],
    n_splits: int = 3,
    class_weight: str | None = "balanced",
) -> SearchResult:
    """Grid search over `param_grid` (keys: 'feature_map', 'kernel_kind', 'C'),
    scoring each combination by mean walk-forward-fold accuracy. Every fold's
    test window strictly follows its train window (core.dataset_interfaces'
    invariant) -- no combination is scored on data its training set couldn't
    have seen.
    """
    X, y = dataset.features(), dataset.targets()
    splits = dataset.split_walk_forward(n_splits=n_splits)
    keys = sorted(param_grid)
    all_results = []
    best_score, best_params = -np.inf, None

    for combo in product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        fold_scores = []
        for train_idx, test_idx in splits:
            model = QuantumSVC(class_weight=class_weight, **params)
            model.fit(X[train_idx], y[train_idx])
            fold_scores.append(model.score(X[test_idx], y[test_idx]))
        mean_score = float(np.mean(fold_scores))
        all_results.append({**params, "mean_score": mean_score, "fold_scores": fold_scores})
        if mean_score > best_score:
            best_score, best_params = mean_score, params

    return SearchResult(best_params=best_params, best_score=best_score, all_results=all_results)
