"""Rolling/walk-forward evaluation as a first-class capability.

Every previously application in this family evaluated a single train/test
split (or, at best, `FinancialDataset.split_walk_forward`'s few folds
averaged into one number -- see `three_way_comparison.py`,
`kernel_classifier.py`). Neither shows *drift*: whether a method's quality
degrades, improves, or stays flat as the training window rolls forward in
time, which is the question that actually matters for a method someone
would deploy. `run_rolling_evaluation` runs one `ClassificationTaskSpec`
across every walk-forward window and reports the metric trajectory, not
just its mean.

No-leakage guarantee: window boundaries come from
`FinancialDataset.split_walk_forward` (strictly increasing, non-overlapping
test windows, train never includes a future index -- enforced there).
Preprocessing statistics (see `preprocessing.Preprocessor`) are fit fresh
per window on that window's training slice only, then applied to that
window's test slice -- never fit on the full series up front, which would
leak later folds' distribution into earlier folds' training.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from applications.classification.dispatcher import default_metric_for, dispatch
from applications.classification.errors import LeakageError
from applications.classification.preprocessing import Preprocessor, PreprocessingConfig
from applications.classification.result import DispatchResult
from applications.classification.task_spec import ClassificationTaskSpec
from core.dataset_interfaces import FinancialDataset


@dataclass
class RollingWindowResult:
    window_index: int
    n_train: int
    n_test: int
    result: DispatchResult


@dataclass
class RollingEvaluationReport:
    """The metric trajectory across every rolling window, plus stability
    diagnostics -- this is what makes rolling evaluation more than "run
    walk-forward and average"."""

    algorithm: str
    task_type: str
    metric_name: str
    windows: list[RollingWindowResult] = field(default_factory=list)

    @property
    def metric_values(self) -> list[float]:
        return [w.result.metric(self.metric_name) for w in self.windows]

    @property
    def mean(self) -> float:
        return float(np.mean(self.metric_values))

    @property
    def std(self) -> float:
        return float(np.std(self.metric_values))

    @property
    def min(self) -> float:
        return float(np.min(self.metric_values))

    @property
    def max(self) -> float:
        return float(np.max(self.metric_values))

    @property
    def drift_slope(self) -> float:
        """Linear-regression slope of the metric against window index -- the
        stability/drift diagnostic. Positive means the metric is improving
        as the window rolls forward, negative means it's degrading (model
        going stale, or a regime shift the algorithm doesn't track), ~0
        means stable. Undefined (returns 0.0) for a single window."""
        values = self.metric_values
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values), dtype=float)
        slope, _ = np.polyfit(x, np.asarray(values, dtype=float), 1)
        return float(slope)

    def to_summary_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "task_type": self.task_type,
            "metric_name": self.metric_name,
            "n_windows": len(self.windows),
            "metric_values": self.metric_values,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "drift_slope": self.drift_slope,
        }


def run_rolling_evaluation(
    spec: ClassificationTaskSpec,
    dataset: FinancialDataset,
    metric_name: str | None = None,
    preprocessing_config: PreprocessingConfig | None = None,
    min_train_fraction: float = 0.5,
) -> RollingEvaluationReport:
    """Run `spec.algorithm` across every walk-forward window of `dataset`
    (`spec.n_splits` windows) and report the per-window metric trajectory.

    If `preprocessing_config` is given, a fresh `Preprocessor` is fit on
    each window's training slice and applied to that window's train+test
    slices independently -- never fit on data outside that window (see
    module docstring for why).
    """
    X, y = dataset.features(), dataset.targets()
    metric_name = metric_name or default_metric_for(spec.task_type)
    splits = dataset.split_walk_forward(n_splits=spec.n_splits, min_train_fraction=min_train_fraction)

    windows: list[RollingWindowResult] = []
    for i, (train_idx, test_idx) in enumerate(splits):
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise LeakageError(f"window {i} has an empty train or test set")
        if train_idx.max() >= test_idx.min():
            raise LeakageError(
                f"window {i}: train index {train_idx.max()} overlaps or precedes "
                f"test start {test_idx.min()} -- walk-forward invariant violated"
            )
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        if preprocessing_config is not None:
            pre = Preprocessor(preprocessing_config).fit(X_train)
            X_train = pre.transform(X_train)
            X_test = pre.transform(X_test)

        result = dispatch(spec, X_train, y_train, X_test, y_test)
        windows.append(RollingWindowResult(window_index=i, n_train=len(train_idx), n_test=len(test_idx), result=result))

    return RollingEvaluationReport(
        algorithm=spec.algorithm.value, task_type=spec.task_type.value, metric_name=metric_name, windows=windows
    )


__all__ = ["RollingWindowResult", "RollingEvaluationReport", "run_rolling_evaluation"]
