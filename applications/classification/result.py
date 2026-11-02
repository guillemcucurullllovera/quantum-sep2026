"""Unified result schema every dispatcher adapter returns, regardless of
which of the ~16 underlying algorithms produced it. Before this platform, every
application module defined its own bespoke `@dataclass ...FoldResult` with
different field names for the same concept (see e.g. `vqc_classifier
.DirectionalClassificationResult` vs. `qnn_regression
.VolatilityRegressionFoldResult`) -- fine for one-off comparison scripts, not
for a dispatcher callers can treat generically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from applications.classification.task_spec import Algorithm, TaskType


@dataclass
class DispatchResult:
    """Normalized output of `dispatcher.dispatch()`.

    `metrics` keys vary by `task_type` (see `dispatcher._binary_metrics` /
    `_multiclass_metrics` / `_regression_metrics` / `_anomaly_metrics` for
    the exact contract) but the container shape is always the same, so a
    caller comparing N algorithms on the same task can iterate `metrics`
    without a per-algorithm branch.
    """

    algorithm: Algorithm
    task_type: TaskType
    n_train: int
    n_test: int
    metrics: dict[str, float]
    predictions: np.ndarray
    probabilities: np.ndarray | None = None
    decision_scores: np.ndarray | None = None
    calibration_error: float | None = None
    fit_time_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def metric(self, name: str, default: float = float("nan")) -> float:
        return float(self.metrics.get(name, default))

    def to_summary_dict(self) -> dict[str, Any]:
        """JSON-serializable summary (drops raw arrays) -- what
        `benchmark_report` and the experiment registry log."""
        return {
            "algorithm": self.algorithm.value,
            "task_type": self.task_type.value,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "metrics": dict(self.metrics),
            "calibration_error": self.calibration_error,
            "fit_time_s": self.fit_time_s,
            "extra": {k: v for k, v in self.extra.items() if _is_jsonable(v)},
        }


def _is_jsonable(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None), list, dict, tuple))
