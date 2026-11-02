"""Uniform probability calibration across every dispatcher algorithm.

Before this platform, `algorithms.vqc_qml.calibration.PlattCalibrator` was wired
into exactly one application (`vqc_classifier.run_directional_classification`)
-- every other binary method either had no calibrated probabilities at all,
or (for `financial_qnn`) calibrated on its own *training*-set scores (a
known, documented simplification in
`algorithms_extended.qiskit_ml_finance.classifier.QNNClassifier.fit`, not a
held-out fit). This module makes held-out Platt calibration available to
*any* dispatcher algorithm that returns `decision_scores` -- which, by
construction of every adapter in `dispatcher.py`, is every BINARY and
ANOMALY algorithm -- by carving a calibration holdout out of the training
set, fitting the calibrator there, and applying it to the real test set.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from algorithms.vqc_qml.calibration import PlattCalibrator
from applications.classification.dispatcher import dispatch
from applications.classification.errors import UnsupportedTaskError
from applications.classification.result import DispatchResult
from applications.classification.task_spec import ClassificationTaskSpec, TaskType

_CALIBRATABLE = {TaskType.BINARY, TaskType.ANOMALY}


def calibrate_dispatch(
    spec: ClassificationTaskSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    calibration_fraction: float = 0.2,
    seed: int | None = None,
) -> DispatchResult:
    """Run `spec.algorithm` with held-out Platt-scaling calibration, uniform
    across every algorithm that produces `decision_scores` (every BINARY and
    ANOMALY dispatcher adapter).

    A `calibration_fraction` slice of `X_train`/`y_train` is held out and
    never used to fit the final model -- it only fits the sigmoid's
    scale/offset, so the resulting `calibration_error` is a fair held-out
    estimate, not a training-set artifact.
    """
    if spec.task_type not in _CALIBRATABLE:
        raise UnsupportedTaskError(
            f"calibrate_dispatch only supports task_type in {sorted(t.value for t in _CALIBRATABLE)}, "
            f"got {spec.task_type.value}"
        )
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    n = len(y_train)
    rng = np.random.default_rng(seed if seed is not None else spec.seed)
    perm = rng.permutation(n)
    n_calib = max(1, int(n * calibration_fraction))
    calib_idx, fit_idx = perm[:n_calib], perm[n_calib:]

    positive_label = np.unique(y_train)[-1] if spec.task_type == TaskType.BINARY else 1

    # 1) fit on the (n - n_calib) remainder, score the held-out calibration slice
    calib_probe = dispatch(spec, X_train[fit_idx], y_train[fit_idx], X_train[calib_idx], y_train[calib_idx])
    if calib_probe.decision_scores is None:
        raise UnsupportedTaskError(
            f"{spec.algorithm.value} does not produce decision_scores; cannot calibrate"
        )
    y_calib_bin = (y_train[calib_idx] == positive_label).astype(float)
    calibrator = PlattCalibrator().fit(calib_probe.decision_scores, y_calib_bin)

    # 2) fit on the FULL training set for the real result, then calibrate its scores
    result = dispatch(spec, X_train, y_train, X_test, y_test)
    if result.decision_scores is None:
        raise UnsupportedTaskError(
            f"{spec.algorithm.value} does not produce decision_scores; cannot calibrate"
        )
    probs = calibrator.predict_proba(result.decision_scores)
    y_test_bin = (np.asarray(y_test) == positive_label).astype(float)
    from core.metrics import calibration_error

    cal_err = calibration_error(probs, y_test_bin)

    new_metrics = dict(result.metrics)
    new_metrics["calibration_error"] = cal_err
    return replace(result, probabilities=probs, calibration_error=cal_err, metrics=new_metrics)


__all__ = ["calibrate_dispatch"]
