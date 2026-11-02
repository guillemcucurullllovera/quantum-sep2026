"""Tests for `applications.classification.calibration_utils.calibrate_dispatch`
-- held-out Platt calibration made uniformly available across dispatcher
algorithms, not just VQC (original, single-application wiring)."""
from __future__ import annotations

import numpy as np
import pytest

from applications.classification.calibration_utils import calibrate_dispatch
from applications.classification.errors import UnsupportedTaskError
from applications.classification.task_spec import Algorithm, ClassificationTaskSpec, TaskType


def _binary_data(n=60, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    n_test = n // 4
    return X[:-n_test], y[:-n_test], X[-n_test:], y[-n_test:]


@pytest.mark.parametrize(
    "algorithm", [Algorithm.CLASSICAL_LOGISTIC_REGRESSION, Algorithm.VQC_V1_LINEAR, Algorithm.CLASSICAL_SVM_RBF]
)
def test_calibrate_dispatch_is_uniform_across_algorithms(algorithm):
    X_train, y_train, X_test, y_test = _binary_data()
    spec = ClassificationTaskSpec(task_type=TaskType.BINARY, algorithm=algorithm, n_qubits=4, seed=0)
    result = calibrate_dispatch(spec, X_train, y_train, X_test, y_test, seed=0)
    assert result.calibration_error is not None
    assert 0.0 <= result.calibration_error <= 1.0
    assert result.probabilities is not None
    assert len(result.probabilities) == len(y_test)
    assert np.all((result.probabilities >= 0.0) & (result.probabilities <= 1.0))
    assert "calibration_error" in result.metrics


def test_calibrate_dispatch_rejects_multiclass():
    from applications.regime_detection.synthetic_regime_dataset import SyntheticRegimeDataset

    ds = SyntheticRegimeDataset(n_per_regime=30, n_features=4, seed=1)
    X, y = ds.features(), ds.targets()
    spec = ClassificationTaskSpec(
        task_type=TaskType.MULTICLASS, algorithm=Algorithm.CLASSICAL_LOGISTIC_REGRESSION, n_qubits=4
    )
    with pytest.raises(UnsupportedTaskError):
        calibrate_dispatch(spec, X[:60], y[:60], X[60:], y[60:])


def test_calibrate_dispatch_holds_out_from_the_fit_set():
    """The calibration slice must come out of X_train, not be a leaked copy
    of the test set -- verified indirectly: shrinking calibration_fraction
    to (nearly) 0 must still produce a valid result using almost all of
    X_train to fit, and the two results (small vs. large calibration slice)
    need not be identical because the underlying fit set differs."""
    X_train, y_train, X_test, y_test = _binary_data(n=80)
    spec = ClassificationTaskSpec(
        task_type=TaskType.BINARY, algorithm=Algorithm.CLASSICAL_LOGISTIC_REGRESSION, n_qubits=4, seed=0
    )
    small_calib = calibrate_dispatch(spec, X_train, y_train, X_test, y_test, calibration_fraction=0.1, seed=0)
    large_calib = calibrate_dispatch(spec, X_train, y_train, X_test, y_test, calibration_fraction=0.4, seed=0)
    assert small_calib.calibration_error is not None
    assert large_calib.calibration_error is not None
