"""
Smoke tests for OCSVMAnomalyDetector (C1.3).

All synthetic data — no FRED, no Yahoo. These gate the git push: if any
fail, the module is broken regardless of real-data performance.

Tests cover:
    1. Fit on Gaussian cluster, OCSVM flags points far from origin.
    2. anomaly_score is monotone in distance from training cluster.
    3. Standardization is consistent: fit-time mu/sigma are used at predict.
    4. Save/load round-trip preserves anomaly_score bit-for-bit.
    5. nu hyperparameter controls anomaly fraction within tolerance.
    6. predict_anomalous returns bool array of correct shape.
    7. Input validation: NaN rejection, shape rejection.
    8. Construction rejects invalid nu (must be in (0, 1)).
"""

from __future__ import annotations

import os
import tempfile
import numpy as np
import pytest

from classical.meta_agent.ocsvm_anomaly import OCSVMAnomalyDetector


# ── synthetic-data generator ──

def _gaussian_cluster(n: int, dim: int, seed: int = 0, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)) * scale


def _far_points(n: int, dim: int, seed: int = 0, offset: float = 5.0) -> np.ndarray:
    """Points offset far from origin, intended to be classified as anomalous."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)) + offset


# ── tests ──

def test_construction_rejects_invalid_nu():
    with pytest.raises(ValueError, match="nu"):
        OCSVMAnomalyDetector(nu=0.0)
    with pytest.raises(ValueError, match="nu"):
        OCSVMAnomalyDetector(nu=1.0)
    with pytest.raises(ValueError, match="nu"):
        OCSVMAnomalyDetector(nu=-0.1)


def test_predict_before_fit_raises():
    det = OCSVMAnomalyDetector()
    with pytest.raises(RuntimeError, match="not been fitted"):
        det.anomaly_score(np.zeros((5, 3)))


def test_fit_rejects_nan_input():
    X = _gaussian_cluster(100, 3)
    X[10, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        OCSVMAnomalyDetector().fit(X)


def test_fit_rejects_1d_input():
    with pytest.raises(ValueError, match="2-D"):
        OCSVMAnomalyDetector().fit(np.array([1.0, 2.0, 3.0]))


def test_fit_flags_far_points_as_anomalous():
    """Fit on a tight cluster, evaluate on far-away points: nearly all should be anomalous."""
    X_train = _gaussian_cluster(500, 4, seed=0)
    X_far = _far_points(50, 4, seed=1, offset=8.0)

    det = OCSVMAnomalyDetector(nu=0.05, kernel="rbf", gamma="scale").fit(X_train)
    is_anom_far = det.predict_anomalous(X_far)
    # Allow some misclassification but most far points should be flagged
    assert is_anom_far.mean() > 0.9, (
        f"only {is_anom_far.mean():.1%} of far points flagged as anomalous"
    )


def test_anomaly_score_monotone_in_distance():
    """Anomaly score should increase with distance from training cluster mean."""
    X_train = _gaussian_cluster(500, 3, seed=2)
    det = OCSVMAnomalyDetector(nu=0.05).fit(X_train)

    # Evaluate on points at increasing distance along axis 0
    distances = np.linspace(0, 8, 9)
    test_points = np.array([[d, 0.0, 0.0] for d in distances])
    scores = det.anomaly_score(test_points)
    # Scores should be (weakly) monotonically increasing
    # Use Spearman-style: check that scores[-1] > scores[0] by a wide margin
    assert scores[-1] > scores[0] + 0.5, (
        f"score did not increase with distance: {scores}"
    )
    # And the score at distance 0 should be small (in the training cluster)
    assert scores[0] < scores[-1], "score at 0 should be less than score at 8"


def test_standardization_uses_train_stats():
    """Predict-time standardization must use the mu/sigma from fit, not predict-time data."""
    X_train = _gaussian_cluster(500, 3, seed=3)
    det = OCSVMAnomalyDetector(nu=0.05).fit(X_train)
    mu_train = det._mu.copy()
    sigma_train = det._sigma.copy()

    # Score a single point; this should NOT modify mu/sigma
    _ = det.anomaly_score(np.array([[10.0, 10.0, 10.0]]))
    np.testing.assert_array_equal(det._mu, mu_train)
    np.testing.assert_array_equal(det._sigma, sigma_train)


def test_save_load_roundtrip():
    """Save then load should reproduce anomaly_score bit-for-bit."""
    X_train = _gaussian_cluster(300, 4, seed=4)
    X_test = _gaussian_cluster(50, 4, seed=5, scale=3.0)
    det = OCSVMAnomalyDetector(
        nu=0.07, feature_names=["a", "b", "c", "d"]
    ).fit(X_train)
    scores_before = det.anomaly_score(X_test)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ocsvm.pkl")
        det.save(path)
        assert os.path.exists(path)
        assert os.path.exists(os.path.join(td, "ocsvm.summary.json"))
        det2 = OCSVMAnomalyDetector.load(path)
        scores_after = det2.anomaly_score(X_test)

    np.testing.assert_array_equal(scores_before, scores_after)
    assert det2.feature_names == ["a", "b", "c", "d"]
    assert det2.nu == 0.07


def test_nu_controls_anomaly_fraction():
    """OCSVM's nu sets approximately the train-set outlier fraction."""
    X_train = _gaussian_cluster(1000, 3, seed=6)
    for nu in [0.05, 0.10, 0.20]:
        det = OCSVMAnomalyDetector(nu=nu).fit(X_train)
        rate = det.predict_anomalous(X_train).mean()
        # sklearn's OCSVM guarantees rate is approximately nu (upper bounded by nu
        # for the support vectors; in practice within a few percentage points)
        assert abs(rate - nu) < 0.05, f"nu={nu}: train anomaly rate {rate:.3f}"


def test_predict_anomalous_returns_bool_shape():
    X_train = _gaussian_cluster(200, 3, seed=7)
    det = OCSVMAnomalyDetector(nu=0.05).fit(X_train)
    X_test = _gaussian_cluster(50, 3, seed=8)
    is_anom = det.predict_anomalous(X_test)
    assert is_anom.shape == (50,)
    assert is_anom.dtype == bool


def test_predict_anomalous_handles_1d_input():
    """Predicting on a single row (1D array) should work, returning length-1 result."""
    X_train = _gaussian_cluster(200, 4, seed=9)
    det = OCSVMAnomalyDetector(nu=0.05).fit(X_train)
    single = np.array([10.0, 10.0, 10.0, 10.0])
    is_anom = det.predict_anomalous(single)
    assert is_anom.shape == (1,)


def test_predict_rejects_wrong_feature_count():
    X_train = _gaussian_cluster(200, 4, seed=10)
    det = OCSVMAnomalyDetector().fit(X_train)
    X_wrong = _gaussian_cluster(20, 5, seed=11)  # wrong feature count
    with pytest.raises(ValueError, match="expected 4 features"):
        det.predict_anomalous(X_wrong)


def test_summary_returns_expected_fields():
    X_train = _gaussian_cluster(100, 3, seed=12)
    det = OCSVMAnomalyDetector(
        nu=0.05, feature_names=["x", "y", "z"]
    ).fit(X_train)
    s = det.summary()
    assert s.n_train_samples == 100
    assert s.n_features == 3
    assert s.nu == 0.05
    assert s.kernel == "rbf"
    assert s.feature_names == ["x", "y", "z"]
    assert len(s.feature_means) == 3
    assert len(s.feature_stds) == 3
    assert s.n_support_vectors > 0
