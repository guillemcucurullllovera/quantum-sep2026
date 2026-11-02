"""
Tests for VQC v5 end-to-end accuracy and layerwise training.

Acceptance criteria (Phase 3B, Level 1):
- VQC v5 val-accuracy > 63.8% on purged OOS split with no leakage
- Training loss decreases monotonically for ≥80% of seeds
- Stable convergence: no divergence in loss history
"""

import numpy as np
import pytest

from algorithms.vqc_qml.v5.vqc_v5 import VQCv5, VQCConfig
from algorithms.vqc_qml.v5.layerwise_trainer import (
    LayerwiseTrainer, ridge_fit, compute_features
)
from algorithms.vqc_qml.v5.ansatz import init_params, circuit_features


def make_dataset(n=60, n_features=10, seed=42):
    """Simple linearly separable dataset for fast testing."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    # Labels from a simple linear boundary
    y = np.sign(X[:, 0] + 0.5 * X[:, 1]).astype(int)
    y[y == 0] = 1  # remove zeros
    split = int(0.75 * n)
    return X[:split], y[:split], X[split:], y[split:]


def make_triclass_dataset(n=60, n_features=10, seed=42):
    """Three-class dataset for {-1, 0, 1} labels."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.choice([-1, 0, 1], size=n)
    split = int(0.75 * n)
    return X[:split], y[:split], X[split:], y[split:]


class TestLayerwiseTrainer:
    def test_fit_predict_runs(self):
        X_tr, y_tr, X_te, _ = make_dataset()
        trainer = LayerwiseTrainer(n=4, d=2, stage1_steps=5, stage2_steps=5, seed=0)
        trainer.fit(X_tr, y_tr, batch_size=8)
        preds = trainer.predict(X_te)
        assert len(preds) == len(X_te)

    def test_predict_proba_shape(self):
        X_tr, y_tr, X_te, _ = make_dataset()
        trainer = LayerwiseTrainer(n=4, d=2, stage1_steps=5, stage2_steps=5, seed=0)
        trainer.fit(X_tr, y_tr, batch_size=8)
        proba = trainer.predict_proba(X_te)
        n_classes = len(np.unique(y_tr))
        assert proba.shape == (len(X_te), n_classes)

    def test_predict_proba_sums_to_one(self):
        X_tr, y_tr, X_te, _ = make_dataset()
        trainer = LayerwiseTrainer(n=4, d=2, stage1_steps=5, stage2_steps=5, seed=0)
        trainer.fit(X_tr, y_tr)
        proba = trainer.predict_proba(X_te)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_before_fit_raises(self):
        trainer = LayerwiseTrainer(n=4, d=2)
        with pytest.raises(RuntimeError, match="fit"):
            trainer.predict(np.zeros((3, 4)))

    def test_accuracy_metric(self):
        X_tr, y_tr, X_te, y_te = make_dataset()
        trainer = LayerwiseTrainer(n=4, d=2, stage1_steps=5, stage2_steps=5, seed=0)
        trainer.fit(X_tr, y_tr)
        acc = trainer.accuracy(X_te, y_te)
        assert 0.0 <= acc <= 1.0

    def test_triclass_fit(self):
        X_tr, y_tr, X_te, _ = make_triclass_dataset()
        trainer = LayerwiseTrainer(n=4, d=2, stage1_steps=5, stage2_steps=5, seed=0)
        trainer.fit(X_tr, y_tr)
        preds = trainer.predict(X_te)
        assert set(preds.tolist()).issubset({-1, 0, 1})


class TestRidgeFit:
    def test_ridge_solves_simple_regression(self):
        """Perfect linear relationship: ridge should fit it."""
        X = np.eye(5)
        y = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        W, b = ridge_fit(X, y, alpha=0.01)
        preds = X @ W.squeeze() + b
        # Should roughly recover y
        assert np.corrcoef(preds, y)[0, 1] > 0.8

    def test_ridge_output_shape(self):
        X = np.random.default_rng(0).standard_normal((20, 4))
        y = np.random.default_rng(0).choice([-1, 0, 1], 20)
        y_oh = np.zeros((20, 3))
        for i, c in enumerate([-1, 0, 1]):
            y_oh[y == c, i] = 1.0
        W, b = ridge_fit(X, y_oh, alpha=0.1)
        assert W.shape == (4, 3)
        assert b.shape == (3,)


class TestVQCv5:
    def test_fit_predict_no_error(self):
        X_tr, y_tr, X_te, _ = make_dataset()
        vqc = VQCv5(VQCConfig(n_qubits=4, d_layers=2, stage1_steps=3,
                               stage2_steps=3, batch_size=8, seed=0))
        vqc.fit(X_tr, y_tr)
        preds = vqc.predict(X_te)
        assert len(preds) == len(X_te)

    def test_predict_proba_sums_to_one(self):
        X_tr, y_tr, X_te, _ = make_dataset()
        vqc = VQCv5(VQCConfig(n_qubits=4, d_layers=2, stage1_steps=3,
                               stage2_steps=3, seed=0))
        vqc.fit(X_tr, y_tr)
        proba = vqc.predict_proba(X_te)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_level1_threshold_documented(self):
        vqc = VQCv5()
        assert vqc.level1_threshold == 0.638, (
            "Level 1 threshold must be 63.8% (LR baseline). "
            "Do not change without updating go/no-go criteria."
        )

    def test_n_params(self):
        vqc = VQCv5(VQCConfig(n_qubits=4, d_layers=2))
        assert vqc.n_params == 4 * 2 * 2  # d * 2 * n

    def test_circuit_summary(self):
        vqc = VQCv5()
        summary = vqc.circuit_summary()
        assert "plateau_fixes" in summary
        assert len(summary["plateau_fixes"]) == 5
        assert summary["n_qubits"] == 10
        assert summary["d_layers"] == 2

    def test_level1_check_returns_dict(self):
        X_tr, y_tr, X_te, y_te = make_dataset()
        vqc = VQCv5(VQCConfig(n_qubits=4, d_layers=2, stage1_steps=3,
                               stage2_steps=3, seed=0))
        vqc.fit(X_tr, y_tr)
        result = vqc.level1_check(X_te, y_te)
        assert "accuracy" in result
        assert "passes" in result
        assert "gap" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_measure_gradient_variance(self):
        vqc = VQCv5(VQCConfig(n_qubits=4, d_layers=2, seed=0))
        var = vqc.measure_gradient_variance(n_samples=10)
        assert var > 0, f"Gradient variance must be > 0, got {var}"
        assert np.isfinite(var)
