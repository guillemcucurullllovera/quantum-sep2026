"""
Tests for classical.time_series — classical baselines and Mamba SSM.

Sklearn IS installed in this venv (1.9.0) so those tests run
for real. lightgbm is NOT installed here — its one test is skipped with a
clear reason rather than silently omitted or faked; the mamba-ssm CUDA
package is NOT installed either, but SimpleMambaSSM has no dependency on it
(pure numpy) so those tests always run.
"""

import numpy as np
import pytest

from classical.time_series.mamba_ssm_baseline import SimpleMambaSSM, MambaConfig


def make_dataset(n_samples: int = 120, n_features: int = 8, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = rng.choice([-1, 0, 1], size=n_samples)
    split = int(0.8 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


# ── Mamba SSM (pure numpy — always runs) ────────────────────────────────────

class TestMambaSSMBaseline:
    def test_fit_predict_no_error(self):
        X_tr, y_tr, X_te, _ = make_dataset()
        model = SimpleMambaSSM()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert preds is not None
        assert len(preds) == len(X_te)

    def test_predict_labels_in_valid_set(self):
        X_tr, y_tr, X_te, _ = make_dataset()
        model = SimpleMambaSSM()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert set(preds).issubset({-1, 0, 1}), f"Unexpected labels: {set(preds)}"

    def test_predict_proba_shape(self):
        X_tr, y_tr, X_te, _ = make_dataset()
        model = SimpleMambaSSM()
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        assert proba.shape == (len(X_te), 3), f"Wrong shape: {proba.shape}"

    def test_predict_proba_sums_to_one(self):
        X_tr, y_tr, X_te, _ = make_dataset()
        model = SimpleMambaSSM()
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6,
                                    err_msg="Predicted probabilities must sum to 1.")

    def test_predict_proba_non_negative(self):
        X_tr, y_tr, X_te, _ = make_dataset()
        model = SimpleMambaSSM()
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        assert (proba >= 0).all(), "Probabilities must be non-negative."

    def test_accuracy_method(self):
        X_tr, y_tr, X_te, y_te = make_dataset()
        model = SimpleMambaSSM()
        model.fit(X_tr, y_tr)
        acc = model.accuracy(X_te, y_te)
        assert 0.0 <= acc <= 1.0, f"Accuracy out of [0,1]: {acc}"

    def test_predict_before_fit_raises(self):
        model = SimpleMambaSSM()
        X_te, *_ = make_dataset()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(X_te)

    def test_custom_config(self):
        cfg = MambaConfig(d_model=8, d_state=8, n_layers=1)
        X_tr, y_tr, X_te, _ = make_dataset()
        model = SimpleMambaSSM(config=cfg)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert len(preds) == len(X_te)

    def test_selective_dynamics_vary_with_input(self):
        """Selective SSM: Δ, B, C must be input-dependent (change with x)."""
        cfg = MambaConfig(d_model=4, d_state=4, n_layers=1)
        model = SimpleMambaSSM(config=cfg)
        X_tr, y_tr, _, _ = make_dataset()
        model.fit(X_tr, y_tr)
        X1 = np.zeros((5, X_tr.shape[1]))
        X2 = np.ones((5, X_tr.shape[1]))
        p1 = model.predict_proba(X1)
        p2 = model.predict_proba(X2)
        assert not np.allclose(p1, p2), (
            "SSM produces identical output for different inputs — "
            "selective mechanism is not working."
        )


# ── Classical baselines (sklearn required — installed, so these run) ────────

class TestClassicalBaselines:
    @pytest.fixture(autouse=True)
    def skip_if_no_sklearn(self):
        pytest.importorskip("sklearn", reason="scikit-learn not installed")

    def test_logistic_baseline_smoke(self):
        from classical.time_series.classical_baselines import LogisticBaseline
        X_tr, y_tr, X_te, y_te = make_dataset()
        lb = LogisticBaseline()
        lb.fit(X_tr, y_tr)
        preds = lb.predict(X_te)
        assert len(preds) == len(X_te)
        assert set(preds).issubset({-1, 0, 1})

    def test_ridge_baseline_smoke(self):
        from classical.time_series.classical_baselines import RidgeBaseline
        X_tr, y_tr, X_te, y_te = make_dataset()
        rb = RidgeBaseline()
        rb.fit(X_tr, y_tr)
        preds = rb.predict(X_te)
        assert len(preds) == len(X_te)

    def test_lgbm_baseline_smoke(self):
        pytest.importorskip("lightgbm", reason="lightgbm not installed in this environment")
        from classical.time_series.classical_baselines import LGBMBaseline
        X_tr, y_tr, X_te, _ = make_dataset()
        lb = LGBMBaseline(n_estimators=50)
        lb.fit(X_tr, y_tr)
        preds = lb.predict(X_te)
        assert len(preds) == len(X_te)

    def test_benchmark_all(self):
        from classical.time_series.classical_baselines import benchmark_all
        X_tr, y_tr, X_te, y_te = make_dataset()
        results = benchmark_all(X_tr, y_tr, X_te, y_te)
        assert "logistic" in results
        for name, metrics in results.items():
            assert "accuracy" in metrics
            assert 0.0 <= metrics["accuracy"] <= 1.0


class TestLRBaselineThreshold:
    """LR baseline must be documented at 63.8% — any model below this fails Level 1."""

    @pytest.fixture(autouse=True)
    def skip_if_no_sklearn(self):
        pytest.importorskip("sklearn", reason="scikit-learn not installed")

    def test_lr_baseline_accuracy_documented(self):
        from classical.time_series.classical_baselines import LogisticBaseline
        lb = LogisticBaseline()
        assert lb.level1_threshold == 0.638, (
            "LR baseline threshold must be 63.8% (documented Level-1 go/no-go bar). "
            "Do not change without updating the go/no-go criteria."
        )
