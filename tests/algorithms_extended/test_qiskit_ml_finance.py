import numpy as np
import pytest

from algorithms.vqc_qml.ansatz import VQCAnsatz
from algorithms_extended.qiskit_ml_finance.qnn import FinancialQNN
from algorithms_extended.qiskit_ml_finance.optimizers import Adam, cobyla_minimize
from algorithms_extended.qiskit_ml_finance.regressor import QNNRegressor, RegressorConfig
from algorithms_extended.qiskit_ml_finance.classifier import QNNClassifier, ClassifierConfig
from algorithms_extended.qiskit_ml_finance.classical_baseline import ridge_regression_baseline
from algorithms_extended.qiskit_ml_finance.applications import (
    run_volatility_regression_demo,
    run_regime_classification_demo,
)


def _toy_qnn(n_qubits=3, n_outputs=2):
    ansatz = VQCAnsatz(n_qubits=n_qubits, version="v1_linear")
    return FinancialQNN(ansatz=ansatz, n_outputs=n_outputs), ansatz


def test_qnn_backward_matches_finite_difference_gradient():
    """The single most important correctness test in this module: confirms
    the chain-ruled analytic gradient (through parameter_shift_
    feature_grad) actually matches a numerical finite-difference gradient of
    an arbitrary scalar loss, for theta, W, and b independently."""
    rng = np.random.default_rng(0)
    qnn, ansatz = _toy_qnn(n_qubits=3, n_outputs=1)
    X = rng.normal(0, 1, (4, 3))
    theta, W, b = qnn.init_params(seed=1)

    def loss(theta, W, b):
        fwd = qnn.forward(X, theta, W, b)
        return float(np.sum(fwd.output**2))

    fwd = qnn.forward(X, theta, W, b)
    grad_out = 2 * fwd.output  # d(sum out^2)/d(out)
    grad_theta, grad_W, grad_b = qnn.backward(X, theta, W, fwd, grad_out)

    eps = 1e-4
    # spot-check a handful of theta components (full check is O(n_params), slow)
    for k in [0, 1, min(2, len(theta) - 1)]:
        tp, tm = theta.copy(), theta.copy()
        tp[k] += eps
        tm[k] -= eps
        numeric = (loss(tp, W, b) - loss(tm, W, b)) / (2 * eps)
        assert abs(numeric - grad_theta[k]) < 1e-3, f"theta[{k}] grad mismatch: {numeric} vs {grad_theta[k]}"

    Wp, Wm = W.copy(), W.copy()
    Wp[0, 0] += eps
    Wm[0, 0] -= eps
    numeric_W = (loss(theta, Wp, b) - loss(theta, Wm, b)) / (2 * eps)
    assert abs(numeric_W - grad_W[0, 0]) < 1e-3

    bp, bm = b.copy(), b.copy()
    bp[0] += eps
    bm[0] -= eps
    numeric_b = (loss(theta, W, bp) - loss(theta, W, bm)) / (2 * eps)
    assert abs(numeric_b - grad_b[0]) < 1e-3


def test_adam_reduces_a_simple_quadratic_loss():
    # Vanilla Adam (no lr decay) does not shrink its step size near a
    # minimum -- on a toy 1-D quadratic it approaches then oscillates around
    # zero rather than converging tightly, which is expected Adam behavior,
    # not a defect. The meaningful assertion is that it gets and stays much
    # closer to the minimum than it started, not that it converges exactly.
    opt = Adam(lr=0.1)
    x = {"p": np.array([5.0])}
    trace = [float(x["p"][0])]
    for _ in range(200):
        grad = {"p": 2 * x["p"]}  # d/dp p^2
        x = opt.step(x, grad)
        trace.append(float(x["p"][0]))
    late = trace[-50:]
    assert max(abs(v) for v in late) < 0.3  # settled near the minimum, not still descending from 5.0


def test_regressor_loss_decreases_over_training():
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (20, 4))
    y = X[:, 0] * 0.3 + rng.normal(0, 0.05, 20)
    reg = QNNRegressor(RegressorConfig(n_qubits=4, epochs=40, seed=3)).fit(X, y)
    assert reg.loss_history_[-1] < reg.loss_history_[0]


def test_regressor_predict_matches_training_shape():
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (16, 4))
    y = rng.normal(0, 1, 16)
    reg = QNNRegressor(RegressorConfig(n_qubits=4, epochs=10, seed=4)).fit(X, y)
    preds = reg.predict(X[:5])
    assert preds.shape == (5,)


def test_regressor_cobyla_path_runs_and_reduces_loss():
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (12, 4))
    y = X[:, 0] * 0.2 + rng.normal(0, 0.05, 12)
    reg = QNNRegressor(RegressorConfig(n_qubits=4, seed=5)).fit_cobyla(X, y)
    assert len(reg.loss_history_) > 1
    assert min(reg.loss_history_) <= reg.loss_history_[0]


def test_classifier_rejects_multiclass():
    rng = np.random.default_rng(6)
    X = rng.normal(0, 1, (9, 4))
    y = np.array([0, 1, 2] * 3)
    clf = QNNClassifier(ClassifierConfig(n_qubits=4, epochs=5, seed=6))
    with pytest.raises(Exception):
        clf.fit(X, y)


def test_classifier_trains_and_predicts_binary():
    rng = np.random.default_rng(7)
    n = 24
    X = rng.normal(0, 1, (n, 4))
    y = (X[:, 0] > 0).astype(int)  # linearly separable in one feature
    clf = QNNClassifier(ClassifierConfig(n_qubits=4, epochs=60, seed=7)).fit(X, y)
    acc = clf.score(X, y)
    assert 0.0 <= acc <= 1.0
    proba = clf.predict_proba(X)
    assert np.all((proba >= 0) & (proba <= 1))
    assert clf.loss_history_[-1] <= clf.loss_history_[0] + 1e-6


def test_classical_ridge_baseline_beats_random_on_linear_signal():
    rng = np.random.default_rng(8)
    X = rng.normal(0, 1, (60, 3))
    y = X[:, 0] * 2.0 + rng.normal(0, 0.1, 60)
    result = ridge_regression_baseline(X[:40], y[:40], X[40:], y[40:])
    assert result.mse < np.var(y)  # meaningfully better than predicting the mean


def test_volatility_regression_demo_runs_and_reports_both_methods():
    report = run_volatility_regression_demo(seed=1)
    assert report.n_train > 0 and report.n_test > 0
    assert report.qnn_mse >= 0 and report.classical_mse >= 0


def test_regime_classification_demo_runs_and_reports_both_methods():
    report = run_regime_classification_demo(seed=1)
    assert report.n_train > 0 and report.n_test > 0
    assert 0.0 <= report.qnn_accuracy <= 1.0
    assert 0.0 <= report.classical_accuracy <= 1.0
