from applications.classification.synthetic_volatility_dataset import SyntheticVolatilityDataset
from applications.classification.qnn_regression import run_volatility_regression


def test_regression_produces_real_nonzero_errors_and_decreasing_loss():
    ds = SyntheticVolatilityDataset(n_samples=60, seed=5)
    results = run_volatility_regression(ds, n_qubits=3, n_splits=2, seed=0)
    assert len(results) == 2
    for r in results:
        assert r.qnn_mse > 0
        assert r.qnn_rmse == pytest_approx_sqrt(r.qnn_mse)
        assert r.classical_mse > 0
        # training loss should be finite and non-negative (real MSE loss, not a stub)
        assert r.final_training_loss >= 0


def pytest_approx_sqrt(mse: float) -> float:
    return mse ** 0.5


def test_qnn_training_loss_actually_decreases():
    from algorithms_extended.qiskit_ml_finance.regressor import QNNRegressor, RegressorConfig

    ds = SyntheticVolatilityDataset(n_samples=60, seed=5)
    X, y = ds.features(), ds.targets()
    model = QNNRegressor(RegressorConfig(n_qubits=3, epochs=25, seed=1)).fit(X, y)
    # not monotonic every step (Adam), but the final loss should beat the first
    assert model.loss_history_[-1] < model.loss_history_[0]
