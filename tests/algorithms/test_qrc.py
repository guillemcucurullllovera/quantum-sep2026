import numpy as np

from algorithms.qrc.encoders import encode_inputs, encode_sequence, triple_band_signal
from algorithms.qrc.reservoir import QuantumReservoir, QuantumReservoirConfig
from algorithms.qrc.training import hyperparameter_sweep, walk_forward_evaluate
from applications.time_series.synthetic import SyntheticVolatilityDataset
from core.experiment_registry import ExperimentRegistry
from core.metrics import mse


def test_encode_inputs_bounded_and_sign_preserving():
    x = np.array([5.0, -5.0, 0.0])
    angles = encode_inputs(x, n1=3)
    assert np.all(np.abs(angles) <= np.pi)
    assert angles[0] > 0 and angles[1] < 0 and angles[2] == 0.0


def test_encode_sequence_shape():
    X = np.random.default_rng(0).standard_normal((10, 5))
    angles = encode_sequence(X, n1=3)
    assert angles.shape == (10, 3)


def test_triple_band_signal_sums_to_one():
    z = np.array([0.5, 0.3, -0.2, 0.1, -0.4, 0.6])
    sig = triple_band_signal(z)
    total = sig["p_buy"] + sig["p_sell"] + sig["p_hold"]
    assert np.isclose(total, 1.0)
    assert sig["confidence"] == max(sig["p_buy"], sig["p_sell"], sig["p_hold"])


def test_reservoir_build_and_windowed_features_shape():
    config = QuantumReservoirConfig(n1=3, n2=2, seed=0)
    reservoir = QuantumReservoir(config).build()
    windows = np.random.default_rng(0).uniform(-np.pi, np.pi, size=(5, 3, 3))
    features = reservoir.windowed_features(windows)
    assert features.shape == (5, config.n)


def test_reservoir_sequence_features_respects_washout():
    config = QuantumReservoirConfig(n1=2, n2=2, seed=1)
    reservoir = QuantumReservoir(config).build()
    X = np.random.default_rng(1).uniform(-np.pi, np.pi, size=(20, 2))
    features = reservoir.sequence_features(X, washout=5)
    assert features.shape == (15, config.n)


def test_walk_forward_evaluate_produces_finite_metrics_and_logs_to_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUM_SEP2026_REGISTRY_DB", str(tmp_path / "reg.sqlite3"))
    from core.config import get_settings

    get_settings(refresh=True)
    dataset = SyntheticVolatilityDataset(n_steps=120, seed=0)
    config = QuantumReservoirConfig(n1=3, n2=2, seed=0)
    result = walk_forward_evaluate(dataset, config, n_splits=2)
    assert np.isfinite(result.metrics["mse_mean"])
    assert result.metrics["n_folds_evaluated"] >= 1

    registry = ExperimentRegistry(db_path=tmp_path / "reg.sqlite3")
    fetched = registry.get(result.metadata.run_id)
    assert fetched.metadata.algorithm == "qrc"


def test_hyperparameter_sweep_returns_sorted_by_mse(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUM_SEP2026_REGISTRY_DB", str(tmp_path / "reg2.sqlite3"))
    from core.config import get_settings

    get_settings(refresh=True)
    dataset = SyntheticVolatilityDataset(n_steps=100, seed=1)
    results = hyperparameter_sweep(
        dataset, n1_values=[2, 3], n2_values=[1, 2], alpha_values=[0.9], n_splits=2, seed=0
    )
    assert len(results) == 4
    mses = [r.metrics["mse_mean"] for r in results if np.isfinite(r.metrics["mse_mean"])]
    assert mses == sorted(mses)
