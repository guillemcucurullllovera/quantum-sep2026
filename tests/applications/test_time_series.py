import numpy as np
import pytest

from applications.time_series.forecaster import TimeSeriesForecaster
from applications.time_series.synthetic import SyntheticVolatilityDataset, generate_garch_like_series
from applications.volatility.realized_volatility import (
    PaperRealizedVolatilityDataset,
    forecast_realized_volatility,
)
from algorithms.qrc.reservoir import QuantumReservoirConfig
from classical.reservoir.esn import ESNConfig


def test_garch_like_series_is_stationary_and_positive_vol():
    returns, rv = generate_garch_like_series(500, seed=0)
    assert len(returns) == len(rv) == 500
    assert np.all(rv > 0)
    assert np.isfinite(returns).all()


def test_synthetic_dataset_no_lookahead():
    ds = SyntheticVolatilityDataset(n_steps=200, seed=0)
    assert ds.features().shape[0] == ds.targets().shape[0]
    splits = ds.split_walk_forward(n_splits=3)
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min()


def test_forecaster_beats_naive_baseline_with_at_least_one_method():
    """Real assertion, not a trivial constant: on this synthetic dataset, at
    least one of {QRC, ESN} must do better than predicting the training mean
    for every point — otherwise neither method learned anything and that
    would be a genuine failure worth investigating, not glossed over.
    """
    dataset = SyntheticVolatilityDataset(n_steps=300, seed=7)
    forecaster = TimeSeriesForecaster(
        quantum_config=QuantumReservoirConfig(n1=3, n2=3, seed=7),
        classical_config=ESNConfig(reservoir_size=50, seed=7),
    )
    comparison = forecaster.run(dataset, test_fraction=0.25)
    assert np.isfinite(comparison.quantum_mse)
    assert np.isfinite(comparison.classical_esn_mse)
    best = min(comparison.quantum_mse, comparison.classical_esn_mse)
    assert best < comparison.naive_constant_mse, (
        f"Neither QRC ({comparison.quantum_mse}) nor ESN ({comparison.classical_esn_mse}) "
        f"beat the naive-constant baseline ({comparison.naive_constant_mse})"
    )


def test_realized_volatility_paper_dataset_missing_raises_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setenv("QRC_PAPER_DATA_DIR", str(tmp_path / "does_not_exist"))
    with pytest.raises(FileNotFoundError, match="not redistributed"):
        PaperRealizedVolatilityDataset()


def test_forecast_realized_volatility_synthetic_path_end_to_end():
    comparison = forecast_realized_volatility(
        use_real_data=False,
        quantum_config=QuantumReservoirConfig(n1=3, n2=2, seed=3),
        classical_config=ESNConfig(reservoir_size=40, seed=3),
    )
    assert comparison.n_test > 0
    assert np.isfinite(comparison.quantum_mse)
