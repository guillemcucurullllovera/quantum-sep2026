"""Time-series forecasting application composing algorithms.qrc and
classical.reservoir as swappable, directly-comparable backends.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from algorithms.qrc.encoders import encode_sequence
from algorithms.qrc.reservoir import QuantumReservoir, QuantumReservoirConfig
from algorithms.qrc.reservoir_math import ridge_fit, ridge_predict
from classical.reservoir.esn import ESNConfig, EchoStateNetwork
from core.dataset_interfaces import FinancialDataset
from core.errors import ValidationError
from core.metrics import mae, mse
from core.registry import APPLICATION_REGISTRY


@dataclass
class ForecastComparison:
    quantum_mse: float
    quantum_mae: float
    classical_esn_mse: float
    classical_esn_mae: float
    naive_constant_mse: float
    n_test: int


def _naive_constant_forecast(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """The weakest sane baseline: predict the training-set mean for every
    test point. Any method that can't beat this has learned nothing.
    """
    pred = np.full_like(y_test, fill_value=y_train.mean())
    return mse(y_test, pred)


@APPLICATION_REGISTRY.register("time_series_forecaster")
class TimeSeriesForecaster:
    """Compares QRC against an ESN baseline and a naive-constant floor on one
    walk-forward split of a FinancialDataset.
    """

    def __init__(
        self,
        quantum_config: QuantumReservoirConfig | None = None,
        classical_config: ESNConfig | None = None,
        ridge_delta: float = 1e-3,
    ) -> None:
        self.quantum_config = quantum_config or QuantumReservoirConfig()
        self.classical_config = classical_config or ESNConfig()
        self.ridge_delta = ridge_delta

    def run(self, dataset: FinancialDataset, test_fraction: float = 0.25) -> ForecastComparison:
        X, y = dataset.features(), dataset.targets()
        n = len(y)
        n_test = int(n * test_fraction)
        if n_test < 2:
            raise ValidationError(f"Dataset too small ({n} rows) for test_fraction={test_fraction}")
        train_idx = np.arange(0, n - n_test)
        test_idx = np.arange(n - n_test, n)

        # -- quantum reservoir --
        angles = encode_sequence(X, n1=self.quantum_config.n1)
        reservoir = QuantumReservoir(self.quantum_config).build()
        q_features = reservoir.sequence_features(angles, washout=0)
        q_train_idx = train_idx[train_idx < len(q_features)]
        q_test_idx = test_idx[test_idx < len(q_features)]
        M_train, M_test = q_features[q_train_idx].T, q_features[q_test_idx].T
        W = ridge_fit(M_train, y[q_train_idx], delta=self.ridge_delta, bias=True)
        q_pred = ridge_predict(W, M_test, bias=True)

        # -- classical ESN baseline, identical train/test split --
        esn = EchoStateNetwork(self.classical_config).build(n_inputs=X.shape[1])
        states = esn.compute_states(X)
        esn.fit_readout(states[train_idx], y[train_idx])
        esn_pred = esn.predict(states[test_idx])

        naive_mse = _naive_constant_forecast(y[train_idx], y[test_idx])

        return ForecastComparison(
            quantum_mse=mse(y[q_test_idx], q_pred),
            quantum_mae=mae(y[q_test_idx], q_pred),
            classical_esn_mse=mse(y[test_idx], esn_pred),
            classical_esn_mae=mae(y[test_idx], esn_pred),
            naive_constant_mse=naive_mse,
            n_test=n_test,
        )


__all__ = ["TimeSeriesForecaster", "ForecastComparison"]
