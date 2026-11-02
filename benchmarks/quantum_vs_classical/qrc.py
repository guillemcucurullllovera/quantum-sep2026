"""QRC vs classical ESN, on an identical synthetic time series."""
from __future__ import annotations

import numpy as np

from algorithms.qrc.reservoir import QuantumReservoir, QuantumReservoirConfig
from algorithms.qrc.encoders import minmax_encode_sequence
from classical.reservoir.esn import ESNConfig, EchoStateNetwork
from core.metrics import mse
from benchmarks.runner import MethodOutput, run_benchmark


def _synthetic_series(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    signal = 0.5 * np.sin(2 * np.pi * t / 40) + 0.2 * np.sin(2 * np.pi * t / 13)
    noise = rng.normal(0, 0.05, size=n)
    return signal + noise


def _sliding_windows(series: np.ndarray, n_samples: int, n_features: int):
    """(n_samples, n_features) feature matrix + (n_samples,) next-step targets,
    windows fully in-bounds by construction: series must have length
    >= n_samples + n_features."""
    X = np.array([series[i : i + n_features] for i in range(n_samples)])
    y = series[n_features : n_features + n_samples]
    return X, y


def qrc_vs_esn_forecast(n_samples: int = 300, n_features: int = 4, seed: int = 0):
    series = _synthetic_series(n_samples + n_features + 1, seed)
    X, y = _sliding_windows(series, n_samples, n_features)
    split = int(n_samples * 0.7)
    n_test = min(50, n_samples - split)

    def run_qrc() -> MethodOutput:
        cfg = QuantumReservoirConfig(n1=n_features, n2=4, seed=seed)
        reservoir = QuantumReservoir(cfg).build()
        X_enc = minmax_encode_sequence(X, n_features)
        feats = reservoir.sequence_features(X_enc)
        n = min(len(feats), split)
        train_X, train_y = feats[:n], y[:n]
        # ridge-regression readout, kept local and simple: same idea as
        # algorithms.vqc_qml.training.RidgeReadout but this benchmark stays
        # self-contained rather than importing across algorithm families
        w, *_ = np.linalg.lstsq(train_X, train_y, rcond=None)
        test_X = feats[n : n + n_test]
        test_y = y[n : n + n_test]
        if len(test_X) == 0:
            return MethodOutput(value=float("nan"))
        pred = test_X @ w
        return MethodOutput(value=mse(test_y, pred))

    def run_esn() -> MethodOutput:
        esn = EchoStateNetwork(ESNConfig(reservoir_size=30, seed=seed)).build(n_inputs=n_features)
        states = esn.compute_states(X)
        esn.fit_readout(states[:split], y[:split])
        pred = esn.predict(states[split : split + n_test])
        return MethodOutput(value=mse(y[split : split + n_test], pred))

    return run_benchmark(
        method_a=run_qrc,
        method_b=run_esn,
        method_a_name="qrc",
        method_b_name="echo_state_network",
        metric_name="forecast_mse",
        seed=seed,
        context={"n_samples": n_samples, "n_features": n_features},
    )
