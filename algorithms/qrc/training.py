"""Walk-forward validation and hyperparameter sweeps for QRC, logged through
core.experiment_registry so sweep history is queryable later instead of
scattered across ad hoc result files.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import asdict

import numpy as np

from algorithms.qrc.encoders import encode_sequence
from algorithms.qrc.reservoir import QuantumReservoir, QuantumReservoirConfig
from algorithms.qrc.reservoir_math import ridge_fit, ridge_predict
from core.dataset_interfaces import FinancialDataset
from core.experiment_registry import ExperimentRegistry
from core.metrics import mae, mse
from core.schemas import BackendDescriptor, BackendKind, ExperimentConfig, ExperimentResult, RunMetadata


def walk_forward_evaluate(
    dataset: FinancialDataset,
    config: QuantumReservoirConfig,
    n_splits: int = 4,
    ridge_delta: float = 1e-3,
    min_train_fraction: float = 0.5,
) -> ExperimentResult:
    """Evaluate a QuantumReservoir config with walk-forward CV, no fold ever
    training on data that postdates its test fold (see
    core.dataset_interfaces.split_walk_forward for the invariant).
    """
    X = dataset.features()
    y = dataset.targets()
    angles = encode_sequence(X, n1=config.n1)

    fold_mses, fold_maes = [], []
    t0 = time.time()
    for train_idx, test_idx in dataset.split_walk_forward(n_splits, min_train_fraction):
        reservoir = QuantumReservoir(config).build()
        features = reservoir.sequence_features(angles, washout=0)

        # features[t] corresponds to y[t] (persistent-state, same length as input)
        train_idx = train_idx[train_idx < len(features)]
        test_idx = test_idx[test_idx < len(features)]
        if len(train_idx) < 5 or len(test_idx) < 1:
            continue

        M_train = features[train_idx].T
        M_test = features[test_idx].T
        W = ridge_fit(M_train, y[train_idx], delta=ridge_delta, bias=True)
        y_pred = ridge_predict(W, M_test, bias=True)

        fold_mses.append(mse(y[test_idx], y_pred))
        fold_maes.append(mae(y[test_idx], y_pred))

    wall_time = time.time() - t0
    backend = BackendDescriptor(name="numpy_statevector", kind=BackendKind.SIMULATOR, num_qubits=config.n)
    exp_config = ExperimentConfig(
        name="qrc_walk_forward",
        algorithm="qrc",
        application="time_series",
        backend=backend,
        seed=config.seed,
        params={**asdict(config), "n_splits": n_splits, "ridge_delta": ridge_delta},
    )
    metadata = RunMetadata(algorithm="qrc", generation="gen2", seed=config.seed)
    result = ExperimentResult(
        metadata=metadata,
        config=exp_config,
        metrics={
            "mse_mean": float(np.mean(fold_mses)) if fold_mses else float("nan"),
            "mse_std": float(np.std(fold_mses)) if fold_mses else float("nan"),
            "mae_mean": float(np.mean(fold_maes)) if fold_maes else float("nan"),
            "n_folds_evaluated": float(len(fold_mses)),
            "wall_time_s": wall_time,
        },
    )
    ExperimentRegistry().record(result)
    return result


def hyperparameter_sweep(
    dataset: FinancialDataset,
    n1_values: list[int],
    n2_values: list[int],
    alpha_values: list[float],
    n_splits: int = 3,
    seed: int = 42,
) -> list[ExperimentResult]:
    """Grid sweep over reservoir hyperparameters; every configuration is
    logged as its own experiment. Returns results sorted by mean MSE
    (ascending), best first.
    """
    results = []
    for n1, n2, alpha in itertools.product(n1_values, n2_values, alpha_values):
        config = QuantumReservoirConfig(n1=n1, n2=n2, alpha=alpha, seed=seed)
        result = walk_forward_evaluate(dataset, config, n_splits=n_splits)
        results.append(result)
    return sorted(results, key=lambda r: r.metrics.get("mse_mean", float("inf")))


__all__ = ["walk_forward_evaluate", "hyperparameter_sweep"]
