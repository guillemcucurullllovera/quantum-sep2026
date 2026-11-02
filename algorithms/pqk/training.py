"""Training/evaluation pipelines for the PQK suite, wired to the platform's
experiment registry so sweep results are queryable later instead of living
only in a notebook's output cell.
"""
from __future__ import annotations

import time

import numpy as np

from algorithms.pqk import kernels, kernels_gen1
from algorithms.pqk.pqk_v2 import PQKConfig, PQKv2
from core.experiment_registry import ExperimentRegistry
from core.registry import ALGORITHM_REGISTRY
from core.schemas import (
    BackendDescriptor,
    BackendKind,
    ExperimentConfig,
    ExperimentResult,
    RunMetadata,
)
from core.seeds import seed_everything


def train_gen1_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_map_name: str = "AngleCNOT",
    method: str = "projected",
    gamma: float = 1.0,
    C: float = 1.0,
    seed: int = 42,
    registry: ExperimentRegistry | None = None,
) -> ExperimentResult:
    """Gen1-style pipeline: quantum kernel -> SimpleSVM, evaluated on a held-
    out validation split, logged as an ExperimentResult."""
    seed_everything(seed)
    y_tr = 2 * y_train.astype(float) - 1
    y_va = 2 * y_val.astype(float) - 1

    t0 = time.time()
    if method == "fidelity":
        K_train = kernels.fidelity_gram(X_train, feature_map_name)
        K_val = kernels_gen1.compute_fidelity_gram(X_val, _fmap(feature_map_name), X_train)
    else:
        K_train = kernels.projected_gram(X_train, feature_map_name, gamma=gamma)
        K_val, _, _ = kernels_gen1.compute_projected_gram(X_val, _fmap(feature_map_name), X_train, gamma=gamma)
    t_gram = time.time() - t0

    svm = kernels_gen1.SimpleSVM(C=C)
    n_sv = svm.fit(K_train, y_tr)
    train_acc = svm.accuracy(K_train, y_tr)
    val_acc = svm.accuracy(K_val, y_va)

    backend = BackendDescriptor(name="statevector_simulator", kind=BackendKind.SIMULATOR, num_qubits=8)
    config = ExperimentConfig(
        name=f"pqk_gen1_{method}_{feature_map_name}",
        algorithm="pqk_gen1",
        backend=backend,
        seed=seed,
        params={"feature_map": feature_map_name, "method": method, "gamma": gamma, "C": C},
    )
    meta = RunMetadata(algorithm="pqk_gen1", generation="gen1", seed=seed)
    result = ExperimentResult(
        metadata=meta,
        config=config,
        metrics={
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "n_support_vectors": float(n_sv),
            "gram_compute_seconds": t_gram,
        },
    )
    (registry or ExperimentRegistry()).record(result)
    return result


def train_pqk_v2(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: PQKConfig | None = None,
    registry: ExperimentRegistry | None = None,
) -> tuple[PQKv2, ExperimentResult]:
    """Gen2 pipeline: KTA-aligned AngleCNOTRingV2 -> Nystrom -> ridge readout."""
    cfg = config or PQKConfig()
    seed_everything(cfg.seed or 0)
    model = PQKv2(cfg).fit(X_train, y_train)

    train_acc = model.accuracy(X_train, y_train)
    val_acc = model.accuracy(X_val, y_val)

    backend = BackendDescriptor(name="statevector_simulator", kind=BackendKind.SIMULATOR, num_qubits=cfg.n_qubits)
    exp_config = ExperimentConfig(
        name="pqk_v2",
        algorithm="pqk_v2",
        backend=backend,
        seed=cfg.seed or 0,
        params={"nystrom_m": cfg.nystrom_m, "kta_target": cfg.kta_target},
    )
    meta = RunMetadata(algorithm="pqk_v2", generation="gen2", seed=cfg.seed)
    result = ExperimentResult(
        metadata=meta,
        config=exp_config,
        metrics={
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "final_kta": model.final_kta_ or 0.0,
            "level1_passed": float(model.level1_passed),
        },
        notes=f"KTA history length={len(model.kta_history_)}",
    )
    (registry or ExperimentRegistry()).record(result)
    return model, result


def sweep_feature_maps(
    X: np.ndarray, y: np.ndarray, method: str = "projected", gammas: tuple[float, ...] = (0.5, 1.0, 2.0)
) -> list[dict]:
    """Score every registered feature map (and, for 'projected', a small
    gamma grid) via KTA/concentration diagnostics -- a cheap pre-training
    screen, not a full train+eval sweep. Returns a flat list of dict rows,
    sortable by 'kta' to pick a starting point before the expensive KTA
    gradient-ascent + Nystrom pipeline."""
    rows = []
    for name in kernels.FEATURE_MAPS:
        if method == "fidelity":
            diag = kernels.score_feature_map(X, y, name, method="fidelity")
            rows.append({"feature_map": name, "gamma": None, **diag})
        else:
            for g in gammas:
                diag = kernels.score_feature_map(X, y, name, method="projected", gamma=g)
                rows.append({"feature_map": name, "gamma": g, **diag})
    return sorted(rows, key=lambda r: r["kta"], reverse=True)


def _fmap(name: str):
    return kernels.FEATURE_MAPS[name]()


ALGORITHM_REGISTRY.register("pqk_gen1")(train_gen1_svm)
ALGORITHM_REGISTRY.register("pqk_v2")(train_pqk_v2)
