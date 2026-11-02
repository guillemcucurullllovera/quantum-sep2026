"""Kernel-based financial classification application: composes the PQK
algorithm suite (Gen1 fidelity/projected + Gen2 PQKv2) and the classical
kernel baselines into a single walk-forward-validated pipeline over a
synthetic directional-classification dataset.

Real synthetic data generator (not a fixture stub): an 8-feature regime
process with a controllable signal-to-noise ratio, so the classifier
comparison is meaningful (a dataset with zero signal makes every method look
equally useless; a noiseless dataset makes every method look equally good --
neither tells you anything about the quantum method specifically).
"""
from __future__ import annotations

import numpy as np

from algorithms.pqk.pqk_v2 import PQKConfig, PQKv2
from classical.kernels.baselines import train_classical_svm
from core.dataset_interfaces import FinancialDataset
from core.registry import APPLICATION_REGISTRY
from core.schemas import DatasetDescriptor
from core.seeds import seed_everything


class SyntheticRegimeDataset(FinancialDataset):
    """8-feature synthetic 'market regime' classification dataset.

    Features are a mix of a genuine (but nonlinear, kernel-favoring) signal
    -- a smooth function of two latent regime variables -- plus independent
    Gaussian noise on the remaining feature dimensions, scaled to
    [0, 1]-ish ranges the platform's angle-encoding feature maps expect
    (they multiply by pi internally).
    """

    def __init__(self, n_samples: int = 200, noise_level: float = 0.3, seed: int = 0) -> None:
        seed_everything(seed)
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 8 * np.pi, n_samples)
        regime_a = np.sin(t) + 0.1 * rng.normal(size=n_samples)
        regime_b = np.cos(t * 0.7) + 0.1 * rng.normal(size=n_samples)

        X = np.zeros((n_samples, 8))
        X[:, 0] = regime_a
        X[:, 1] = regime_b
        X[:, 2] = regime_a * regime_b  # nonlinear interaction term
        X[:, 3:] = noise_level * rng.normal(size=(n_samples, 5))
        # squash to roughly [0, 1] so angle encoding (features * pi) is well-scaled
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-9)

        y = (regime_a * regime_b > 0).astype(int)  # nonlinear XOR-like boundary

        self._X, self._y = X, y

    @property
    def descriptor(self) -> DatasetDescriptor:
        return DatasetDescriptor(
            name="synthetic_regime_v1",
            n_samples=len(self._y),
            n_features=self._X.shape[1],
            domain="directional_classification",
        )

    def features(self) -> np.ndarray:
        return self._X

    def targets(self) -> np.ndarray:
        return self._y


def run_kernel_classification_comparison(
    n_samples: int = 200, noise_level: float = 0.3, n_splits: int = 3, seed: int = 0
) -> dict:
    """Walk-forward comparison: PQKv2 (Gen2, KTA-aligned) vs. classical RBF-
    kernel SVM, on the same synthetic regime dataset and the same splits.

    Returns per-fold and averaged accuracy for both, plus PQKv2's KTA
    diagnostic (whether it cleared the 0.3 acceptance bar on this data) --
    reported honestly even if the quantum method doesn't win, consistent
    with this platform's documented no-overclaiming policy.
    """
    ds = SyntheticRegimeDataset(n_samples=n_samples, noise_level=noise_level, seed=seed)
    X, y = ds.features(), ds.targets()
    splits = ds.split_walk_forward(n_splits=n_splits, min_train_fraction=0.5)

    fold_results = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        pqk_model, pqk_result = _fit_pqk(X_tr, y_tr, seed)
        pqk_val_acc = pqk_model.accuracy(X_te, y_te)

        classical = train_classical_svm(X_tr, y_tr, X_te, y_te, kernel="rbf", gamma=1.0)

        fold_results.append(
            {
                "fold": fold_i,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "pqk_v2_val_accuracy": pqk_val_acc,
                "pqk_v2_final_kta": pqk_model.final_kta_,
                "pqk_v2_level1_passed": pqk_model.level1_passed,
                "classical_rbf_val_accuracy": classical["val_accuracy"],
            }
        )

    avg_pqk = float(np.mean([r["pqk_v2_val_accuracy"] for r in fold_results]))
    avg_classical = float(np.mean([r["classical_rbf_val_accuracy"] for r in fold_results]))

    return {
        "dataset": ds.descriptor.model_dump(),
        "n_splits": n_splits,
        "folds": fold_results,
        "avg_pqk_v2_accuracy": avg_pqk,
        "avg_classical_rbf_accuracy": avg_classical,
        "pqk_beats_classical": avg_pqk > avg_classical,
    }


def _fit_pqk(X_tr: np.ndarray, y_tr: np.ndarray, seed: int) -> tuple[PQKv2, dict]:
    # n_qubits=4 (not the full 8 feature dims) and a modest kta_steps budget:
    # KTA gradient ascent is O(kta_steps * n_qubits * m_kta^2) statevector
    # evaluations (finite-difference gradient over n_qubits params, each
    # requiring 2 extra Gram-matrix passes) -- this keeps a walk-forward
    # sweep tractable without changing the algorithm itself.
    cfg = PQKConfig(n_qubits=4, seed=seed, kta_steps=20, nystrom_m=min(30, len(X_tr)))
    model = PQKv2(cfg).fit(X_tr, y_tr)
    return model, {"final_kta": model.final_kta_}


APPLICATION_REGISTRY.register("kernel_classification")(run_kernel_classification_comparison)
