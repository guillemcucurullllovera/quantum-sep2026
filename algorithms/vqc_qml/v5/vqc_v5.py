"""

NOTE (Q10 audit, 2026-07-11): this 'Level 2 quantum advantage' bar is HISTORICAL. All three
primitives (PQK-v2, QRC-10, VQC-v5) were audited as CSIM (classically simulable), so beating
a classical baseline here is NOT evidence of quantum advantage — the circuit is classically
reproducible. No advantage claim is supportable. Shadow-only; never on the money path.
VQC v5 — Variational Quantum Classifier (Five Barren Plateau Fixes)

Root causes of VQC v4 failure (two independent problems):
  1. Barren plateau: Var[∂f/∂θ] ≈ 2^{-14} ≈ 6e-5. Global cost + deep ansatz.
  2. Overparameterization: γ = n_params/N_training = 28/16384 ≈ 8.5e-4.

VQC v5 — five fixes, each grounded in a theorem:

| Fix | Choice | Theorem |
|-----|--------|---------|
| 1 | 10 qubits (not 14) | McClean 2018: plateau threshold scales as n |
| 2 | σ=0.01 init (small angles) | Cerezo 2021: local minima near identity |
| 3 | Layerwise pretraining | Cerezo 2021: avoids global plateau in early training |
| 4 | d=2 re-uploading | Jerbi 2023 Th.2 + Schuld Th.5.1: richer Fourier spectrum |
| 5 | Ridge regression readout | Schuld Representer Theorem §6.5: convex output layer |

Two-level go/no-go (confirmed by Guillem Llovera, 2026-06-30):
  Level 1 (QPU exploratory): >63.8% OOS accuracy vs LR baseline, stable convergence, no leakage
  Level 2 (HISTORICAL -- WITHDRAWN, Q10: VQC-v5 is CSIM, so beating Mamba proves NO advantage)

Pure numpy simulation. Qiskit path injectable via backend_factory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .ansatz import (
    circuit_features,
    param_count,
    gradient_variance_probe,
    init_params,
)
from .layerwise_trainer import LayerwiseTrainer, ridge_fit, compute_features


@dataclass
class VQCConfig:
    n_qubits: int = 10        # Fix 1: 10 not 14
    d_layers: int = 2         # Fix 4: d=2 re-uploading
    sigma_init: float = 0.01  # Fix 2: small-angle init
    stage1_steps: int = 100
    stage2_steps: int = 200
    stage1_lr: float = 0.01
    stage2_lr: float = 0.005
    ridge_alpha: float = 0.01  # Fix 5: ridge regression readout
    batch_size: int = 16
    seed: Optional[int] = 42

    # Go/no-go thresholds
    level1_accuracy_threshold: float = 0.638  # must beat LR baseline
    grad_var_threshold: float = 1e-4           # must exceed (vs VQC v4's 6e-5)


class VQCv5:
    """
    VQC v5 classifier with all five plateau fixes.

    The architecture is defined in ansatz.py (HEA-S with re-uploading).
    Training is done by LayerwiseTrainer (Fix 3 — layerwise pretraining).
    The readout is ridge regression (Fix 5 — Representer Theorem).
    """

    def __init__(self, config: Optional[VQCConfig] = None):
        self.cfg = config or VQCConfig()
        self.trainer_: Optional[LayerwiseTrainer] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VQCv5":
        """
        Train VQC v5 on (X, y) with layerwise pretraining.

        Args:
            X: Feature matrix, shape (N, F). First n_qubits features used.
            y: Labels, shape (N,). Values in {-1, 0, 1}.
        """
        trainer = LayerwiseTrainer(
            n=self.cfg.n_qubits,
            d=self.cfg.d_layers,
            stage1_steps=self.cfg.stage1_steps,
            stage2_steps=self.cfg.stage2_steps,
            stage1_lr=self.cfg.stage1_lr,
            stage2_lr=self.cfg.stage2_lr,
            ridge_alpha=self.cfg.ridge_alpha,
            sigma_init=self.cfg.sigma_init,
            seed=self.cfg.seed,
        )
        trainer.fit(X, y, batch_size=self.cfg.batch_size)
        self.trainer_ = trainer
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Output format (matching C1.4 interface when 3 classes):
          columns = [P(sell), P(hold), P(buy)] = classes [-1, 0, 1]
          (ordered by self.trainer_.classes_, which is [-1, 0, 1] by default)

        Returns: shape (N, n_classes), rows sum to 1.
        """
        if self.trainer_ is None:
            raise RuntimeError("Call fit() first.")
        return self.trainer_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.trainer_ is None:
            raise RuntimeError("Call fit() first.")
        return self.trainer_.predict(X)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    def measure_gradient_variance(
        self, n_samples: int = 50
    ) -> float:
        """
        Measure Var[∂L/∂θ_k] for diagnostics.

        Phase 3B acceptance criterion: > 1e-4 (vs VQC v4's 6e-5).
        Uses the probing utility from ansatz.py.
        """
        return gradient_variance_probe(
            self.cfg.n_qubits, self.cfg.d_layers,
            n_samples=n_samples, seed=self.cfg.seed or 42
        )

    @property
    def level1_threshold(self) -> float:
        """Level 1 go/no-go accuracy threshold (LR baseline = 63.8%)."""
        return self.cfg.level1_accuracy_threshold

    @property
    def n_params(self) -> int:
        """Total number of trainable circuit parameters."""
        return param_count(self.cfg.n_qubits, self.cfg.d_layers)

    def level1_check(self, X_oos: np.ndarray, y_oos: np.ndarray) -> dict:
        """
        Run Level 1 go/no-go gate.

        Returns dict with:
          - accuracy: OOS accuracy
          - passes: bool (accuracy > 63.8%)
          - gap: accuracy - threshold
        """
        acc = self.accuracy(X_oos, y_oos)
        return {
            "accuracy": acc,
            "threshold": self.level1_threshold,
            "passes": acc > self.level1_threshold,
            "gap": acc - self.level1_threshold,
        }

    def circuit_summary(self) -> dict:
        """Return circuit configuration for IBM meeting materials."""
        return {
            "n_qubits": self.cfg.n_qubits,
            "d_layers": self.cfg.d_layers,
            "n_params": self.n_params,
            "architecture": "HEA-S with RY+RZ+CNOT_ring, d=2 re-uploading",
            "encoding": "angle_encoding_RY",
            "readout": "ridge_regression (Schuld Representer Theorem §6.5)",
            "plateau_fixes": [
                "Fix 1: 10 qubits (McClean 2018 threshold)",
                "Fix 2: σ=0.01 init (Cerezo 2021 near-identity)",
                "Fix 3: Layerwise pretraining (Cerezo 2021)",
                "Fix 4: d=2 re-uploading (Jerbi 2023, Schuld 5.1)",
                "Fix 5: Ridge regression readout (Schuld §6.5)",
            ],
        }


# Backwards-compatible alias — benchmark scripts may use the old name
VQCv5Config = VQCConfig
