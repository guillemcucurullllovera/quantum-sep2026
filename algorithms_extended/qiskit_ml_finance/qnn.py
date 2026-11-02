"""FinancialQNN: an EstimatorQNN-pattern wrapper, adapted architecturally from
qiskit-machine-learning's `neural_networks/estimator_qnn.py` (Apache-2.0) --
no source copied, see THIRD_PARTY_NOTICES.md.

The key architectural difference from `algorithms.vqc_qml.training.VQCTrainer`: that trainer treats the circuit's Z-expectation features as fixed
once computed for a candidate parameter vector, and solves the readout layer
in closed form (ridge regression) inside a gradient-free outer loop (COBYLA/
SPSA) over the circuit parameters. This module instead separates circuit
("weight") parameters from a *trainable* linear readout head and computes
exact gradients for BOTH end-to-end via the chain rule, using
`VQCAnsatz.parameter_shift_feature_grad` (already already implemented --
reused here, not reimplemented) for the circuit-parameter half of the chain
rule. This mirrors EstimatorQNN's `input_params`/`weight_params` separation
and `_forward`/`_backward` pair, letting weights and circuit parameters be
optimized jointly by a single gradient-based optimizer (see optimizers.py)
instead of a two-level closed-form/gradient-free split.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from algorithms.vqc_qml.ansatz import VQCAnsatz
from core.errors import ValidationError


@dataclass
class QNNForwardResult:
    output: np.ndarray  # (n_samples, n_outputs)
    features: np.ndarray  # (n_samples, n_qubits) -- Z-expectations, cached for backward()


@dataclass
class FinancialQNN:
    """Linear-readout QNN: output = Z(x; theta) @ W + b.

    `theta` (circuit parameters) and `(W, b)` (readout weights) are both
    trainable and both receive exact analytic gradients from `backward()`.
    """

    ansatz: VQCAnsatz
    n_outputs: int = 1

    @property
    def n_qubits(self) -> int:
        return self.ansatz.n_qubits

    @property
    def n_circuit_params(self) -> int:
        return self.ansatz.param_count

    @property
    def n_readout_params(self) -> int:
        return self.n_qubits * self.n_outputs + self.n_outputs  # W + b

    def init_params(self, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta = self.ansatz.init_params(seed=seed)
        rng = np.random.default_rng(seed + 1)
        W = rng.normal(0, 1.0 / np.sqrt(self.n_qubits), size=(self.n_qubits, self.n_outputs))
        b = np.zeros(self.n_outputs)
        return theta, W, b

    def forward(
        self, X_encoded: np.ndarray, theta: np.ndarray, W: np.ndarray, b: np.ndarray
    ) -> QNNForwardResult:
        if X_encoded.shape[1] != self.n_qubits:
            raise ValidationError(
                f"X_encoded has {X_encoded.shape[1]} features, ansatz expects {self.n_qubits} qubits"
            )
        Z = np.array([self.ansatz.z_expectations(x, theta) for x in X_encoded])  # (n, n_qubits)
        out = Z @ W + b  # (n, n_outputs)
        return QNNForwardResult(output=out, features=Z)

    def backward(
        self,
        X_encoded: np.ndarray,
        theta: np.ndarray,
        W: np.ndarray,
        forward: QNNForwardResult,
        grad_output: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Chain rule from dLoss/d(output) back to dLoss/d(theta), dLoss/dW, dLoss/db.

        grad_output: (n_samples, n_outputs), i.e. dLoss/d(forward.output).
        Returns (grad_theta, grad_W, grad_b).
        """
        Z = forward.features  # (n, n_qubits)
        n = X_encoded.shape[0]

        grad_W = Z.T @ grad_output  # (n_qubits, n_outputs)
        grad_b = grad_output.sum(axis=0)  # (n_outputs,)

        # dLoss/dZ_i = grad_output_i @ W.T   -> (n, n_qubits)
        grad_Z = grad_output @ W.T

        # dLoss/dtheta = sum_i grad_Z_i . dZ_i/dtheta   (chain rule through the
        # circuit -- dZ_i/dtheta is the analytic parameter-shift gradient
        # already implemented by VQCAnsatz, reused verbatim here)
        grad_theta = np.zeros(self.n_circuit_params)
        for i in range(n):
            dZ_dtheta = self.ansatz.parameter_shift_feature_grad(X_encoded[i], theta)  # (n_params, n_qubits)
            grad_theta += dZ_dtheta @ grad_Z[i]

        return grad_theta, grad_W, grad_b
