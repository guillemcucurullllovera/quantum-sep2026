"""
phase0.quantum_qiskit.circuit_v4_qiskit
=======================================

Qiskit twin of :class:`phase0.quantum_signal.circuit_v4.TrainableVQC_v4`.

This is Step 2 of EXECUTION_MASTER_PLAN.md. The module mirrors the
14-qubit, 2-layer hardware-efficient ansatz (HEA) with data re-uploading
and domain-aware cross entanglement, parameter for parameter:

    |0⟩^14
      → RY(features * scales)         # scaled encoding
      → [CNOT ring + cross pairs + RY(var_L1)]
      → RY(features * scales)         # data re-upload
      → [CNOT ring + cross pairs + RY(var_L2)]
      → ⟨Z_i⟩ per qubit  →  weighted sum + bias  →  sigmoid

Parameter layout (identical to the NumPy reference):

    params[  0 : 14]   encoding scales (one per qubit)
    params[ 14 : 42]   variational RY angles (n_layers * n_qubits = 28)
    params[ 42 : 56]   output weights (one per qubit)
    params[ 56 ]       bias
    total: 57

The companion test ``tests/test_vqc_parity.py`` asserts

    max | get_expectations_numpy(x, θ) − get_expectations_qiskit(x, θ) |  ≤  1e-6

across random ``(θ, x)`` tuples, and the same bound on the forward output.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# ── Canonical cross-entanglement pairs (match NumPy reference exactly) ──

DEFAULT_CROSS_PAIRS_14Q: List[Tuple[int, int]] = [
    # Momentum cluster
    (0, 1),
    # Volatility cluster
    (2, 3),
    (4, 2),
    # Sentiment cluster
    (8, 9),
    (10, 11),
    # Macro-risk cluster
    (5, 6),
    (7, 13),
    (12, 5),
    # Cross-cluster bridges
    (0, 13),
    (8, 2),
]


def _auto_cross_pairs(n_qubits: int) -> List[Tuple[int, int]]:
    """Generate cross-entanglement pairs for n != 14 (parity with NumPy)."""
    pairs: List[Tuple[int, int]] = []
    for i in range(0, n_qubits - 2, 2):
        pairs.append((i, i + 2))
    if n_qubits > 4:
        pairs.append((0, n_qubits - 1))
    return pairs


# ── Circuit builder ────────────────────────────────────────────────

def build_circuit_v4(
    features: Sequence[float],
    var_params: Sequence[float],
    scales: Sequence[float],
    cross_pairs: Sequence[Tuple[int, int]],
    n_qubits: int,
    n_layers: int,
) -> QuantumCircuit:
    """
    Construct the VQC v4 circuit for a single feature vector + parameter set.

    Gate sequence matches :func:`phase0.quantum_signal.circuit_v4._run_circuit`:
    initial scaled encoding → for each layer (CNOT ring, cross CNOTs,
    variational RY, then re-upload if not last layer).

    The returned :class:`QuantumCircuit` has **no measurement**; callers
    compute ⟨Z_i⟩ via :class:`qiskit.quantum_info.Statevector`.
    """
    if len(features) != n_qubits:
        raise ValueError(f"Expected {n_qubits} features, got {len(features)}")
    if len(scales) != n_qubits:
        raise ValueError(f"Expected {n_qubits} scales, got {len(scales)}")
    if len(var_params) != n_qubits * n_layers:
        raise ValueError(
            f"Expected {n_qubits * n_layers} var_params, got {len(var_params)}"
        )

    qc = QuantumCircuit(n_qubits)

    # Initial scaled encoding
    for i in range(n_qubits):
        qc.ry(float(features[i]) * float(scales[i]), i)

    for layer in range(n_layers):
        # CNOT ring: 0→1, 1→2, ..., (n-2)→(n-1), then (n-1)→0
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        qc.cx(n_qubits - 1, 0)

        # Domain-aware cross entanglement
        for control, target in cross_pairs:
            qc.cx(int(control), int(target))

        # Variational RY layer
        offset = layer * n_qubits
        for i in range(n_qubits):
            qc.ry(float(var_params[offset + i]), i)

        # Data re-uploading between layers (not after the last)
        if layer < n_layers - 1:
            for i in range(n_qubits):
                qc.ry(float(features[i]) * float(scales[i]), i)

    return qc


# ── Vectorized ⟨Z_q⟩ extraction from a Qiskit statevector ───────────

def _z_expectations_from_statevector(
    sv: Statevector, n_qubits: int
) -> np.ndarray:
    """
    Compute ⟨Z_q⟩ for every qubit q from a Qiskit Statevector.

    Qiskit uses **little-endian** amplitude indexing: basis state index ``k``
    encodes bit ``q`` at position ``(k >> q) & 1``. For a single-qubit Z
    observable on qubit q,

        ⟨Z_q⟩ = sum_k  p_k  *  (1 − 2 * bit_q(k))

    where ``p_k = |ψ_k|²``. Vectorized over all qubits at once: O(2^n · n).
    """
    probs = np.abs(sv.data) ** 2
    dim = 1 << n_qubits
    indices = np.arange(dim, dtype=np.int64)
    exps = np.zeros(n_qubits, dtype=np.float64)
    for q in range(n_qubits):
        bits = (indices >> q) & 1
        # sign = +1 for bit=0, −1 for bit=1
        exps[q] = float(np.sum(probs * (1 - 2 * bits)))
    return exps


def _sigmoid(x: float) -> float:
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


# ── Qiskit twin of TrainableVQC_v4 ──────────────────────────────────

class TrainableVQC_v4_Qiskit:
    """
    Qiskit port of ``TrainableVQC_v4`` with identical parameter layout.

    The class deliberately implements **only** the forward path
    (expectations, raw logit, sigmoid output). Training (parameter-shift
    gradient descent) is deferred to later steps — Step 2's success
    criterion is forward-parity, not end-to-end training.

    Parameters
    ----------
    n_qubits : int, default 14
    n_layers : int, default 2
    cross_pairs : list of (int, int), optional
        If ``None``, uses :data:`DEFAULT_CROSS_PAIRS_14Q` for ``n_qubits == 14``
        else :func:`_auto_cross_pairs`.
    params : np.ndarray, optional
        If provided, must have length ``n_qubits*(n_layers + 2) + 1``.
        Same layout as the NumPy reference.
    seed : int, default 42
        Used only when ``params is None`` to draw a default initialization
        that matches the NumPy reference's behaviour.
    """

    def __init__(
        self,
        n_qubits: int = 14,
        n_layers: int = 2,
        cross_pairs: Optional[List[Tuple[int, int]]] = None,
        params: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)

        if cross_pairs is not None:
            self.cross_pairs = [tuple(p) for p in cross_pairs]
        elif self.n_qubits == 14:
            self.cross_pairs = list(DEFAULT_CROSS_PAIRS_14Q)
        else:
            self.cross_pairs = _auto_cross_pairs(self.n_qubits)

        # Parameter layout (identical to NumPy reference)
        self.n_encoding_scales = self.n_qubits
        self.n_variational_params = self.n_qubits * self.n_layers
        self.n_output_weights = self.n_qubits
        self.n_total_params = (
            self.n_encoding_scales
            + self.n_variational_params
            + self.n_output_weights
            + 1
        )

        if params is not None:
            if len(params) != self.n_total_params:
                raise ValueError(
                    f"Expected {self.n_total_params} params, got {len(params)}"
                )
            self.params = np.asarray(params, dtype=float).copy()
        else:
            rng = np.random.RandomState(seed)
            self.params = np.zeros(self.n_total_params, dtype=float)

            # Encoding scales: init near π (match NumPy reference)
            self.params[: self.n_encoding_scales] = (
                math.pi + rng.uniform(-0.3, 0.3, self.n_encoding_scales)
            )

            # Variational params: small random init (Xavier-like)
            var_scale = 1.0 / math.sqrt(self.n_qubits)
            self.params[
                self.n_encoding_scales : self.n_encoding_scales
                + self.n_variational_params
            ] = rng.uniform(
                -var_scale, var_scale, self.n_variational_params
            )

            # Output weights: uniform small
            w_start = self.n_encoding_scales + self.n_variational_params
            self.params[w_start : w_start + self.n_output_weights] = (
                rng.uniform(-0.3, 0.3, self.n_output_weights)
            )

            # Bias: zero
            self.params[-1] = 0.0

    # ── views into the flat parameter vector ──────────────────────

    @property
    def encoding_scales(self) -> np.ndarray:
        return self.params[: self.n_encoding_scales]

    @property
    def variational_params(self) -> np.ndarray:
        return self.params[
            self.n_encoding_scales : self.n_encoding_scales
            + self.n_variational_params
        ]

    @property
    def output_weights(self) -> np.ndarray:
        w_start = self.n_encoding_scales + self.n_variational_params
        return self.params[w_start : w_start + self.n_output_weights]

    @property
    def output_bias(self) -> float:
        return float(self.params[-1])

    # ── circuit + expectations ────────────────────────────────────

    def build_circuit(self, features: Sequence[float]) -> QuantumCircuit:
        """Build the QuantumCircuit for ``features`` under current params."""
        return build_circuit_v4(
            features=features,
            var_params=self.variational_params,
            scales=self.encoding_scales,
            cross_pairs=self.cross_pairs,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
        )

    def get_expectations(self, features: Sequence[float]) -> np.ndarray:
        """Return ⟨Z_0⟩ … ⟨Z_{n-1}⟩ as a length-n array."""
        qc = self.build_circuit(features)
        sv = Statevector.from_instruction(qc)
        return _z_expectations_from_statevector(sv, self.n_qubits)

    def forward_raw(self, features: Sequence[float]) -> float:
        """Return the pre-sigmoid logit."""
        exps = self.get_expectations(features)
        return float(np.dot(self.output_weights, exps) + self.output_bias)

    def forward(self, features: Sequence[float]) -> float:
        """Return the probability output (sigmoid of logit)."""
        return _sigmoid(self.forward_raw(features))

    def predict(self, features: Sequence[float]) -> int:
        return 1 if self.forward(features) > 0.5 else 0


__all__ = [
    "DEFAULT_CROSS_PAIRS_14Q",
    "build_circuit_v4",
    "TrainableVQC_v4_Qiskit",
]
