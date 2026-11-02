"""Circuit-resource estimates across algorithm families, using
hardware.circuit_analysis for depth/gate counts, plus a real
shots-for-target-confidence-interval estimator (standard Wald-interval
sample-size formula) for the QAE/IQAE amplitude-estimation family, where
"how many shots for a given precision" is an actual operational question,
not just a diagnostic.
"""
from __future__ import annotations

import math

import numpy as np

from algorithms.qaoa_qubo.solvers import _build_qiskit_qaoa_circuit
from qiskit import QuantumCircuit

from algorithms.pqk.feature_maps import N_QUBITS
from hardware.circuit_analysis.metrics import CircuitStats, analyze


def qaoa_circuit_resources(n_qubits: int = 6, p: int = 2, seed: int = 0) -> CircuitStats:
    rng = np.random.default_rng(seed)
    J = rng.uniform(-1, 1, size=(n_qubits, n_qubits))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)
    h = rng.uniform(-1, 1, size=n_qubits)
    gamma, beta = np.full(p, 0.5), np.full(p, 0.3)
    circuit = _build_qiskit_qaoa_circuit(J, h, p=p, gamma=gamma, beta=beta)
    return analyze(circuit)


def pqk_feature_map_resources(features: np.ndarray | None = None, reps: int = 2) -> CircuitStats:
    """Builds the AngleCNOT feature-map circuit directly (mirroring
    algorithms.pqk.feature_maps.AngleCNOTMap_Qiskit.encode's construction)
    rather than calling .encode(), which returns a Statevector -- resource
    analysis needs the QuantumCircuit object itself, before simulation.
    """
    x = features if features is not None else np.random.default_rng(0).uniform(-1, 1, size=N_QUBITS)
    qc = QuantumCircuit(N_QUBITS)
    for _ in range(reps):
        for i in range(N_QUBITS):
            qc.ry(float(x[i]) * np.pi, i)
        for i in range(N_QUBITS - 1):
            qc.cx(i, i + 1)
        qc.cx(N_QUBITS - 1, 0)
    return analyze(qc)


def shots_for_target_confidence_interval(
    estimated_probability: float, target_half_width: float, confidence: float = 0.95
) -> int:
    """Standard Wald-interval sample-size formula: n = z^2 * p(1-p) / e^2.
    A real, checkable formula (not a heuristic) -- z=1.96 for 95% confidence.
    """
    z_by_confidence = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_by_confidence.get(round(confidence, 2))
    if z is None:
        raise ValueError(f"Unsupported confidence level {confidence}; use one of {sorted(z_by_confidence)}")
    p = estimated_probability
    n = (z**2) * p * (1 - p) / (target_half_width**2)
    return math.ceil(n)


def resource_summary(n_qubits: int = 6, p: int = 2, seed: int = 0) -> dict:
    qaoa_stats = qaoa_circuit_resources(n_qubits, p, seed)
    pqk_stats = pqk_feature_map_resources()
    return {
        "qaoa": qaoa_stats.__dict__,
        "pqk": pqk_stats.__dict__,
        "qae_shots_for_1pct_ci_at_p0.1": shots_for_target_confidence_interval(0.1, 0.01),
        "qae_shots_for_1pct_ci_at_p0.5": shots_for_target_confidence_interval(0.5, 0.01),
    }
