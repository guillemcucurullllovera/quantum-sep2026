"""Dynamical Decoupling (DD) — XY-4 pulse sequence.

Internal Asymmetrica code, not a third-party adaptation. XY-4 suppresses
T2*/ZZ-coupling drift on idle qubits between gates. Provides a numpy
statevector simulator for offline correctness verification, a Qiskit
circuit builder for manual insertion, and a suppression-factor estimate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)
Y_GATE = np.array([[0, -1j], [1j, 0]], dtype=complex)
I_GATE = np.eye(2, dtype=complex)

XY4_SEQUENCE: list[np.ndarray] = [X_GATE, Y_GATE, X_GATE, Y_GATE]
XY4_NAMES: list[str] = ["X", "Y", "X", "Y"]


def apply_xy4_sequence(state: np.ndarray, noise_channel: np.ndarray | None = None) -> np.ndarray:
    if state.shape != (2,):
        raise ValueError(f"Expected single-qubit state shape (2,), got {state.shape}")
    noise_channel = I_GATE if noise_channel is None else noise_channel
    s = state.copy()
    for pulse in XY4_SEQUENCE:
        s = noise_channel @ s
        s = pulse @ s
    return s


def xy4_suppression_factor(dephasing_rate: float, t_idle: float, n_pulses: int = 4, n_cycles: int = 1) -> float:
    if dephasing_rate <= 0 or t_idle <= 0:
        return 1.0
    tau_interval = t_idle / (n_pulses * n_cycles)
    t2 = 1 / dephasing_rate
    return float(min((t2 / tau_interval) ** 0.5, 100.0))


def xy4_is_identity() -> bool:
    result = I_GATE.copy()
    for pulse in XY4_SEQUENCE:
        result = pulse @ result
    off_diag_zero = np.allclose([result[0, 1], result[1, 0]], [0, 0], atol=1e-10)
    diag_equal = np.allclose(result[0, 0], result[1, 1], atol=1e-10)
    unit_norm = np.allclose(abs(result[0, 0]), 1.0, atol=1e-10)
    return bool(off_diag_zero and diag_equal and unit_norm)


def build_xy4_qiskit_circuit(n_qubits: int, idle_qubit: int):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)
    qc.x(idle_qubit)
    qc.y(idle_qubit)
    qc.x(idle_qubit)
    qc.y(idle_qubit)
    return qc


@dataclass
class DDMitigator:
    sequence: str = "XY4"

    def __post_init__(self) -> None:
        if self.sequence != "XY4":
            raise ValueError(f"Only XY4 supported, got: {self.sequence}")
        assert xy4_is_identity(), "XY-4 sequence is NOT identity up to global phase"

    def suppression_estimate(self, t2_us: float = 50.0, t_idle_us: float = 1.0) -> float:
        return xy4_suppression_factor(1 / t2_us, t_idle_us)

    def summary(self) -> dict:
        return {
            "sequence": self.sequence,
            "pulses": XY4_NAMES,
            "net_operation": "Identity up to global phase (noiseless)",
            "t2_improvement_factor": round(self.suppression_estimate(), 2),
        }


__all__ = ["DDMitigator", "apply_xy4_sequence", "xy4_suppression_factor", "xy4_is_identity", "build_xy4_qiskit_circuit"]
