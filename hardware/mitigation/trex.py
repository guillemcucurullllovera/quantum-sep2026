"""TREX (Twirled Readout Error eXpectation) mitigation — van den Berg 2022.

Internal Asymmetrica code. Eliminates asymmetric readout bias via
bit-flip twirling + classical correction; O(n) extra circuits vs O(2^n)
for full readout calibration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def apply_bitflip_twirl(state: np.ndarray, bitflip_string: np.ndarray) -> np.ndarray:
    n = int(np.log2(len(state)))
    if len(bitflip_string) != n:
        raise ValueError(f"bitflip_string must have length n={n}")
    x_gate = np.array([[0, 1], [1, 0]], dtype=complex)
    i_gate = np.eye(2, dtype=complex)
    op = np.array([[1.0]], dtype=complex)
    for i in range(n):
        op = np.kron(op, x_gate if bitflip_string[i] else i_gate)
    return op @ state


def correct_bitflip_twirl(counts: np.ndarray, bitflip_string: np.ndarray) -> np.ndarray:
    n = len(bitflip_string)
    d = 2 ** n
    flip_mask = sum(int(bitflip_string[i]) << i for i in range(n))
    corrected = np.zeros(d, dtype=counts.dtype)
    for k in range(d):
        corrected[k ^ flip_mask] = counts[k]
    return corrected


def trex_calibrate(n: int, readout_error_0: float = 0.007, readout_error_1: float = 0.009) -> np.ndarray:
    """Build the n-qubit TREX calibration matrix A = A_1 ⊗ ... ⊗ A_n, where
    A_i = [[1-e0, e1], [e0, 1-e1]] for one qubit's asymmetric readout error.
    """
    a_single = np.array([[1 - readout_error_0, readout_error_1], [readout_error_0, 1 - readout_error_1]])
    a = np.array([[1.0]])
    for _ in range(n):
        a = np.kron(a, a_single)
    return a


def trex_apply_correction(raw_counts: np.ndarray, calibration_matrix: np.ndarray, regularization: float = 1e-6) -> np.ndarray:
    n_shots = raw_counts.sum()
    if n_shots == 0:
        return raw_counts.astype(float)
    p_noisy = raw_counts.astype(float) / n_shots
    a_reg = calibration_matrix + regularization * np.eye(len(p_noisy))
    try:
        p_ideal = np.linalg.solve(a_reg, p_noisy)
    except np.linalg.LinAlgError:
        return p_noisy
    p_ideal = np.clip(p_ideal, 0, None)
    s = p_ideal.sum()
    return p_ideal / s if s > 1e-12 else p_ideal


def trex_z_expectations(corrected_probs: np.ndarray, n: int) -> np.ndarray:
    basis = np.arange(len(corrected_probs))
    z_exp = np.zeros(n)
    for i in range(n):
        bits = (basis >> i) & 1
        z_exp[i] = float(np.dot(corrected_probs, 1 - 2 * bits))
    return z_exp


@dataclass
class TREXMitigator:
    n_qubits: int
    readout_error_0: float = 0.007
    readout_error_1: float = 0.009
    n_twirl_circuits: int = 128

    def __post_init__(self) -> None:
        self._calibration_matrix = trex_calibrate(self.n_qubits, self.readout_error_0, self.readout_error_1)

    def correct(self, raw_counts: np.ndarray) -> np.ndarray:
        return trex_apply_correction(raw_counts, self._calibration_matrix)

    def z_expectations(self, raw_counts: np.ndarray) -> np.ndarray:
        return trex_z_expectations(self.correct(raw_counts), self.n_qubits)

    def summary(self) -> dict:
        return {
            "method": "TREX (van den Berg 2022)",
            "n_qubits": self.n_qubits,
            "readout_error_0": self.readout_error_0,
            "readout_error_1": self.readout_error_1,
            "n_twirl_circuits": self.n_twirl_circuits,
        }


__all__ = ["TREXMitigator", "trex_calibrate", "trex_apply_correction", "trex_z_expectations",
           "apply_bitflip_twirl", "correct_bitflip_twirl"]
