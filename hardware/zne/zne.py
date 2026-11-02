"""Zero-Noise Extrapolation (ZNE) — pure-numpy Richardson extrapolation.

Internal Asymmetrica code, independently implemented from the published
general method (Temme et al. 2017), not adapted from any third-party
library. See `hardware.mitigation.__init__` for why this platform does
not depend on Mitiq.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def _richardson_coefficients(noise_factors: list[int]) -> np.ndarray:
    n = len(noise_factors)
    a = np.array([[lam ** j for lam in noise_factors] for j in range(n)], dtype=float)
    b = np.zeros(n)
    b[0] = 1.0
    return np.linalg.solve(a, b)


def richardson_extrapolate(values: np.ndarray, noise_factors: list[int] | None = None) -> float:
    n = len(values)
    noise_factors = noise_factors or list(range(1, n + 1))
    if len(noise_factors) != n:
        raise ValueError("len(values) must equal len(noise_factors)")
    coeffs = _richardson_coefficients(noise_factors)
    return float(np.dot(coeffs, values))


def linear_extrapolate(values: np.ndarray, noise_factors: list[int] | None = None) -> float:
    if len(values) < 2:
        return float(values[0])
    noise_factors = noise_factors or list(range(1, len(values) + 1))
    lam0, lam1 = float(noise_factors[0]), float(noise_factors[1])
    e0, e1 = float(values[0]), float(values[1])
    slope = (e1 - e0) / (lam1 - lam0)
    return e0 - slope * lam0


def fold_circuit_numpy(unitary: np.ndarray, scale_factor: int) -> np.ndarray:
    """Noise amplification via unitary folding: U -> U (U^dagger U)^k where
    k=(scale_factor-1)/2. For hardware use, fold at the Qiskit-circuit level
    (repeat gate-inverse-gate blocks) rather than on the dense unitary —
    this numpy version is for controlled small-scale verification."""
    if scale_factor < 1 or (scale_factor - 1) % 2 != 0:
        raise ValueError(f"scale_factor must be an odd integer >= 1, got {scale_factor}")
    k = (scale_factor - 1) // 2
    result = unitary.copy()
    fold = unitary.conj().T @ unitary
    for _ in range(k):
        result = result @ fold
    return result


def fold_circuit_gates(circuit, scale_factor: int):
    """Gate-level folding for a real Qiskit circuit: appends (gate, gate^-1)
    pairs `k` times after the original circuit, which is the standard
    circuit-level ZNE noise-scaling technique (as opposed to the
    dense-unitary version above, which is for small verification cases)."""
    if scale_factor < 1 or (scale_factor - 1) % 2 != 0:
        raise ValueError(f"scale_factor must be an odd integer >= 1, got {scale_factor}")
    k = (scale_factor - 1) // 2
    folded = circuit.copy()
    for _ in range(k):
        folded.compose(circuit.inverse(), inplace=True)
        folded.compose(circuit, inplace=True)
    return folded


@dataclass
class ZNEMitigator:
    noise_factors: list[int] | None = None
    extrapolator: str = "richardson"

    def __post_init__(self) -> None:
        self.noise_factors = self.noise_factors or [1, 2, 3]
        if self.extrapolator not in ("richardson", "linear"):
            raise ValueError("extrapolator must be 'richardson' or 'linear'")

    def mitigate_from_values(self, values: list[float]) -> float:
        vals = np.array(values, dtype=float)
        if self.extrapolator == "richardson":
            return richardson_extrapolate(vals, self.noise_factors)
        return linear_extrapolate(vals, self.noise_factors)

    def mitigate(self, run_fn: Callable[[int], float]) -> float:
        values = [run_fn(lam) for lam in self.noise_factors]
        return self.mitigate_from_values(values)

    def mitigation_overhead(self) -> float:
        coeffs = _richardson_coefficients(self.noise_factors)
        return float(np.sum(np.abs(coeffs)) ** 2)

    def summary(self) -> dict:
        coeffs = _richardson_coefficients(self.noise_factors).tolist()
        return {
            "extrapolator": self.extrapolator,
            "noise_factors": self.noise_factors,
            "richardson_coefficients": [round(c, 6) for c in coeffs],
            "shot_overhead": round(self.mitigation_overhead(), 2),
        }


__all__ = ["ZNEMitigator", "richardson_extrapolate", "linear_extrapolate", "fold_circuit_numpy", "fold_circuit_gates"]
