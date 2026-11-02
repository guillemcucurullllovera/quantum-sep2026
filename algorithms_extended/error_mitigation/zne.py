"""Zero-Noise Extrapolation (ZNE).

Original implementation of the published algorithm from Temme, Bravyi &
Gambetta, "Error Mitigation for Short-Depth Quantum Circuits", Phys. Rev.
Lett. 119, 180509 (2017), arXiv:1612.02058 — general, non-copyrightable
technique description. No third-party ZNE library source (Mitiq or
otherwise) was read or adapted; see THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from core.errors import ValidationError


def fold_circuit_globally(circuit: QuantumCircuit, scale_factor: int) -> QuantumCircuit:
    """Global unitary folding: U -> U (U^dagger U)^k, giving noise scale
    factor (2k+1) for k = 0, 1, 2, .... `scale_factor` must be an odd
    positive integer.
    """
    if scale_factor < 1 or scale_factor % 2 == 0:
        raise ValidationError(f"scale_factor must be an odd positive integer, got {scale_factor}")
    k = (scale_factor - 1) // 2
    folded = circuit.copy()
    inverse = circuit.inverse()
    for _ in range(k):
        folded.compose(inverse, inplace=True)
        folded.compose(circuit, inplace=True)
    return folded


def zero_noise_extrapolate(
    scale_factors: list[float], expectation_values: list[float], method: str = "linear"
) -> float:
    """Extrapolate (noise_scale, measured_expectation) points to the
    zero-noise limit.

    method='linear': least-squares line fit, evaluated at x=0. Robust to
    statistical noise across repeated shots.
    method='richardson': exact polynomial extrapolation through every point
    (Richardson extrapolation) — matches the folded data exactly but is more
    sensitive to shot noise than 'linear'.
    """
    x = np.asarray(scale_factors, dtype=float)
    y = np.asarray(expectation_values, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        raise ValidationError("Need at least 2 matching (scale_factor, expectation_value) points")

    if method == "linear":
        coeffs = np.polyfit(x, y, deg=1)
    elif method == "richardson":
        coeffs = np.polyfit(x, y, deg=len(x) - 1)
    else:
        raise ValidationError(f"Unknown extrapolation method: {method!r}")
    return float(np.polyval(coeffs, 0.0))
