"""Local (tensored) readout-error mitigation.

Independently re-implemented against this platform's own interfaces from the
calibrate-then-invert algorithm used in qiskit-experiments' readout
mitigators (Apache-2.0). See THIRD_PARTY_NOTICES.md — no source lines copied.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import reduce

import numpy as np

from core.errors import ValidationError


@dataclass
class CalibrationMatrix:
    """Per-qubit 2x2 assignment matrices: per_qubit[i][j, k] = P(measured j | prepared k)."""

    per_qubit: list[np.ndarray]

    @classmethod
    def from_counts(
        cls, prep0_counts: list[dict[str, int]], prep1_counts: list[dict[str, int]]
    ) -> "CalibrationMatrix":
        """Build calibration from per-qubit |0> and |1> preparation-and-measure
        count dictionaries (single-bit key '0'/'1' -> shots), one pair per qubit.
        """
        if len(prep0_counts) != len(prep1_counts):
            raise ValidationError("prep0_counts and prep1_counts must have the same length")
        matrices = []
        for c0, c1 in zip(prep0_counts, prep1_counts):
            n0, n1 = sum(c0.values()), sum(c1.values())
            if n0 == 0 or n1 == 0:
                raise ValidationError("Calibration counts must have at least one shot per state")
            col0 = np.array([c0.get("0", 0) / n0, c0.get("1", 0) / n0])
            col1 = np.array([c1.get("0", 0) / n1, c1.get("1", 0) / n1])
            matrices.append(np.column_stack([col0, col1]))
        return cls(per_qubit=matrices)

    def full_matrix(self) -> np.ndarray:
        return reduce(np.kron, self.per_qubit)

    @property
    def num_qubits(self) -> int:
        return len(self.per_qubit)


class LocalReadoutMitigator:
    """Corrects measured bitstring counts via least-squares pseudo-inverse of
    the tensored calibration matrix (robust to near-singular calibration
    matrices, unlike a naive exact inverse).
    """

    def __init__(self, calibration: CalibrationMatrix) -> None:
        self.calibration = calibration
        self._A = calibration.full_matrix()
        self._A_pinv = np.linalg.pinv(self._A)

    def mitigate_counts(self, counts: dict[str, int]) -> dict[str, float]:
        """Return mitigated quasi-probabilities. May contain small negative
        values, as is standard for matrix-inversion mitigation — this method
        deliberately does not silently clip/renormalize so callers can
        inspect the raw mitigation artifact; use `clip_to_valid_distribution`
        if a proper probability simplex is required downstream.
        """
        n = self.calibration.num_qubits
        total = sum(counts.values())
        if total == 0:
            raise ValidationError("counts is empty")
        noisy = np.zeros(2**n)
        for bitstring, c in counts.items():
            noisy[int(bitstring, 2)] = c / total
        mitigated = self._A_pinv @ noisy
        return {format(i, f"0{n}b"): float(p) for i, p in enumerate(mitigated)}

    def mitigated_expectation(self, counts: dict[str, int], observable: dict[str, float]) -> float:
        """observable: bitstring -> eigenvalue, e.g. {'00': 1.0, '01': -1.0, ...}."""
        probs = self.mitigate_counts(counts)
        return float(sum(probs.get(b, 0.0) * v for b, v in observable.items()))


def clip_to_valid_distribution(quasi_probs: dict[str, float]) -> dict[str, float]:
    """Clip negative quasi-probabilities to zero and renormalize to sum to 1."""
    clipped = {k: max(v, 0.0) for k, v in quasi_probs.items()}
    total = sum(clipped.values())
    if total == 0:
        raise ValidationError("All quasi-probabilities were non-positive; cannot renormalize")
    return {k: v / total for k, v in clipped.items()}
